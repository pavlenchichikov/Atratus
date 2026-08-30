"""Auto-research loop (local). The agent proposes feature variants
from the DSL, the harness A/B-tests them base vs variant on a selection subset,
and winners are checked on a held-out subset and flagged for a human. It never
retrains production.

Proposer is autonomous (evolutionary search, no LLM) by default. Set
GTRADE_AR_PROPOSER=llm to use the LLM proposer instead; the LLM layer lives in
core/llm_proposer.py and supports anthropic (default) and openai providers
(ollama arrives in the next task). GTRADE_AR_SEED makes the evolutionary search
reproducible.

Run:  python auto_research.py
      GTRADE_AR_PROPOSER=llm python auto_research.py
"""
import bisect
import copy
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

# .env holds this run's LLM settings (GTRADE_AR_LLM_TIMEOUT above all: a local
# 26B model needs far longer than the 600s SDK default, and without this the
# file was read by push_signals and ab_genomes but never by the agent that
# actually calls the model - the configured timeout was silently ignored and
# every call died at ten minutes). load_dotenv does not override variables that
# are already set, so the launcher menu still wins over the file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from core import ar_memory, ar_rl, ar_wiki, llm_proposer, qd_surrogate
from core.feature_dsl import validate_spec
from core.logger import get_logger

logger = get_logger("auto_research")

BASE = os.path.dirname(os.path.abspath(__file__))
SELECTION_ASSETS = "SP500,NVDA,BTC,ETH,EURUSD,GBPJPY,GAS,AAPL,SBER,DAX"

# A cheaper search set: half the assets, one per class kept, so a cycle costs
# about half. It is NOT a free speedup - fewer assets means a noisier selection
# signal and an objective computed over five numbers instead of ten - which is
# why it is a named choice and not a knob with a number on it.
FAST_SELECTION = "SP500,BTC,EURUSD,NVDA,GAS"


def selection_assets():
    """GTRADE_AR_SELECTION: 'full' (default), 'fast', or a verbatim comma list.

    The set the SEARCH scores candidates on. Not the holdout: adoption is still
    decided on heldout_assets(), so narrowing this trades selection precision
    for speed and never touches what a result has to clear.
    """
    v = (os.getenv("GTRADE_AR_SELECTION") or "full").strip()
    if v.lower() == "full":
        return SELECTION_ASSETS
    if v.lower() == "fast":
        return FAST_SELECTION
    return v

# The production holdout: a deliberately mixed set, so an adoption decision is
# taken on assets that look like the book as a whole.
PROD_HELDOUT = "MSFT,GOLD,USDJPY,ADA,CAC40,XOM,GOOGL,SOL,SILVER,GBPUSD,NASDAQ,DXY,TNX,AIRBUS"

# The NEURAL-DIAGNOSTIC holdout: the assets whose stacker actually leans on the
# neural members. Measured 2026-08-14 from the |coef| shares in models/*_meta.pkl
# (share of the stacker's absolute weight held by lstm+tf+tcn):
#
#     AFLT .69  IMOEX .62  LKOH .61  SBER .54  DOW .54  ...
#     PROD_HELDOUT median .17, with ADA .05, MSFT .08, TNX .09, GOOGL .09
#
# A neural change measured on PROD_HELDOUT is measured mostly where the nets are
# nearly irrelevant, which throws away effect size for nothing. This set exists
# to make a neural effect VISIBLE, not to decide adoption: it is biased by
# construction and over-represents RU equities, so a winner here still has to
# clear the production holdout before it means anything.
NEURAL_HELDOUT = "AFLT,IMOEX,LKOH,SBER,DOW,SILVER,XOM,GBPUSD,NASDAQ,BTC,EURUSD,AAPL,GAS,SOL"


def heldout_assets():
    """GTRADE_AR_HELDOUT: 'prod' (default) or 'neural' - see the two lists above.
    Anything else is taken verbatim as a comma-separated asset list."""
    v = (os.getenv("GTRADE_AR_HELDOUT") or "prod").strip()
    if v.lower() == "prod":
        return PROD_HELDOUT
    if v.lower() == "neural":
        return NEURAL_HELDOUT
    return v


HELDOUT_ASSETS = PROD_HELDOUT   # back-compat alias; live code calls heldout_assets()
BUDGET = int(os.getenv("AR_BUDGET", "15"))
ADOPT_MEAN_SCORE_DELTA = 0.5

PROGRESS_KEEP = 12          # measurements kept per key, newest last
# Unit-kind wall times measured on the 2026-07-23 run, seeded so the first run
# after this change can already estimate instead of saying "no history yet".
# Per-asset service times are NOT seedable: the console only shows completion
# stamps of overlapping workers, so assets starts empty and fills as runs go.
PROGRESS_SEED = {"holdout_14": [35765, 37058, 46714, 29930],
                 "tier_4": [4183, 3817, 3735, 3533],
                 "screen_10": [90, 107, 162, 90],
                 "screen_14": [222, 94, 215, 116],
                 "assets": {}}

# The workload knobs (folds, epochs, promotion) are hardware-independent and stay
# here. The CONCURRENCY knobs deliberately do NOT: GTRADE_LIGHT asks train_hybrid
# for its own light profile, because only the child knows whether TF found a GPU.
# Hard-coding workers/slots/threads here pinned CatBoost to a single core even on
# the GPU box, which is how a 12-thread machine ended up running at 1.5 cores.
LIGHT_ENV = {
    "GTRADE_LIGHT": "1", "GTRADE_MAX_FOLDS": "5", "GTRADE_ADAPTIVE_NETS": "1",
    "GTRADE_NET_CAP": "80", "GTRADE_EPOCHS_LSTM": "90", "GTRADE_EPOCHS_TF": "60",
    "GTRADE_EPOCHS_TCN": "50", "GTRADE_FORCE_PROMOTE": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
}


def _reduce_deltas(deltas, objective):
    """Reduce per-asset Score deltas to one objective value. 'mean' (average lift) and
    'min' (lift-the-floor) are the originals; the diversifiers are 'median' (robust
    average), 'cvar' (mean of the worst quartile - a softer, less-noisy floor than min),
    'trimmed_mean' (average without the single best/worst), and 'sharpe' (mean/std =
    consistency; a DIFFERENT, dimensionless scale gated by GTRADE_AR_ADOPT_SHARPE, not
    the Score-delta floor). Unknown - mean."""
    n = len(deltas)
    if n == 0:
        return 0.0
    if objective == "min":
        return min(deltas)
    if objective == "median":
        return statistics.median(deltas)
    if objective == "cvar":
        k = max(1, math.ceil(n * 0.25))
        return sum(sorted(deltas)[:k]) / k
    if objective == "trimmed_mean":
        if n < 4:
            return sum(deltas) / n
        core = sorted(deltas)[1:-1]
        return sum(core) / len(core)
    if objective == "sharpe":
        sd = statistics.pstdev(deltas) if n > 1 else 0.0
        return statistics.mean(deltas) / (max(1e-09, sd))
    return sum(deltas) / n


def _objective_delta(var_rows, base_score, objective="mean"):
    """Paired (variant minus base) Score deltas over shared assets, reduced by the
    objective (see _reduce_deltas). Returns (value, deltas)."""
    e = {r["Asset"]: r.get("Score", 0.0) for r in var_rows}
    common = sorted(set(e) & set(base_score))
    if not common:
        return 0.0, []
    deltas = [e[a] - base_score[a] for a in common]
    return _reduce_deltas(deltas, objective), deltas


def _mean_delta(var_rows, base_score):
    """Mean paired Score delta (backward-compatible wrapper over _objective_delta)."""
    return _objective_delta(var_rows, base_score, "mean")


def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up FDR. Returns a bool per input p-value (significant)
    in the ORIGINAL order. Empty input returns []."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    thresh_rank = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= (rank / m) * alpha:
            thresh_rank = rank
    if thresh_rank == 0:
        return [False] * m
    cutoff = pvals[order[thresh_rank - 1]]
    return [pvals[i] <= cutoff for i in range(m)]


def _sign_test_p(deltas):
    """One-sided sign-test p-value: P(X >= k) under a fair coin, k = assets improved.
    Small p means the improvement is unlikely to be chance across the held-out set."""
    n = len(deltas)
    if n == 0:
        return 1.0
    k = sum(1 for d in deltas if d > 0)
    return sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)


def _wilcoxon_p(deltas):
    """One-sided Wilcoxon signed-rank p that the per-asset deltas are > 0 - a
    magnitude-aware improvement test (a consistent small edge across most assets
    passes even if a few are slightly negative), far more powerful than the sign
    test at this n. Returns 1.0 (no evidence) on inputs scipy cannot test: fewer
    than 2 non-zero deltas, or any scipy error."""
    nz = [d for d in deltas if d != 0]
    if len(nz) < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(deltas, alternative="greater").pvalue)
    except Exception:
        return 1.0


_OBJECTIVES = ("mean", "min", "median", "cvar", "sharpe", "trimmed_mean")


def _objective():
    """The search/gate objective (see _reduce_deltas): mean (default) / min / median /
    cvar / sharpe / trimmed_mean."""
    o = (os.getenv("GTRADE_AR_OBJECTIVE") or "mean").strip().lower()
    if o not in _OBJECTIVES:
        logger.warning("unknown GTRADE_AR_OBJECTIVE %r, using mean", o)
        return "mean"
    return o


def _adopt_floor(objective="mean", basis=None):
    """The practical-effect floor the objective value must beat to adopt.

    The floor follows the UNITS of what is being reduced, so it depends on the
    basis as well as the objective. On net_auc the values are areas under a
    curve near 0.5, where a real move is a few thousandths; the Score floor of
    0.5 there would demand an AUC gain no model can produce and reject
    everything. Score-scale objectives use ADOPT_MEAN_SCORE_DELTA; the
    dimensionless 'sharpe' uses GTRADE_AR_ADOPT_SHARPE (default 0.5).

    `basis` defaults to the SEARCH basis, which is what every search-time caller
    wants. An adoption passes decision_basis() so the floor is in the units the
    verdict is actually read in."""
    if (basis or _score_basis()) in ("net_auc", "net_gain", "ens_auc"):
        try:
            return float(os.getenv("GTRADE_AR_ADOPT_AUC") or "0.005")
        except ValueError:
            return 0.005
    if objective == "sharpe":
        try:
            return float(os.getenv("GTRADE_AR_ADOPT_SHARPE") or "0.5")
        except ValueError:
            return 0.5
    return ADOPT_MEAN_SCORE_DELTA


def neural_floor():
    """How far the neural members are allowed to fall for a candidate to still be
    adoptable, as a Score delta (default: one adopt floor down, i.e. -0.5).

    Always on the plain Score scale, never the objective's: neural_lift is
    computed with the "mean" reduction whatever GTRADE_AR_OBJECTIVE is, so the
    dimensionless 'sharpe' floor would be comparing two different units.

    Switched OFF on net_auc and net_gain. There the objective already IS the
    neural read-out, so the clause would be redundant - and worse, harmful:
    neural_lift is a Score difference and carries the Score's instability, so it
    would veto good candidates on noise the basis was chosen to escape.

    NOT switched off on ens_auc, which reads like a net basis and is not one.
    Measured over 160 champions, Ens_AUC tracks CB_AUC at rho 0.869 and the nets
    at 0.680: the ensemble is mostly CatBoost, so a genome can raise it while
    starving the sequence members - exactly what this clause exists to catch.
    With it disabled there, every one of the 17 genomes an ens_auc campaign
    flagged carried a NEGATIVE neural_lift, down to -2.38."""
    if _score_basis() in ("net_auc", "net_gain"):
        return float("-inf")
    try:
        return float(os.getenv("GTRADE_AR_NEURAL_FLOOR")
                     or str(-ADOPT_MEAN_SCORE_DELTA))
    except ValueError:
        return -ADOPT_MEAN_SCORE_DELTA


def adopt_ok(significant, value, objective, neural_lift=None):
    """The shared adoption decision: statistically significant, over the practical
    floor, and not paid for by killing the nets.

    The neural clause is the point. The search fitness is the CatBoost-only screen
    (see the note in run_qd), so a genome is selected purely on CatBoost and is free
    to win by starving the sequence members - which is what the 2026-08 elite did,
    tagged ADOPTABLE while carrying neural_lift -2.38. Without this clause
    neural_lift was reported and then ignored. neural_lift=None (not measured)
    never blocks."""
    return bool(significant and value > _adopt_floor(objective)
                and (neural_lift is None or neural_lift > neural_floor()))


def holdout_stats(base_rows, ext_rows, objective="mean"):
    """Raw held-out stats for a variant: (wilcoxon p, objective value, deltas, tag).
    No adoption decision - main applies BH across the axis-winners."""
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in base_rows}
    value, deltas = _objective_delta(ext_rows, base_score, objective)
    if not deltas:
        return 1.0, 0.0, [], "no common held-out assets"
    p = _wilcoxon_p(deltas)
    up = sum(1 for d in deltas if d > 0)
    tag = "%s dScore %.2f, wilcoxon p=%.3f (%d/%d up)" % (objective, value, p, up, len(deltas))
    return p, value, deltas, tag


PROMOTION_MARGIN = 0.2      # train_hybrid: score > champion + 0.2 or no change


def promotion_stats(base_rows, var_rows, margin=PROMOTION_MARGIN, column="Score"):
    """The champion-challenger decision, counted on a held-out set.

    Every other statistic here reduces a set of assets to a mean. Production
    never sees that mean: it walks the assets one at a time and keeps the
    champion unless the challenger beats it by `margin`. The two can disagree
    completely, and on 2026-08-18 they did - an A/B passed on a mean of +0.036
    while the same rows held 3 promotions against 10 demotions, which is what
    the ten-hour retrain then went and rediscovered asset by asset.

    Deliberately on the raw Score column whatever the basis is, because Score is
    the quantity train_hybrid compares. A basis is a way of reading a search; a
    promotion is a fact about production.

    The sign test is one-sided over promoted against demoted, and the assets
    inside the margin are excluded rather than counted as ties, which is the
    same shape as the promotion rule itself.
    """
    from scipy.stats import binomtest

    base = {r.get("Asset"): r.get(column) for r in base_rows or []}
    promoted = demoted = 0
    for row in var_rows or []:
        was, now = base.get(row.get("Asset")), row.get(column)
        if was is None or now is None:
            continue
        if now > was + margin:
            promoted += 1
        elif now < was - margin:
            demoted += 1
    n = promoted + demoted
    p = 1.0 if not n else float(
        binomtest(promoted, n, 0.5, alternative="greater").pvalue)
    return {"promoted": promoted, "demoted": demoted, "n": n, "p": p}


def promotion_tag(st):
    """One line for a console verdict, empty when nothing was comparable."""
    if not st or not st["n"]:
        return ""
    return "would promote %d, demote %d (sign p=%.3f)" % (
        st["promoted"], st["demoted"], st["p"])


def is_adoptable(base_rows, ext_rows, n_experiments, budget, alpha=0.05, objective="mean"):
    """Single-test adoption (kept for ab_labeling and single-axis use): significant
    (sign-test p < alpha) AND practically meaningful (objective value over the
    threshold), within budget."""
    if n_experiments > budget:
        return False, "over iteration budget (%d > %d)" % (n_experiments, budget)
    p, value, deltas, tag = holdout_stats(base_rows, ext_rows, objective)
    if not deltas:
        return False, "no common held-out assets"
    if p < alpha and value > _adopt_floor(objective):
        return True, tag
    return False, tag + " (below bar)"


def _train(subset, env_overrides, model_dir):
    env = dict(os.environ)
    env.update(LIGHT_ENV)
    env["GTRADE_ASSETS"] = subset
    env["GTRADE_MODEL_DIR"] = model_dir
    # A genome is an ABSOLUTE specification, so research children must never
    # inherit a production adoption. This module does not import config, but the
    # train_hybrid child does, and config fills in any GTRADE_* key the genome
    # left unset - which would silently train every candidate as candidate plus
    # the adopted genome while genome_sig recorded the candidate alone, and would
    # compare it against a base cached before the adoption.
    env["GTRADE_ADOPTED_PATH"] = os.path.join(BASE, "_no_adoption.json")
    env.update(env_overrides)
    subprocess.run([sys.executable, "train_hybrid.py"], cwd=BASE, env=env,
                   check=False)
    path = os.path.join(model_dir, "quality_report.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _train_once(subset, env_overrides):
    """ONE train_hybrid process over the whole subset; rows back, [] on failure.

    The temp model dir is removed after the rows are read back, so long runs
    (e.g. a large re-gate) do not leak thousands of ar_* dirs into %TEMP%."""
    tmp = tempfile.mkdtemp(prefix="ar_")
    try:
        return _train(subset, dict(env_overrides), os.path.join(tmp, "run"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def train_env(subset, env_overrides, split=True):
    """Train the subset and return the quality rows. The primitive EVERY path uses.

    The split lives here, and that is the point. It used to live in
    _cached_train, where it did almost nothing: the search hands `train_env`
    itself to run_axis as the trainer, and run_qd's illumination calls it too,
    so every candidate evaluation went straight past the chunking. Only the base
    training was ever parallel - minutes of the run, while the hours went down
    the one path the split did not cover. Observed on 2026-08-17: the 10-asset
    base came back "2 chunks on 2 processes", and every 4-asset candidate after
    it ran alone on the full pool.

    split=False is for a caller that has ALREADY chunked (_cached_train, which
    chunks to cache per chunk), so a chunk is never split a second time.
    """
    if not split:
        return _train_once(subset, env_overrides)
    chunks = chunk_subsets(subset, effective_chunk_size(subset))
    jobs = min(train_jobs(), len(chunks))
    if jobs <= 1:
        return _train_once(subset, env_overrides)
    done = _train_chunks_parallel(chunks, env_overrides, jobs, "train")
    rows = []
    for part in chunks:
        rows.extend(done.get(part) or [])
    return rows


def train_chunk_size():
    """Assets per training chunk, or 0 for the whole subset in one process.

    0 by default so every existing path stays byte-identical; the unattended
    loop turns it on.
    """
    try:
        return max(0, int(os.getenv("GTRADE_AR_TRAIN_CHUNK", "0") or 0))
    except ValueError:
        return 0


def chunk_subsets(subset, size):
    """Split a comma-separated subset into training chunks, order preserved."""
    assets = [a.strip() for a in subset.split(",") if a.strip()]
    if not assets:
        return []
    if size <= 0 or size >= len(assets):
        return [",".join(assets)]
    return [",".join(assets[i:i + size]) for i in range(0, len(assets), size)]


# Memory a training process holds OUTSIDE the TF pool: the CUDA context plus
# cuDNN workspaces. MEASURED 2026-08-29 on a free card, two chunk processes at
# a 0.17 share each: pool 673 MiB apiece, peak 3682 MiB of 4096, so the part
# outside the pool is (3682 - 1346) / 2 = 1168 per process. The earlier value
# here was 950, inferred from the 2026-08-17 total rather than measured, and it
# understated the real figure by 23 percent - which made gpu_fit_jobs promise
# room the card does not have on a partly occupied GPU.
VRAM_OVERHEAD_MB = 1170


def free_vram_mb():
    """Free VRAM in MiB, or None when the card cannot be asked."""
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    return None


def gpu_fit_jobs(jobs, pool_pct=None, free_mb=None):
    """How many training processes the card can actually take right now.

    The 2026-08-24 failure was not an exception. Two processes on a card with
    140 MiB of headroom did not OOM and did not raise; they ran for 15000 s
    without finishing a single asset. Nothing checked first, so there was
    nothing to report and no way to tell a stall from slow progress.

    This is that check. It is deliberately arithmetic and pessimistic: pool
    plus a measured per-process overhead, against what the card says is free.
    Returning fewer jobs than asked costs wall clock; not returning fewer costs
    a night.
    """
    if jobs <= 1:
        return jobs
    free = free_mb if free_mb is not None else free_vram_mb()
    if not free:
        return jobs          # no card or no answer: leave the caller's choice
    try:
        pct = float(pool_pct if pool_pct is not None
                    else (os.getenv("GTRADE_TF_POOL_PCT") or "0.34"))
    except (TypeError, ValueError):
        pct = 0.34
    per_process = max(640, int(free * pct / jobs)) + VRAM_OVERHEAD_MB
    fits = max(1, int(free // per_process))
    return min(jobs, fits)


def apply_manual_load_profile():
    """Give a hand-started run the same load profile the campaign runs at.

    auto_loop.LOAD_KEYS was named for exactly this - "so a phase started BY HAND
    can borrow the same profile" - and nothing here ever called it. The cost was
    not a slower run but a silently different one: GTRADE_AR_TRAIN_CHUNK
    defaults to 0, at which chunk_subsets returns a single chunk, jobs collapses
    to min(train_jobs(), 1), and GTRADE_AR_TRAIN_JOBS=2 means nothing. So every
    manual run trained sequentially at a 60 percent pool with 6 undivided
    workers, while the campaign ran two processes at 34 percent with 3 each, and
    the log gave no sign of it beyond those numbers.

    setdefault semantics, inherited from apply_load_profile: an explicit value
    in the shell still wins. Only the LOAD half is borrowed, never the gate or
    the objective - a wrong load setting costs time, a wrong basis costs the
    comparison, and a manual run is usually manual because it varies the latter.
    """
    try:
        from auto_loop import apply_load_profile

        took = apply_load_profile()
    except Exception:
        return {}
    if took:
        print("[ar] load profile from campaign: %s"
              % ", ".join("%s=%s" % kv for kv in sorted(took.items())))
    return took


def refuse_contradictory_campaign():
    """True when this run's environment contradicts itself. Prints why.

    auto_loop.campaign_problems already knew that a net basis with the CatBoost
    screen or CatBoost illumination measures nothing: the screen stubs every
    neural member to a constant 0.5, so the archive is a pure CatBoost selection
    and the basis only re-scores the final gate. But that check ran in the
    LAUNCHER, so `python auto_research.py` walked straight past it and spent the
    night illuminating on CatBoost under a net basis. The guard belongs where
    the run starts, not where one of its two callers starts.

    Refuses rather than warns on purpose: a warning at 06:12 scrolls off the
    console behind the first training unit, which is exactly how this was missed.
    """
    try:
        from auto_loop import campaign_problems

        problems = campaign_problems(os.environ)
    except Exception:
        return False
    if not problems:
        return False
    print("[ar] this run's settings contradict each other:")
    for p in problems:
        print("  - %s" % p)
    print("  fix: GTRADE_AR_ILLUM=full with GTRADE_AR_SCREEN=0 on a net basis, "
          "or GTRADE_AR_SCORE_BASIS=raw to search CatBoost on purpose.")
    return True


def train_jobs():
    """How many training chunks run at once, after the card has been asked.

    Two by default: measured 27 percent faster than one on the same four
    assets, because net training is host-bound and the gain comes from overlap.
    gpu_fit_jobs is what keeps that from turning into the 15000-second stall.
    """
    try:
        want = max(1, int(os.getenv("GTRADE_AR_TRAIN_JOBS", "2") or 2))
    except ValueError:
        want = 2
    fits = gpu_fit_jobs(want)
    if fits < want:
        print("[ar] GPU has %s MiB free, which fits %d training process(es), "
              "not %d. Running %d." % (free_vram_mb(), fits, want, fits))
    return fits


def effective_chunk_size(subset, size=None, jobs=None):
    """Assets per chunk, small enough that every job actually gets one.

    GTRADE_AR_TRAIN_CHUNK is a CAP, picked for how much an interruption may cost
    and how much RAM one process may hold. It is not a target, and treating it as
    one silently disabled the whole parallel path: at 7 it never split the
    4-asset search unit, so `len(chunks) <= 1` took the single-process branch and
    GTRADE_AR_TRAIN_JOBS was ignored exactly where the time goes. Only the
    14-asset A/B arm ever reached it.

    Chunking stays off entirely at size 0, jobs included: that is the default and
    it must remain byte-identical.
    """
    size = train_chunk_size() if size is None else size
    jobs = train_jobs() if jobs is None else jobs
    n = len([a for a in subset.split(",") if a.strip()])
    if size <= 0 or jobs <= 1 or n <= 1:
        return size
    return min(size, -(-n // jobs))     # ceil(n / jobs)


def split_load(env, jobs, progress_dir=None):
    """One process's share of the box when `jobs` trainers run side by side.

    The settings sized against the WHOLE machine have to be divided, or the
    second process OOMs a 4 GB card that was already sitting at 3097 MB with
    one. Read from the overrides first and the ambient environment second,
    because the campaign sets them there.

    GTRADE_NEURAL_SLOTS is pinned to 1 no matter what: the parallelism now comes
    from the process count, and a second slot INSIDE a process is exactly the
    configuration that handed models the wrong sequence length and emptied 27
    genomes on 2026-08-17.
    """
    out = dict(env)
    if jobs > 1:
        try:
            pool = float(out.get("GTRADE_TF_POOL_PCT")
                         or os.getenv("GTRADE_TF_POOL_PCT") or 0.6)
            out["GTRADE_TF_POOL_PCT"] = "%.2f" % max(0.15, pool / jobs)
        except (TypeError, ValueError):
            pass
        try:
            threads = int(out.get("GTRADE_CB_THREADS")
                          or os.getenv("GTRADE_CB_THREADS") or 0)
        except (TypeError, ValueError):
            threads = 0
        if threads:
            out["GTRADE_CB_THREADS"] = str(max(1, threads // jobs))
        out["GTRADE_NEURAL_SLOTS"] = "1"
    if progress_dir:
        out["AR_PROGRESS_DIR"] = progress_dir
    return out


def _train_chunks_parallel(parts, env, jobs, label, key_of=None):
    """Train several chunks at once, one train_hybrid PROCESS per chunk.

    By process, never by thread. Two assets sharing one TF graph inside a single
    process is what emptied every unit on 2026-08-17: a model built for one
    sequence length was handed another's data ("padded_shape[0]=55 is not
    divisible by block_shape[0]=2"). Separate processes cannot do that to each
    other, because they do not share a graph at all.

    Only the FIRST chunk keeps the real progress files. ar_progress documents
    each trainer as their sole writer while it runs, and two writers would make
    the per-asset ETA on the research page a lie; the others are pointed at a
    scratch directory through AR_PROGRESS_DIR, which exists for exactly this.
    """
    from concurrent.futures import ThreadPoolExecutor

    quiet = tempfile.mkdtemp(prefix="ar_prog_")
    out = {}
    try:
        envs = [split_load(env, jobs, quiet if i else None)
                for i in range(len(parts))]
        _say("[%s] %d chunks on %d processes: %s"
             % (label, len(parts), jobs, " | ".join(parts)))
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            # _train_once, never train_env: these parts ARE the chunks, and
            # train_env would split each of them again.
            results = pool.map(lambda pair: _train_once(pair[0], pair[1]),
                               list(zip(parts, envs)))
            # Consumed in THIS thread, and banked as each one lands. cache_put is
            # a read-modify-write of a single JSON file, so a worker doing it
            # would lose the other worker's entry; and caching only after the
            # whole set finished would throw away every completed chunk the
            # moment one of them failed, which is the opposite of why chunks
            # exist.
            for part, rows in zip(parts, results):
                out[part] = rows or []
                if rows and key_of is not None:
                    ar_memory.cache_put(key_of(part), rows)
    finally:
        shutil.rmtree(quiet, ignore_errors=True)
    return out


def _cached_train(subset, env, key_of, label):
    """Train a subset behind the cross-run cache, one chunk at a time when
    GTRADE_AR_TRAIN_CHUNK is set.

    The same trick train_chunked.py uses on the production retrain, for the same
    reason. One held-out arm is 8 to 11 hours inside a single train_hybrid
    process, and an interruption loses all of it: the cache entry is only written
    once the whole subset is back. Per-chunk entries mean a resumed run reads the
    finished chunks and trains only the one that was in flight.

    Assets train independently, so the split changes no per-asset row - only how
    much work an interruption costs. Chunks below GTRADE_WORKERS do waste worker
    slots, which is the price of the smaller loss window.
    """
    # The whole-subset key first, always. Chunking changes which keys a train
    # writes, so without this every arm cached before it was switched on would
    # read as a miss and be retrained - hours of finished work thrown away by a
    # setting meant to protect exactly that work.
    whole = key_of(subset)
    rows = ar_memory.cache_get(whole)
    if rows is not None:
        _say("[%s] cache hit: %s" % (label, subset))
        return rows
    chunks = chunk_subsets(subset, effective_chunk_size(subset))
    if len(chunks) <= 1:
        rows = train_env(subset, env)     # one chunk: let train_env decide
        if rows:
            ar_memory.cache_put(whole, rows)
        return rows
    done, pending = {}, []
    for i, part in enumerate(chunks, 1):
        rows = ar_memory.cache_get(key_of(part))
        if rows is None:
            pending.append(part)
        else:
            _say("[%s] chunk %d/%d cache hit: %s" % (label, i, len(chunks), part))
            done[part] = rows

    jobs = min(train_jobs(), len(pending))
    if jobs > 1:
        done.update(_train_chunks_parallel(pending, env, jobs, label, key_of))
    else:
        for i, part in enumerate(pending, 1):
            _say("[%s] chunk %d/%d training: %s" % (label, i, len(pending), part))
            rows = train_env(part, env, split=False)   # already a chunk
            done[part] = rows or []
            # Banked immediately, not after the loop: an interruption must cost
            # the chunk in flight and nothing else.
            if rows:
                ar_memory.cache_put(key_of(part), rows)

    out = []
    for part in chunks:
        out.extend(done.get(part) or [])
    return out


def train_base_cached(subset, env):
    """Cache-first BASE training: identical subset + env + feature space +
    data snapshot reuses the stored quality rows instead of retraining.
    Candidate runs never go through here (their envs embed temp paths)."""
    return _cached_train(subset, env,
                         lambda sub: ar_memory.base_key(sub, env),
                         "auto-research base")


def neural_contribution(full_rows, cbonly_rows):
    """Per-asset neural-member contribution = full ensemble Score minus the CB-only
    Score (nets replaced by neutral 0.5 under GTRADE_SCREEN_ONLY), over the shared
    assets. Assets missing from either side are skipped."""
    cb = {r["Asset"]: r.get("Score", 0.0) for r in cbonly_rows}
    out = {}
    for r in full_rows:
        a = r["Asset"]
        if a in cb:
            out[a] = r.get("Score", 0.0) - cb[a]
    return out


def contribution_rows(subset, env, full_fn):
    """Neural-contribution rows for a config: a full train (via full_fn, e.g.
    train_base_cached for bases or train_env for candidates) minus a CB-only train
    (also via full_fn with GTRADE_SCREEN_ONLY added). Empty if either train yields no
    rows. Using full_fn for the CB train means a cached base (full_fn=train_base_cached)
    gets a cached CB train too, and an injected fake trainer intercepts the CB train in
    tests. For candidates and winners full_fn is train_env, so their CB train is
    unchanged. This is a consistency change only; production result numbers are the same."""
    full = full_fn(subset, env)
    cb = full_fn(subset, screen_env(env))
    return [{"Asset": a, "Score": c}
            for a, c in neural_contribution(full, cb).items()]


def _heldout_eval(subset, env, full_fn, done_out=None):
    """(full_rows, contribution_rows) for one config, sharing the single full train
    so the metric never pays for a redundant full train. The CB train uses full_fn
    (with GTRADE_SCREEN_ONLY added): a cached base gets a cached CB train, and an
    injected fake trainer intercepts the CB train in tests. For candidates and winners
    full_fn is train_env, so their CB train is unchanged.

    done_out, if given, must be a list; it receives exactly one entry: the
    training-unit file's per-asset done pairs from the FULL train specifically,
    captured immediately after it returns and before the CB-only train starts.
    Both trains are separate train_hybrid.py subprocesses and each is the sole
    writer of ar_progress_unit.json while it runs, and the CB train always runs
    SECOND here - so a caller that waits until both finish and then reads the
    unit file always sees the CB train's second-scale per-asset times, never
    the full train's hour-scale ones. The entry is None when the full train
    did not actually publish a fresh unit record
    (e.g. it was satisfied from cache and unit_begin never ran), so a caller
    never mistakes a stale leftover file for this training's own times."""
    mark = _progress_unit_marker() if done_out is not None else None
    full = full_fn(subset, env)
    if done_out is not None:
        done_out.append(_progress_unit_done_since(mark))
    cb = full_fn(subset, screen_env(env))
    contrib = [{"Asset": a, "Score": c}
               for a, c in neural_contribution(full, cb).items()]
    return full, contrib


def _score_basis():
    """What the search scores.

    raw      the ensemble Score (default).
    neural   the neural CONTRIBUTION, i.e. full Score minus a CatBoost-only run.
             Still a Score, so it still carries the Score's instability.
    net_auc  the fold-averaged AUC of the neural members on their own raw
             probabilities (train_hybrid writes Net_AUC). This is the only
             neural read-out that is measurable on this GPU: the Score is a
             backtest of discrete signals behind a fold-admission threshold and
             moves by whole points when the nets drift in the 4th decimal
             (measured 2026-08-14), while AUC is a rank statistic and does not.
    net_gain the ensemble's AUC MINUS CatBoost's own (train_hybrid writes
             Ens_AUC and CB_AUC). This is what neural_lift was always trying to
             ask - what do the nets ADD - on a rank statistic instead of a
             backtest Score. Unlike net_auc it stays correct if the nets are
             given a target other than direction, so it is the basis P3 needs.
    ens_auc  the ENSEMBLE's own fold-averaged AUC. The right basis whenever a
             candidate changes BOTH learners: net_gain would then reward simply
             damaging CatBoost, since
                 d(net_gain) = d(Ens_AUC) - d(CB_AUC)
             and a candidate that removes a feature from CatBoost drives the
             second term negative, inflating the score while the ensemble gets
             worse. ens_auc asks the only question that matters there - did the
             final ensemble end up ranking better - on the same stable scale.
             DIFFERENT UNITS: all three AUC bases live near 0.5 and their deltas
             are thousandths, so their adoption floor is GTRADE_AR_ADOPT_AUC,
             not the Score floor."""
    b = (os.getenv("GTRADE_AR_SCORE_BASIS") or "raw").strip().lower()
    if b not in ("raw", "neural", "net_auc", "net_gain", "ens_auc"):
        logger.warning("unknown GTRADE_AR_SCORE_BASIS %r, using raw", b)
        return "raw"
    return b


def decision_basis():
    """What an ADOPTION is judged on, which need not be what the search optimised.

    The search basis is picked for signal-to-noise: raw Score cannot measure a
    neural change on this box (the same elite gated three times read +3.90,
    +3.89 and -2.49 against an adopt floor of 0.5), so the campaign searches on
    net_auc. That is an argument about MEASURABILITY and it says nothing about
    which quantity production decides on. One constant for both silently
    promoted a search convenience into the adoption criterion, and on
    2026-08-18 the two came apart completely: over the same 14 held-out assets
    and the same two trainings, the mean Net_AUC delta was +0.036 while the mean
    Score delta was -1.85, rank correlation -0.24 (p=0.40), sign agreement 7 of
    14. The A/B passed; 10 of those 14 assets would have been demoted.

    Unset means "the same basis", so every existing campaign is unchanged.
    """
    b = (os.getenv("GTRADE_AR_DECISION_BASIS") or "").strip().lower()
    if not b:
        return _score_basis()
    if b not in ("raw", "neural", "net_auc", "net_gain", "ens_auc"):
        logger.warning("unknown GTRADE_AR_DECISION_BASIS %r, using the search "
                       "basis", b)
        return _score_basis()
    return b


def ens_auc_rows(rows):
    """Re-key quality rows onto the ensemble's own AUC (a LEVEL, not a
    difference), so a candidate that changes both learners is judged on where
    the ensemble ends up."""
    out = []
    for r in rows:
        v = r.get("Ens_AUC")
        if v is None:
            continue
        try:
            out.append({"Asset": r["Asset"], "Score": float(v)})
        except (TypeError, ValueError):
            continue
    return out


def net_gain_rows(rows):
    """Re-key quality rows onto the ensemble's AUC gain over CatBoost alone.

    A DIFFERENCE of two rank statistics, so a change that helps both learners
    equally reads as zero here by construction - the same caveat the `neural`
    basis carries, minus the Score's instability. Assets missing either column
    are dropped rather than scored 0, which would read as a catastrophic loss
    instead of a missing measurement."""
    out = []
    for r in rows:
        e, c = r.get("Ens_AUC"), r.get("CB_AUC")
        if e is None or c is None:
            continue
        try:
            out.append({"Asset": r["Asset"], "Score": float(e) - float(c)})
        except (TypeError, ValueError):
            continue
    return out


def net_auc_rows(rows):
    """Re-key quality rows onto Net_AUC so the whole agent (delta, objective,
    Wilcoxon, gate) keeps working unchanged on the new basis. Assets whose
    training produced no usable AUC are dropped rather than scored 0, which
    would read as a catastrophic loss instead of a missing measurement."""
    out = []
    for r in rows:
        v = r.get("Net_AUC")
        if v is None:
            continue
        try:
            out.append({"Asset": r["Asset"], "Score": float(v)})
        except (TypeError, ValueError):
            continue
    return out


def score_rows(subset, env, full_fn):
    """Scoring rows for a config under the active basis. raw - full_fn(subset, env)
    (current behavior, caching preserved). neural - neural-contribution rows.
    net_auc / net_gain - the same single train, re-keyed onto the fold-averaged
    neural AUC or onto the ensemble's AUC gain over CatBoost (no extra training,
    so both bases are free)."""
    basis = _score_basis()
    if basis == "neural":
        return contribution_rows(subset, env, full_fn)
    return rekey_rows(full_fn(subset, env))


def rekey_rows(rows, basis=None):
    """Re-key already-trained rows onto a basis, the active one by default.

    Split out of score_rows so the TIER gate can use it too: the tier stage
    trains its own mini rows and used to compare them on the raw Score no matter
    which basis was selected, which silently pruned candidates on the one metric
    measured to be unusable here (2026-08-14: same config, same seed, 0.64
    apart). The `neural` basis is NOT here - it needs a second training run, so
    it stays in score_rows.

    The explicit `basis` argument is what lets an ADOPTION be judged on a
    different column than the SEARCH optimised: the rows already carry every
    column, so re-keying the same training onto another basis costs nothing.
    """
    basis = basis or _score_basis()
    if basis == "net_auc":
        return net_auc_rows(rows)
    if basis == "net_gain":
        return net_gain_rows(rows)
    if basis == "ens_auc":
        return ens_auc_rows(rows)
    return rows


def _feature_env(specs, extra_names):
    """Feature-axis env overrides: materialize DSL specs to a temp file and point
    train_hybrid at them. Empty specs means no overrides (the plain base set)."""
    if not specs:
        return {}
    tmp = tempfile.mkdtemp(prefix="ar_specs_")
    spath = os.path.join(tmp, "specs.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(specs, f)
    return {"GTRADE_DSL_SPECS": spath, "GTRADE_EXTRA_FEATURES": ",".join(extra_names)}


def train_rows(subset, specs, extra_names):
    """Back-compat wrapper: train the subset with DSL feature specs as extra features."""
    return train_env(subset, _feature_env(specs, extra_names))


# Gene palettes for the model-hyperparameter and net-hygiene groups. Values are
# RELATIVE (deltas/multipliers on each asset's tuned baseline - see train_hybrid
# cb_params_for/lookback_for), so one candidate composes with 181 baselines.
DEPTH_DELTAS = (-2, -1, 0, 1, 2)
LR_MULTS = (0.5, 0.7, 1.0, 1.5, 2.0)
ITER_MULTS = (0.7, 1.0, 1.5)
LOOKBACK_DELTAS = (-10, -5, 0, 5, 10)
NET_SEED_CHOICES = (1, 3)
LABEL_MODES = ("direction", "rel_median", "triple_barrier")
TB_HORIZONS = (5, 10, 20)  # triple_barrier reuses label_window as the horizon H
THR_MARGINS = (0, 0.02, 0.05)
BAND_DELTAS = (-0.01, 0, 0.01, 0.02)
REGIME_MODES = ("both", "off", "sma_only", "taleb_only")

_HYPER_DEFAULTS = (0, 1.0, 1.0, 0)
_NET_DEFAULTS = (1, 0, 0, 0)
_TUNING_DEFAULTS = (0.0, 0.0, "both")


@dataclass
class Genome:
    """A composable cross-axis experiment: feature drops + extra DSL specs + labeling
    + relative model hyperparameters + net-hygiene toggles. The objective is NOT here
    (it is the QD fitness). Empty == production default.

    label_window doubles as the horizon H when label_mode == "triple_barrier".
    net_uniqueness is a SEPARATE gene from the label on purpose: the negative
    2026-07-11 triple-barrier A/B bundled them, so the agent could never tell which
    one hurt - now it can test the label with and without uniqueness weighting.
    cb_uniqueness follows the same reasoning: it is CatBoost's own copy of the
    uniqueness-weighting toggle, kept separate from net_uniqueness so the agent
    can test the shared lever on each learner independently (or both at once via
    the weighting axis) instead of the two always moving together."""
    drops: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    label_mode: str = "direction"
    label_window: int = 30
    cb_depth_delta: int = 0
    cb_lr_mult: float = 1.0
    cb_iter_mult: float = 1.0
    lookback_delta: int = 0
    net_seeds: int = 1
    net_uniqueness: int = 0
    cb_uniqueness: int = 0
    net_calibrate: int = 0
    thr_margin: float = 0.0
    band_delta: float = 0.0
    regime_mode: str = "both"


def _hyper_genes(g):
    return (g.cb_depth_delta, g.cb_lr_mult, g.cb_iter_mult, g.lookback_delta)


def _net_genes(g):
    return (g.net_seeds, g.net_uniqueness, g.cb_uniqueness, g.net_calibrate)


def _tuning_genes(g):
    return (float(g.thr_margin), float(g.band_delta), g.regime_mode)


def genome_to_env(g):
    """Compose the genome into one training env-override dict (empty - no overrides).

    The composition rules live in core.adopted so the research loop and
    production cannot disagree about what a genome means. Only the temp spec file
    is added here, because it is per-candidate and exists for this process only.
    """
    from dataclasses import asdict

    from core import adopted as _adopted

    env = dict(_feature_env(g.extra, [s["name"] for s in g.extra]))
    env.update(_adopted.env_overrides(asdict(g)))
    return env


def _spec_signature(spec):
    """Identity of a spec ignoring its name, used to dedup against the log."""
    return (spec["op"], tuple(spec.get("inputs") or []),
            tuple(sorted((spec.get("params") or {}).items())))


def _canon_genome(g):
    """direction ignores the window; canonicalize so equivalent genomes dedup."""
    if g.label_mode == "direction":
        g.label_window = 30
    g.thr_margin = round(float(g.thr_margin), 4)
    g.band_delta = round(float(g.band_delta), 4)
    g.cb_lr_mult = round(float(g.cb_lr_mult), 4)
    g.cb_iter_mult = round(float(g.cb_iter_mult), 4)
    g.cb_depth_delta = int(g.cb_depth_delta)
    g.lookback_delta = int(g.lookback_delta)
    return g


def genome_sig(g):
    """Canonical cross-run identity of a genome (drop order and the arbitrary
    spec NAMES are ignored; only the spec signatures matter).

    The hyper/nets gene groups enter the signature ONLY when they differ from the
    production default, so every genome signature recorded before those genes
    existed stays byte-identical - the tried-registry and the replication gate
    keep their history."""
    d = {
        "drops": sorted(g.drops),
        "extra": sorted(json.dumps(_spec_signature(s)) for s in g.extra),
        "label": [g.label_mode, g.label_window],
    }
    if _hyper_genes(g) != _HYPER_DEFAULTS:
        d["hyper"] = list(_hyper_genes(g))
    if _net_genes(g) != _NET_DEFAULTS:
        d["nets"] = list(_net_genes(g))
    if _tuning_genes(g) != _TUNING_DEFAULTS:
        d["tuning"] = list(_tuning_genes(g))
    return json.dumps(d, sort_keys=True)


def valid(g, active, prune_min, continuous=False):
    """Well-formedness against the active feature set and the prune floor."""
    aset = set(active)
    extra_names = {s.get("name") for s in g.extra}
    if any(d not in aset for d in g.drops):
        return False
    if set(g.drops) & extra_names:
        return False
    cols = aset | extra_names
    if any(not validate_spec(s, cols) for s in g.extra):
        return False
    if g.label_mode not in LABEL_MODES:
        return False
    if g.label_window <= 0:
        return False
    if continuous:
        from core.ar_rl import CMA_DIMS
        for name, lo, hi, is_int in CMA_DIMS:
            v = getattr(g, name)
            if not (lo <= v <= hi):
                return False
            if is_int and v != int(v):
                return False
    else:
        if g.cb_depth_delta not in DEPTH_DELTAS:
            return False
        if g.cb_lr_mult not in LR_MULTS or g.cb_iter_mult not in ITER_MULTS:
            return False
        if g.lookback_delta not in LOOKBACK_DELTAS:
            return False
        if g.thr_margin not in THR_MARGINS or g.band_delta not in BAND_DELTAS:
            return False
    if g.net_seeds not in NET_SEED_CHOICES:
        return False
    if (g.net_uniqueness not in (0, 1) or g.cb_uniqueness not in (0, 1)
            or g.net_calibrate not in (0, 1)):
        return False
    if g.regime_mode not in REGIME_MODES:
        return False
    return not len(aset) - len(set(g.drops)) < prune_min


def random_genome(active, base_features):
    """A random valid starting genome for QD initialization."""
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    g = Genome()
    max_drops = max(0, len(active) - prune_min)
    n_drop = random.randint(0, min(3, max_drops))
    g.drops = random.sample(list(active), n_drop) if n_drop else []
    for i in range(random.randint(0, 2)):
        spec = _random_spec(base_features, "g%d_%d" % (random.randint(0, 99999), i), None)
        if spec and spec["name"] not in g.drops:
            g.extra.append(spec)
    r = random.random()
    if r < 0.35:
        g.label_mode = "rel_median"
        g.label_window = random.choice([20, 30, 60])
    elif r < 0.5:
        g.label_mode = "triple_barrier"
        g.label_window = random.choice(TB_HORIZONS)
    if random.random() < 0.3:
        _mutate_hyper(g)
    if random.random() < 0.2:
        _mutate_nets(g)
    if random.random() < 0.2:
        _mutate_tuning(g)
    return g if valid(g, active, prune_min) else Genome()


def _mutate_hyper(g):
    """Set one random hyperparameter gene to a random palette value (may be the
    default - that is how a gene walks back to baseline)."""
    gene = random.choice(("depth", "lr", "iter", "lookback"))
    if gene == "depth":
        g.cb_depth_delta = random.choice(DEPTH_DELTAS)
    elif gene == "lr":
        g.cb_lr_mult = random.choice(LR_MULTS)
    elif gene == "iter":
        g.cb_iter_mult = random.choice(ITER_MULTS)
    else:
        g.lookback_delta = random.choice(LOOKBACK_DELTAS)


def _mutate_nets(g, active=None):
    """Flip one net-hygiene gene."""
    gene = random.choice(["seeds", "uniq", "cb_uniq", "calib"])
    if gene == "seeds":
        g.net_seeds = random.choice([s for s in NET_SEED_CHOICES if s != g.net_seeds])
    elif gene == "uniq":
        g.net_uniqueness = 1 - g.net_uniqueness
    elif gene == "cb_uniq":
        g.cb_uniqueness = 1 - g.cb_uniqueness
    else:
        g.net_calibrate = 1 - g.net_calibrate


def _mutate_tuning(g):
    """Set one random tuning gene to a random palette value."""
    gene = random.choice(("margin", "band", "regime"))
    if gene == "margin":
        g.thr_margin = random.choice(THR_MARGINS)
    elif gene == "band":
        g.band_delta = random.choice(BAND_DELTAS)
    else:
        g.regime_mode = random.choice(REGIME_MODES)


def mutate(g, active, base_features, ops=None):
    """One random gene change; always returns a valid genome (or the input if no valid
    single change exists). ops=None keeps today's behavior byte-identical; a list
    restricts which ops are tried."""
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    all_ops = ["add_drop", "rm_drop", "add_extra", "rm_extra", "flip_label",
               "win", "hyper", "nets", "tuning"]
    ops = list(all_ops if ops is None else ops)
    random.shuffle(ops)
    for op in ops:
        ng = copy.deepcopy(g)
        if op == "add_drop":
            taken = set(ng.drops) | {s.get("name") for s in ng.extra}
            cand = [f for f in active if f not in taken]
            if cand:
                ng.drops.append(random.choice(cand))
        elif op == "rm_drop":
            if ng.drops:
                ng.drops.pop(random.randrange(len(ng.drops)))
        elif op == "add_extra":
            spec = _random_spec(base_features, "m%d" % random.randint(0, 99999), None)
            if spec and spec["name"] not in ng.drops:
                ng.extra.append(spec)
        elif op == "rm_extra":
            if ng.extra:
                ng.extra.pop(random.randrange(len(ng.extra)))
        elif op == "flip_label":
            ng.label_mode = random.choice([m for m in LABEL_MODES if m != ng.label_mode])
            if ng.label_mode == "triple_barrier" and ng.label_window not in TB_HORIZONS:
                ng.label_window = random.choice(TB_HORIZONS)
        elif op == "win":
            pool = TB_HORIZONS if ng.label_mode == "triple_barrier" else (20, 30, 60)
            ng.label_window = random.choice(
                [w for w in pool if w != ng.label_window] or [pool[1]])
        elif op == "hyper":
            _mutate_hyper(ng)
        elif op == "nets":
            _mutate_nets(ng)
        elif op == "tuning":
            _mutate_tuning(ng)
        if ng != g and valid(ng, active, prune_min):
            return ng
    return g


def crossover(g1, g2, active):
    """Uniform crossover over gene-groups; resolves drop-extra conflicts (drop loses)
    and the prune floor; always returns a valid genome."""
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    child = Genome()
    child.drops = copy.deepcopy(g1.drops if random.random() < 0.5 else g2.drops)
    child.extra = copy.deepcopy(g1.extra if random.random() < 0.5 else g2.extra)
    if random.random() < 0.5:
        child.label_mode, child.label_window = g1.label_mode, g1.label_window
    else:
        child.label_mode, child.label_window = g2.label_mode, g2.label_window
    hp = g1 if random.random() < 0.5 else g2
    child.cb_depth_delta, child.cb_lr_mult = hp.cb_depth_delta, hp.cb_lr_mult
    child.cb_iter_mult, child.lookback_delta = hp.cb_iter_mult, hp.lookback_delta
    np_ = g1 if random.random() < 0.5 else g2
    child.net_seeds, child.net_uniqueness = np_.net_seeds, np_.net_uniqueness
    child.cb_uniqueness = np_.cb_uniqueness
    child.net_calibrate = np_.net_calibrate
    tp = g1 if random.random() < 0.5 else g2
    child.thr_margin, child.band_delta = tp.thr_margin, tp.band_delta
    child.regime_mode = tp.regime_mode
    aset = set(active)
    cols = aset | {s.get("name") for s in child.extra}
    child.extra = [s for s in child.extra if validate_spec(s, cols)]
    extra_names = {s.get("name") for s in child.extra}
    child.drops = [d for d in dict.fromkeys(child.drops) if d in aset and d not in extra_names]
    while len(aset) - len(set(child.drops)) < prune_min and child.drops:
        child.drops.pop()
    return child


_FLOOR_EDGES = (-1.0, -0.25, 0.25, 1.0)
_COUNT_EDGES = (12, 18, 24, 30)


def _bin(value, edges):
    """Bin index 0..len(edges) by right-insertion into the edge list."""
    return bisect.bisect_right(edges, value)


def _gene_group(genome):
    """Categorical lever-group for the v2 niche descriptor: which non-feature
    lever the genome touches. 0 none/features-only, 1 label, 2 hyper, 3 nets,
    4 tuning (thresholds/regime), 5 mixed. Keeps one lever class from
    monopolizing the archive - each group competes in its own niches."""
    touched = []
    if genome.label_mode != "direction":
        touched.append(1)
    if _hyper_genes(genome) != _HYPER_DEFAULTS:
        touched.append(2)
    if _net_genes(genome) != _NET_DEFAULTS:
        touched.append(3)
    if _tuning_genes(genome) != _TUNING_DEFAULTS:
        touched.append(4)
    if not touched:
        return 0
    return touched[0] if len(touched) == 1 else 5


def fitness(rows, base_score):
    """Archive cell quality: the MEAN paired Score delta."""
    return _objective_delta(rows, base_score, "mean")[0]


def behavior(genome, rows, base_score, active):
    """Behavior descriptors: (worst-asset delta bin, feature-count bin,
    lever-group bin) - the v2 descriptor."""
    min_delta = _objective_delta(rows, base_score, "min")[0]
    count = len(active) - len(set(genome.drops)) + len(genome.extra)
    return _bin(min_delta, _FLOOR_EDGES), _bin(count, _COUNT_EDGES), _gene_group(genome)


_QD_ARCHIVE_PATH = os.path.join(BASE, "_qd_archive.json")


def archive_put(archive, genome, rows, base_score, active):
    """Place a genome in its (floor, complexity) niche if it beats the niche's mean
    fitness (or the niche is empty). Returns True when stored."""
    bd = behavior(genome, rows, base_score, active)
    key = "%d_%d_%d" % bd
    f = fitness(rows, base_score)
    cur = archive.get(key)
    if cur is None or f > cur["fitness"]:
        archive[key] = {"genome": genome, "fitness": f, "rows": rows}
        return True
    return False


def _qd_save(archive):
    out = {k: {"genome": asdict(v["genome"]), "fitness": v["fitness"]}
           for k, v in archive.items()}
    with open(_QD_ARCHIVE_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh)


def _qd_load():
    """Reload the archive (genomes only; rows are re-derived on resume) or {}.
    Old two-part "f_c" keys (pre lever-group descriptor) are migrated in place
    by appending the genome's lever-group bin - lossless, no retraining."""
    if not os.path.exists(_QD_ARCHIVE_PATH):
        return {}
    try:
        with open(_QD_ARCHIVE_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
        out = {}
        for k, v in raw.items():
            g = Genome(**v["genome"])
            key = k if k.count("_") == 2 else "%s_%d" % (k, _gene_group(g))
            out[key] = {"genome": g, "fitness": v["fitness"], "rows": []}
        return out
    except Exception:
        return {}


_LLM_WARNED = False


def _llm_warn(reason):
    """Report the FIRST LLM failure of the run, then stay quiet. Without this a dead
    backend is indistinguishable from a working one: the fallback is silent, so the
    search looks healthy while the LLM arm contributes nothing."""
    global _LLM_WARNED
    if _LLM_WARNED:
        return
    _LLM_WARNED = True
    print("[llm] proposer unavailable, falling back to the evolutionary operators "
          f"for the rest of this run: {reason}")


def _llm_child(elites, active, base_features):
    """A genome proposed by the LLM, converted and validated; None on ANY
    problem (the QD loop then falls back to the evolutionary operators, so an
    unreachable Ollama can never kill the search).

    Once the backend has failed, this stops calling it for the rest of the run.
    The failures that happen here are structural (a model too slow for the
    machine, a stopped Ollama), not transient, so retrying every step buys
    nothing and costs the timeout each time - fifteen steps at the 600s default
    is two and a half hours of a search doing nothing but waiting.
    """
    if _LLM_WARNED:
        return None
    top = sorted(elites, key=lambda e: e["fitness"], reverse=True)[:5]
    parent = random.choice(top)["genome"]
    summary = [{"genome": asdict(e["genome"]), "fitness": e["fitness"]} for e in top]
    avoid = ar_memory.tried_recent("genome", 30)
    try:
        obj = llm_proposer.propose_genome(
            asdict(parent), summary, active, base_features, avoid=avoid)
    except Exception as exc:
        _llm_warn(exc)
        return None
    if obj is None:
        _llm_warn("backend returned no parseable genome (empty reply? raise "
                  "GTRADE_AR_LLM_MAX_TOKENS - reasoning models spend the cap "
                  "before answering)")
    if not isinstance(obj, dict):
        return None
    try:
        g = Genome(drops=list(obj.get("drops") or []),
                   extra=list(obj.get("extra") or []),
                   label_mode=str(obj.get("label_mode", "direction")),
                   label_window=int(obj.get("label_window", 30)))
    except (TypeError, ValueError):
        return None
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
    return g if valid(g, active, prune_min) else None


def _surrogate_child(archive, active, base_features):
    """Generate up to n_candidates() unseen children via mutate/crossover (valid by
    construction, like the plain loop), score each with a surrogate fit on the archive,
    and return the highest-predicted previously-untried child. None when the surrogate
    cannot be fit or every candidate is already tried."""
    elites = list(archive.values())
    samples = [(qd_surrogate.genome_vector(e["genome"], active, base_features), e["fitness"])
               for e in elites]
    model = qd_surrogate.fit_surrogate(samples)
    if model is None:
        return None
    best, best_pred = None, None
    for _ in range(qd_surrogate.n_candidates()):
        parent = random.choice(elites)["genome"]
        if len(elites) >= 2 and random.random() < 0.5:
            child = crossover(parent, random.choice(elites)["genome"], active)
        else:
            child = mutate(parent, active, base_features)
        child = _canon_genome(child)
        if ar_memory.tried_seen("genome", genome_sig(child)):
            continue
        pred = qd_surrogate.predict(
            model, qd_surrogate.genome_vector(child, active, base_features))
        if best is None or pred > best_pred:
            best, best_pred = child, pred
    return best


def next_child(archive, active, base_features, attempts=10):
    """One unseen child genome for the QD loop: LLM-proposed (when the llm proposer is selected, with probability GTRADE_AR_QD_LLM_P) else mutate/crossover retried against the tried-registry.
    None when the archive is empty or every attempt lands on an already-tried genome."""
    if ar_rl.rl_on():
        return _rl_controller().next_child(archive, active, base_features,
                                           attempts=attempts)
    elites = list(archive.values())
    if not elites:
        return None
    if llm_proposer.llm_selected() and random.random() < float(
            os.getenv("GTRADE_AR_QD_LLM_P") or "0.3"):
        child = _llm_child(elites, active, base_features)
        if child is not None:
            child = _canon_genome(child)
            if not ar_memory.tried_seen("genome", genome_sig(child)):
                return child
    if qd_surrogate.surrogate_on():
        try:
            child = _surrogate_child(archive, active, base_features)
        except Exception:
            child = None
        if child is not None:
            return child
    for _ in range(attempts):
        parent = random.choice(elites)["genome"]
        if len(elites) >= 2 and random.random() < 0.5:
            child = crossover(parent, random.choice(elites)["genome"], active)
        else:
            child = mutate(parent, active, base_features)
        child = _canon_genome(child)
        if not ar_memory.tried_seen("genome", genome_sig(child)):
            return child
    return None


_FEAT_OPS = ["add_drop", "rm_drop", "add_extra", "rm_extra", "flip_label", "win"]
_GROUP_TO_OPS = {2: ["hyper"], 3: ["nets"], 4: ["tuning"], 1: ["flip_label", "win"]}
_TOTAL_CELLS = (len(_FLOOR_EDGES) + 1) * (len(_COUNT_EDGES) + 1) * 6


class _RlController:
    """Wires core.ar_rl into the QD loop. Exists only under GTRADE_AR_RL=1."""

    def __init__(self):
        state = ar_memory.blob_get(ar_rl.STATE_KEY) or {}
        self.sched = ar_rl.Scheduler(state.get("scheduler"))
        self.cur = ar_rl.CuriosityMap(state.get("curiosity"))
        self.cma = ar_rl.CmaEmitter(state.get("cma"))
        self.monitor = ar_rl.FallbackMonitor(state.get("monitor"))
        self.origin = dict(state.get("origin") or {})
        # A trip lasts one run, which is what its own message promises. The
        # windows are what carried it further: while disabled every draw is
        # recorded as a floor draw, so sched_hits freezes at the evidence that
        # tripped it, and the next run - which starts enabled, since `disabled`
        # itself was never persisted - re-tripped on that stale window before
        # the scheduler chose anything. Starting the run with both windows
        # cleared is what gives it a fresh MIN_SCHED draws to answer for itself.
        self.disabled = False
        if state.get("disabled"):
            self.monitor.clear()
        base = ar_memory.base_key(selection_assets(), {})
        if state.get("base_key") and state["base_key"] != base:
            self.sched.halve()
        self.base_key = base

    # -- persistence -------------------------------------------------------
    def save(self):
        ar_memory.blob_put(ar_rl.STATE_KEY, {
            "version": 1, "base_key": self.base_key,
            "scheduler": self.sched.to_state(),
            "curiosity": self.cur.to_state(),
            "cma": self.cma.to_state(),
            "monitor": self.monitor.to_state(),
            "disabled": self.disabled,
            "origin": dict(list(self.origin.items())[-ar_rl.ORIGIN_CAP:]),
        })

    def _note_origin(self, sig, arm, phase):
        self.origin[sig] = [arm, phase]
        if len(self.origin) > ar_rl.ORIGIN_CAP:
            self.origin = dict(list(self.origin.items())[-ar_rl.ORIGIN_CAP:])

    # -- child generation --------------------------------------------------
    def _pick_parent(self, archive, rng):
        sigs = {genome_sig(e["genome"]): e for e in archive.values()}
        sig = self.cur.pick(list(sigs.keys()), rng)
        return sigs[sig]["genome"], sig

    def next_child(self, archive, active, base_features, attempts=10):
        elites = list(archive.values())
        if not elites:
            return None
        occupancy = len(archive) / float(_TOTAL_CELLS)
        phase = ar_rl.phase_of(occupancy)
        # "nets" is absent under the CB screen, which replaces every neural member
        # with a constant 0.5 (train_hybrid._screen_only): mutate(ops=["nets"])
        # changes ONLY net genes, so such a child scores identically to its
        # parent, can never enter the archive, and the bandit books a guaranteed
        # failure. That is not evidence about net genes, it is an artifact of the
        # screen - and it had already taught the scheduler to avoid them (measured
        # posterior 2026-08-14: nets 1.8/9.7, P=0.16, the worst of the informative
        # arms; reset that arm before trusting a full-illumination run). Under
        # GTRADE_AR_ILLUM=full the nets are real during illumination, so net genes
        # score differently from their parent and the arm is informative again.
        available = ["feat", "hyper", "tuning", "novelty"]
        if illum_full():
            available.append("nets")
        if len(elites) >= 2:
            available.append("cross")
        if llm_proposer.llm_selected():
            available.append("llm")
        if qd_surrogate.surrogate_on():
            available.append("surr")
        available.append("cma")
        for _ in range(attempts):
            if self.disabled:
                arm, was_floor = random.choice(available), True
            else:
                arm, was_floor = self.sched.choose(available, phase)
            child = self._emit(arm, archive, active, base_features)
            if child is None:
                continue
            child = _canon_genome(child)
            cont = arm == "cma"
            prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))
            if not valid(child, active, prune_min, continuous=cont):
                continue
            sig = genome_sig(child)
            if ar_memory.tried_seen("genome", sig):
                continue
            self._note_origin(sig, arm, phase)
            self._last_draw = (arm, phase, was_floor)
            self._last_parent_sig = getattr(self, "_pending_parent_sig", None)
            return child
        return None

    def _emit(self, arm, archive, active, base_features):
        elites = list(archive.values())
        parent, psig = self._pick_parent(archive, random)
        self._pending_parent_sig = psig
        if arm == "feat":
            return mutate(parent, active, base_features, ops=_FEAT_OPS)
        if arm == "hyper":
            return mutate(parent, active, base_features, ops=["hyper"])
        if arm == "nets":
            return mutate(parent, active, base_features, ops=["nets"])
        if arm == "tuning":
            return mutate(parent, active, base_features, ops=["tuning"])
        if arm == "cross":
            other, _ = self._pick_parent(archive, random)
            return crossover(parent, other, active)
        if arm == "llm":
            return _llm_child(elites, active, base_features)
        if arm == "surr":
            try:
                return _surrogate_child(archive, active, base_features)
            except Exception:
                return None
        if arm == "cma":
            best = max(elites, key=lambda e: e["fitness"])
            if not self.cma.to_state()["evals"]:
                self.cma.seed_from(best["genome"])
            return self.cma.ask(parent)
        if arm == "novelty":
            emitter = ar_rl.NoveltyEmitter(
                count_bins=len(_COUNT_EDGES) + 1, groups=6, rng=random)

            def mutate_toward(p, tbin, tgroup):
                ops = _GROUP_TO_OPS.get(tgroup, _FEAT_OPS)
                cand = mutate(p, active, base_features, ops=ops)
                cnt = len(active) - len(set(cand.drops)) + len(cand.extra)
                pcnt = len(active) - len(set(p.drops)) + len(p.extra)
                cb, pb = _bin(cnt, _COUNT_EDGES), _bin(pcnt, _COUNT_EDGES)
                good_group = tgroup in (0, 5) or _gene_group(cand) == tgroup
                if good_group and (cb == tbin or abs(cb - tbin) < abs(pb - tbin)):
                    return cand
                return None

            return emitter.emit(
                list(archive.keys()), elites,
                count_of=lambda g: len(active) - len(set(g.drops)) + len(g.extra),
                group_of=_gene_group,
                count_bin_of=lambda c: _bin(c, _COUNT_EDGES),
                mutate_toward=mutate_toward)
        return None

    # -- rewards -----------------------------------------------------------
    def on_result(self, sig, stored):
        info = self.origin.get(sig)
        if info is None:
            return
        arm, phase = info
        was_floor = getattr(self, "_last_draw", (None, None, False))[2]
        if not self.disabled:
            self.sched.update(arm, phase, stored)
        self.monitor.record(was_floor, stored)
        if self.monitor.tripped() and not self.disabled:
            self.disabled = True
            _say("[rl] fallback tripped: scheduler underperforms the uniform "
                 "floor - reverting to uniform for the rest of this run.")
        psig = getattr(self, "_last_parent_sig", None)
        if psig:
            self.cur.reward(psig) if stored else self.cur.penalize(psig)
        if arm == "cma":
            pass  # cma.tell happens in run_qd where fitness is known
        self.save()

    def on_adopt(self, sig):
        info = self.origin.get(sig)
        if info is None:
            return
        arm, phase = info
        self.sched.bonus(arm, phase)
        self.save()

    # -- ranking + telemetry ----------------------------------------------
    def rank_bonus(self, elite):
        info = self.origin.get(genome_sig(elite["genome"]))
        if info is None:
            return 0.0
        arm, phase = info
        return 0.5 * self.sched.posterior_mean(arm, phase)

    def report(self, tag):
        lines = [f"[rl] {tag} scheduler snapshot:"]
        for p in ar_rl.PHASES:
            means = ", ".join(f"{a}={self.sched.posterior_mean(a, p):.2f}"
                              for a in ar_rl.ARMS)
            lines.append(f"[rl]   {p}: {means}")
        lines.append(f"[rl]   curiosity top: {self.cur.top(5)}")
        lines.append(f"[rl]   disabled={self.disabled}")
        for ln in lines:
            _say(ln)


_RL_CTL = None


def _rl_controller():
    global _RL_CTL
    if _RL_CTL is None:
        _RL_CTL = _RlController()
    return _RL_CTL


def _rl_controller_reset_for_tests():
    global _RL_CTL
    _RL_CTL = None


def run_qd(train_fn=None):
    """MAP-Elites: illuminate an archive of diverse genomes via the cheap CB screen,
    then full-evaluate + honest-gate the top elites. Returns the archive."""
    from core.features import active_candidate_features

    base_fn = train_base_cached if train_fn is None else train_fn
    train_fn = train_fn or train_env
    base_features = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_z", "rsi",
                     "macd_hist", "bb_pos", "trend_strength", "atr"]
    active = active_candidate_features()
    init = int(os.getenv("GTRADE_AR_QD_INIT", "8"))
    n_final = int(os.getenv("GTRADE_AR_QD_FINAL", "3"))

    # Derived, not hardcoded, so a changed SELECTION_ASSETS still finds its own
    # seeded/measured history bucket (see PROGRESS_SEED's "screen_10").
    _illum_assets = tier_assets() if illum_full() else selection_assets()
    _illum_env = tier_env if illum_full() else screen_env
    if illum_full() and _score_basis() == "raw":
        logger.warning(
            "GTRADE_AR_ILLUM=full on the raw Score basis: net training does not "
            "reproduce on this GPU (same seed, same config, 0.45-1.52 Score apart), "
            "so the archive would rank noise. Use GTRADE_AR_SCORE_BASIS=net_auc.")

    def _illum_rows(rows):
        # Only the full illumination has real nets to re-key onto; under the CB
        # screen every net column is the 0.5 stub, so re-keying there would score
        # every genome identically instead of measuring anything.
        return rekey_rows(rows) if illum_full() else rows

    screen_kind = ("illum_%d" if illum_full() else "screen_%d") % len(
        _illum_assets.split(","))

    _t0 = time.time()
    _mark = _progress_unit_marker()
    screen_base = _illum_rows(base_fn(_illum_assets, _illum_env({})))
    _progress_fold_unit(screen_kind, time.time() - _t0, since=_mark)
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in screen_base}

    try:
        from core import ar_progress as _prog
        _prog.start_heartbeat("agent")
    except Exception:
        pass
    _progress_publish("warmup", step={"kind": "screen", "unit_kind": screen_kind})

    def _screen_eval(g):
        return _illum_rows(train_fn(_illum_assets, _illum_env(genome_to_env(g))))

    archive = _qd_load()
    if not archive:
        for _ in range(init):
            g = _canon_genome(random_genome(active, base_features))
            ar_memory.tried_add("genome", genome_sig(g))
            _t0 = time.time()
            _mark = _progress_unit_marker()
            rows = _screen_eval(g)
            _progress_fold_unit(screen_kind, time.time() - _t0, since=_mark)
            archive_put(archive, g, rows, base_score, active)
        _qd_save(archive)

    # NOTE: by default archive illumination uses the cheap raw CB screen (fitness vs
    # the CB base_score), so GTRADE_AR_SCORE_BASIS only re-scores the FINAL elite gate
    # and does NOT change which genomes become elites - the nets are stubbed out here,
    # so a net basis would score every genome identically. GTRADE_AR_ILLUM=full trains
    # the tier assets with real nets instead, and then the active basis DOES decide
    # which genomes are illuminated. That is the only way a search can hunt net levers.
    max_misses = int(os.getenv("GTRADE_AR_QD_MAX_MISSES", "5"))
    misses = 0
    for _step in range(BUDGET):
        _progress_publish("search", step={"i": _step + 1, "n": BUDGET, "kind": "screen",
                                          "unit_kind": screen_kind})
        if out_of_time():
            print("[qd] time budget spent after %d of %d steps; going to the final "
                  "gate with the archive as it stands." % (_step, BUDGET))
            break
        if not archive:
            break
        child = next_child(archive, active, base_features)
        if child is None:
            misses += 1
            print("[qd] dedup: no unseen child this step, skipping.")
            if misses >= max_misses:
                print("[qd] search space exhausted vs the tried-registry after %d "
                      "misses; stopping early (raise GTRADE_AR_QD_MAX_MISSES to "
                      "keep trying)." % misses)
                break
            continue
        misses = 0
        csig = genome_sig(child)
        ar_memory.tried_add("genome", csig)
        _t0 = time.time()
        _mark = _progress_unit_marker()
        crows = _screen_eval(child)
        _progress_fold_unit(screen_kind, time.time() - _t0, since=_mark)
        stored = archive_put(archive, child, crows, base_score, active)
        if ar_rl.rl_on():
            ctl = _rl_controller()
            ctl.on_result(csig, stored)
            info = ctl.origin.get(csig)
            if info and info[0] == "cma":
                ctl.cma.tell(ctl.cma.vector_of(child),
                             fitness(crows, base_score))
                ctl.save()
        _qd_save(archive)

    if ar_rl.rl_on():
        ctl = _rl_controller()
        ctl.report("run start")
        fits = [e["fitness"] for e in archive.values()]
        mu = sum(fits) / len(fits) if fits else 0.0
        sd = (sum((f - mu) ** 2 for f in fits) / len(fits)) ** 0.5 if fits else 1.0
        sd = sd or 1.0
        elites = sorted(
            archive.values(),
            key=lambda e: (e["fitness"] - mu) / sd + ctl.rank_bonus(e),
            reverse=True)[:n_final]
    else:
        elites = sorted(archive.values(),
                        key=lambda e: e["fitness"], reverse=True)[:n_final]
    if not elites:
        print("[qd] no elites in the archive.")
        finding_winners = []
    else:
        obj = _objective()
        basis = _score_basis()
        # Honest about which units will actually run: with GTRADE_AR_TIER=0 there
        # is no tier check at all, so pending_units and the tier_4 fold must not
        unit_seq = ["tier_4", "holdout_14"] if tier_on() else ["holdout_14"]
        _progress_publish("gate", step={"i": 0, "n": len(elites), "kind": "base_holdout",
                                        "unit_kind": "holdout_14"},
                          pending_units=unit_seq * len(elites))
        _t0 = time.time()
        _mark = _progress_unit_marker()
        _snap = []
        ho_base_full, ho_base_contrib = _heldout_eval(heldout_assets(), {}, base_fn, done_out=_snap)
        _progress_fold_unit("holdout_14", time.time() - _t0, since=_mark,
                            done_pairs=(_snap[0] if _snap else None))
        qd_tier_base = _tier_base(base_fn) if tier_on() else None
        qd_tier_neural = _tier_neural_base(base_fn) if tier_on() else None
        base_contrib = {r["Asset"]: r["Score"] for r in ho_base_contrib}
        results = []
        for _i, e in enumerate(elites, 1):
            g = e["genome"]
            if qd_tier_base is not None:
                _progress_publish("gate", step={"i": _i, "n": len(elites), "kind": "elite_tier",
                                                "unit_kind": "tier_4"},
                                  pending_units=(unit_seq * (len(elites) - _i)) + ["holdout_14"])
                _t0 = time.time()
                _mark = _progress_unit_marker()
                tp, td = _passes_tier(genome_to_env(g), genome_sig(g),
                                      qd_tier_base, obj, train_fn=train_fn)
                _progress_fold_unit("tier_4", time.time() - _t0, since=_mark)
                if not tp:
                    print("[qd] elite tiered out (mini dScore %+.2f): drops=%s "
                          "label=%s/%d" % (td, g.drops, g.label_mode, g.label_window))
                    continue
                nok, nlift = _tier_neural_ok(genome_to_env(g), genome_sig(g),
                                             qd_tier_neural, train_fn=train_fn)
                if not nok:
                    print("[qd] elite tiered out (nets pay for it: tier "
                          "neural_lift %+.2f): drops=%s label=%s/%d"
                          % (nlift, g.drops, g.label_mode, g.label_window))
                    continue
            _progress_publish("gate", step={"i": _i, "n": len(elites), "kind": "elite_holdout",
                                            "unit_kind": "holdout_14"},
                              pending_units=unit_seq * (len(elites) - _i))
            _t0 = time.time()
            _mark = _progress_unit_marker()
            _snap = []
            var_full, var_contrib = _heldout_eval(
                heldout_assets(), genome_to_env(g), train_fn, done_out=_snap)
            _progress_fold_unit("holdout_14", time.time() - _t0, since=_mark,
                                done_pairs=(_snap[0] if _snap else None))
            nl, _d = _objective_delta(var_contrib, base_contrib, "mean")
            nl = round(nl, 4) if _d else None
            if basis == "neural":
                p, value, _d, tag = holdout_stats(ho_base_contrib, var_contrib, obj)
            else:
                _st = _gate_stats(ho_base_full, var_full, obj)
                if _st is None:
                    print("[gate] skipped: the candidate's holdout rows carry no "
                          "column for basis %s" % basis)
                    continue
                p, value, _d, tag = _st
            results.append((g, p, value, tag, nl))
        flags = benjamini_hochberg([r[1] for r in results])
        ts_qd = datetime.utcnow().isoformat()
        finding_winners = []
        for (g, p, value, tag, nl), s in zip(results, flags):
            ok = adopt_ok(s, value, obj, nl)
            replicated = clears = None
            if ok:
                gsig = genome_sig(g)
                replicated = ar_memory.replication_seen(gsig)
                clears = ar_memory.replication_add(gsig, ts_qd)
                if ar_rl.rl_on():
                    _rl_controller().on_adopt(gsig)
                if ar_wiki.wiki_on() and clears >= 2:
                    ar_wiki.note_replicated(gsig, "replicated (%d clears)" % clears)
            finding_winners.append({"axis": "qd", "genome": asdict(g), "p": p,
                                    "value": value, "tag": tag, "adoptable": ok,
                                    "neural_lift": nl, "replicated": bool(replicated),
                                    "clears": clears or 0})
            nl_str = "" if nl is None else f" | neural_lift {nl:+.2f}"
            print("[qd] elite drops=%s label=%s/%d extra=%d: %s | %s%s" % (
                g.drops, g.label_mode, g.label_window, len(g.extra),
                _gate_verdict(ok, bool(replicated), clears, nl, s), tag, nl_str))
    ar_memory.findings_append({
        # _score_basis(), not the local `basis`: that one is bound only inside the
        # elites branch, and a run with an empty archive still journals.
        "ts": datetime.utcnow().isoformat(), "mode": "qd", "basis": _score_basis(),
        "budget": BUDGET, "winners": finding_winners})
    if ar_wiki.wiki_on():
        ar_wiki.compile_wiki()
    mem = ar_memory.findings_summary()
    print("[auto-research] memory: %d experiments tried, %d adoptable, %d replicated so far."
          % (mem["experiments"], mem["adoptable"], mem["replicated"]))
    print("[qd] %d niches illuminated; review _qd_archive.json; nothing auto-adopted." % len(archive))
    if ar_rl.rl_on():
        ctl = _rl_controller()
        ctl.cur.prune({genome_sig(e["genome"]) for e in archive.values()})
        ctl.save()
        ctl.report("run end")
        if ar_wiki.wiki_on():
            try:
                ar_wiki.note_replicated(
                    "rl-scheduler",
                    "posteriors: " + ", ".join(
                        f"{p}/{a}={ctl.sched.posterior_mean(a, p):.2f}"
                        for p in ar_rl.PHASES for a in ar_rl.ARMS))
            except Exception:
                pass
    _progress_publish("done")
    try:
        from core import ar_progress as _prog
        _prog.stop_heartbeat()
    except Exception:
        pass
    return archive


def _regate_candidates(archive_raw, findings, k):
    """Distinct stored candidate genomes to re-gate, capped at k. Findings winners
    (they carry a held-out `value` + neural_lift) rank first by that value; archive-
    only elites (a bare selection `fitness`) fill the remaining slots. Pure - no
    training. archive_raw is the raw JSON dict {cell: {"genome": <dict>, "fitness"}}."""
    by_sig = {}  # sig - (Genome, value, neural_lift)
    basis = _score_basis()
    for rec in findings or []:
        # A held-out `value` is only comparable inside its own basis: a Score
        # delta is ~1-5, a net_auc delta ~0.01, and ranking them in one list puts
        # every legacy Score winner ahead of every AUC winner regardless of merit
        # (measured 2026-08-16: k=8 filled with eight Score-scale genomes whose
        # neural_lift ran -0.49 to -2.38, so the AUC elites were never reached).
        # Records written before this tag existed are raw Score by construction.
        if rec.get("basis", "raw") != basis:
            continue
        for w in rec.get("winners", []):
            gd, v = w.get("genome"), w.get("value")
            if not isinstance(gd, dict) or v is None:
                continue
            try:
                g = Genome(**gd)
            except (TypeError, ValueError):
                continue
            sig = genome_sig(g)
            if sig not in by_sig or v > by_sig[sig][1]:
                by_sig[sig] = (g, v, w.get("neural_lift"))
    found = set(by_sig)
    arch = []
    for cell in (archive_raw or {}).values():
        gd = cell.get("genome") if isinstance(cell, dict) else None
        f = cell.get("fitness") if isinstance(cell, dict) else None
        if not isinstance(gd, dict) or f is None:
            continue
        try:
            g = Genome(**gd)
        except (TypeError, ValueError):
            continue
        if genome_sig(g) not in found:
            arch.append((g, f, None))
    findings_ranked = sorted(by_sig.values(),
                             key=lambda c: (c[1], c[2] if c[2] is not None else -1e9),
                             reverse=True)
    arch_ranked = sorted(arch, key=lambda c: c[1], reverse=True)
    return (findings_ranked + arch_ranked)[:k]


def _regate_load_archive_raw():
    if not os.path.exists(_QD_ARCHIVE_PATH):
        return {}
    try:
        with open(_QD_ARCHIVE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


_REGATE_PROGRESS_PATH = os.path.join(BASE, "_regate_progress.json")


def _say(msg):
    """Print + mirror into the shared log file, so a lost console (the way the
    2026-07-13 re-gate died silently) never loses the run trail."""
    print(msg)
    logger.info(msg)


def _regate_progress_load(base_sig):
    """Per-candidate checkpoint of a re-gate run: {gsig: result fields}. Discarded
    (fresh start) when the stored base signature no longer matches - results are only
    comparable within one data snapshot + objective/basis/screen config."""
    try:
        with open(_REGATE_PROGRESS_PATH, encoding="utf-8") as fh:
            saved = json.load(fh)
    except Exception:
        return {}
    return saved.get("done", {}) if saved.get("base_sig") == base_sig else {}


def _regate_progress_save(base_sig, done):
    with open(_REGATE_PROGRESS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"base_sig": base_sig, "done": done}, fh, ensure_ascii=True, indent=2)


def _regate_progress_clear():
    try:
        os.remove(_REGATE_PROGRESS_PATH)
    except OSError:
        pass


def _candidate_train_cached(subset, env, gsig):
    """train_env with a cross-run cache keyed by the genome signature (the env embeds
    temp spec-file paths, so the raw env cannot key the cache). The CB-only screen
    train and the full train cache under different kinds; the screen's CB train is
    therefore REUSED by the held-out eval instead of being trained twice. A resumed
    or repeated re-gate reuses every finished candidate train (hours each)."""
    kind = "cb" if env.get("GTRADE_SCREEN_ONLY") else "full"
    return _cached_train(subset, env,
                         lambda sub: ar_memory.genome_key(sub, gsig, kind),
                         "regate %s %s" % (kind, gsig[:12]))


def regate(k=8, screen=False):
    """Re-evaluate the best already-found candidate genomes under the CURRENT gate
    (Wilcoxon + enlarged held-out), reusing the 900+ prior experiments instead of
    re-running the search. Trains only the top-k on the held-out set. Adopts nothing;
    journals mode='regate' and feeds the same replication gate.

    Crash-safe: every finished candidate checkpoints to _regate_progress.json and its
    trains cache by genome signature, so an interrupted run RESUMES (same data + gate
    config) instead of restarting from zero."""
    cands = _regate_candidates(_regate_load_archive_raw(), ar_memory.findings_all(), k)
    if not cands:
        _say("[regate] no stored candidates to re-gate.")
        return
    obj, basis = _objective(), _score_basis()
    _say("[regate] %d candidate(s) | objective=%s basis=%s screen=%s"
         % (len(cands), obj, basis, screen))
    ho_base_full, ho_base_contrib = _heldout_eval(heldout_assets(), {}, train_base_cached)
    base_contrib = {r["Asset"]: r["Score"] for r in ho_base_contrib}
    base_sig = "|".join([ar_memory.base_key(heldout_assets(), {}), obj, basis,
                         "screen" if screen else "noscreen"])
    done = _regate_progress_load(base_sig)
    if done:
        _say("[regate] resuming: %d candidate(s) already evaluated." % len(done))
    if screen:
        screen_base = {r["Asset"]: r["Score"]
                       for r in train_base_cached(heldout_assets(), screen_env({}))}
        cands = [c for c in cands
                 if genome_sig(c[0]) in done or _regate_passes_screen(c[0], screen_base)]
        _say("[regate] %d candidate(s) after the CB-only screen." % len(cands))
    results = []
    for i, (g, old_score, old_nl) in enumerate(cands, start=1):
        gsig = genome_sig(g)
        if gsig in done:
            r = done[gsig]
            results.append((g, old_score, r["p"], r["value"], r["tag"], r["nl"]))
            continue
        _say("[regate] %d/%d evaluating %s (stored score %.2f)..."
             % (i, len(cands), gsig[:12], old_score))

        def _fn(s, e, _sig=gsig):
            return _candidate_train_cached(s, e, _sig)

        var_full, var_contrib = _heldout_eval(heldout_assets(), genome_to_env(g), _fn)
        nl, _d = _objective_delta(var_contrib, base_contrib, "mean")
        nl = round(nl, 4) if _d else None
        if basis == "neural":
            p, value, _d, tag = holdout_stats(ho_base_contrib, var_contrib, obj)
        else:
            _st = _gate_stats(ho_base_full, var_full, obj)
            if _st is None:
                _say("[regate] %d/%d skipped: no column for basis %s" % (i, len(cands), basis))
                continue
            p, value, _d, tag = _st
        results.append((g, old_score, p, value, tag, nl))
        done[gsig] = {"p": p, "value": value, "tag": tag, "nl": nl}
        _regate_progress_save(base_sig, done)
        _say("[regate] %d/%d done: %s" % (i, len(cands), tag))
    flags = benjamini_hochberg([r[2] for r in results])
    ts = datetime.utcnow().isoformat()
    finding_winners = []
    for (g, old_score, p, value, tag, nl), s in zip(results, flags):
        ok = adopt_ok(s, value, obj, nl)
        replicated = clears = None
        if ok:
            gsig = genome_sig(g)
            replicated = ar_memory.replication_seen(gsig)
            clears = ar_memory.replication_add(gsig, ts)
        finding_winners.append({"axis": "regate", "genome": asdict(g), "p": p,
                                "value": value, "tag": tag, "adoptable": ok,
                                "neural_lift": nl, "replicated": bool(replicated),
                                "clears": clears or 0})
        nl_str = "" if nl is None else f" | neural_lift {nl:+.2f}"
        _say(f"[regate] old {old_score:.2f} - "
             f"{_gate_verdict(ok, bool(replicated), clears, nl, s)} | {tag}{nl_str}")
    ar_memory.findings_append({"ts": ts, "mode": "regate", "k": k, "basis": basis,
                               "screen": bool(screen), "winners": finding_winners})
    _regate_progress_clear()
    _say("[regate] re-gated %d candidate(s); nothing adopted automatically." % len(results))


def _regate_passes_screen(g, screen_base):
    """Optional cheap CB-only prefilter: keep the candidate unless its CB-only held-out
    mean delta vs the CB-only base is clearly negative (below the existing SCREEN_MIN
    floor). `screen_base` is a {asset: CB-only Score} dict. The train is cached by
    genome signature, so the held-out eval reuses it as its CB side."""
    try:
        rows = _candidate_train_cached(heldout_assets(), screen_env(genome_to_env(g)),
                                       genome_sig(g))
        d, deltas = _objective_delta(rows, screen_base, "mean")
        return (not deltas) or d >= SCREEN_MIN
    except Exception:
        return True


_SAMPLE_WINDOWS = [5, 10, 20, 50]
_SAMPLE_KS = [1, 2, 3, 5, 10]
_SAMPLE_HORIZONS = [1, 2, 5]
_SINGLE_OPS = ["zscore", "lag", "diff", "rolling"]
_PAIR_OPS = ["ratio", "interaction"]
_AGGS = ["mean", "std", "sum"]
_LEAD_LEADERS = ["sp500", "vix", "btc", "gold", "dxy", "tnx"]


def _random_spec(base_features, name, prefer):
    """One spec sampled from the DSL space. prefer biases the input choice toward
    columns that have shown positive deltas; it falls back to base_features."""
    pool = prefer or base_features
    if len(base_features) >= 2 and random.random() < 0.35:
        op = random.choice(_PAIR_OPS)
        a = random.choice(pool)
        b = random.choice([f for f in base_features if f != a])
        return {"name": name, "op": op, "inputs": [a, b], "params": {}}
    if random.random() < 0.12:
        return {"name": name, "op": "lead_lag",
                "inputs": [random.choice(_LEAD_LEADERS)],
                "params": {"horizon": random.choice(_SAMPLE_HORIZONS)}}
    op = random.choice(_SINGLE_OPS)
    params = {}
    if op in ("zscore", "rolling"):
        params["window"] = random.choice(_SAMPLE_WINDOWS)
        if op == "rolling":
            params["agg"] = random.choice(_AGGS)
    else:
        params["k"] = random.choice(_SAMPLE_KS)
    return {"name": name, "op": op, "inputs": [random.choice(pool)], "params": params}


def _mutate(spec, name):
    """A small variation of a past spec: tweak its one numeric param, new name."""
    out = {"name": name, "op": spec["op"],
           "inputs": list(spec.get("inputs") or []),
           "params": dict(spec.get("params") or {})}
    p = out["params"]
    if "window" in p:
        p["window"] = random.choice(_SAMPLE_WINDOWS)
    elif "k" in p:
        p["k"] = random.choice(_SAMPLE_KS)
    elif "horizon" in p:
        p["horizon"] = random.choice(_SAMPLE_HORIZONS)
    if out["op"] == "rolling":
        p["agg"] = random.choice(_AGGS)
    return out


def propose_evolutionary(log, base_features, avoid=None):
    """Autonomous (no LLM) proposer. Explores the DSL early; once a past spec shows
    a positive mean selection delta it biases input choice toward the good inputs and
    mutates the best spec. Reads the log, dedups against it, returns one valid spec.
    avoid is accepted for a uniform proposer signature but unused (this path already
    dedups against the log and the tried registry)."""
    seed = os.getenv("GTRADE_AR_SEED")
    if seed:
        random.seed(int(seed) + len(log))
    cols = set(base_features)
    seen = set()
    scored = []
    good_inputs = []
    for e in log:
        for s in (e.get("spec") or []):
            seen.add(_spec_signature(s))
        sc = e.get("score")
        if sc is not None and e.get("spec"):
            scored.append((sc, e["spec"]))
            if sc > 0:
                for s in e["spec"]:
                    good_inputs += [c for c in (s.get("inputs") or []) if c in cols]
    best = max(scored, key=lambda x: x[0]) if scored else None
    for attempt in range(30):
        name = "ar_%d_%d" % (len(log), attempt)
        if best and best[0] > 0 and random.random() < 0.6:
            spec = _mutate(random.choice(best[1]), name)
        else:
            spec = _random_spec(base_features, name, good_inputs)
        if (validate_spec(spec, cols) and _spec_signature(spec) not in seen
                and not ar_memory.tried_seen("spec", json.dumps(_spec_signature(spec)))):
            return [spec]
    return []


def _select_proposer():
    """Evolutionary (no LLM) by default; the LLM proposer when GTRADE_AR_PROPOSER=llm."""
    if llm_proposer.llm_selected():
        return llm_proposer.propose_specs
    return propose_evolutionary


PRESCREEN_MIN_ABS_CORR = float(os.getenv("AR_PRESCREEN_MIN", "0.02"))


def _prescreen_ok(spec, df, target_col="target", threshold=None):
    """Cheap univariate screen run BEFORE the expensive training: keep a spec only
    if its materialized feature has at least a small absolute correlation with the
    target. lead_lag (an engine op), a missing target, or any error passes through,
    to be judged by the full A/B."""
    if threshold is None:
        threshold = PRESCREEN_MIN_ABS_CORR
    if spec.get("op") == "lead_lag" or target_col not in getattr(df, "columns", []):
        return True
    try:
        import pandas as pd

        from core.feature_dsl import materialize
        c = pd.Series(materialize(df, spec)).corr(df[target_col])
        if pd.isna(c):  # NaN correlation (constant feature)
            return True
        return abs(c) >= threshold
    except Exception:
        return True


SCREEN_MIN = float(os.getenv("GTRADE_AR_SCREEN_MIN", "0.0"))


def _screen_on():
    """Whether the cheap CatBoost-only screen runs before the full eval.

    The DEFAULT follows the basis (auto_loop.default_screen): off on a net
    basis, where the screen stubs every neural member to a constant and so
    discards net levers on CatBoost's opinion; on everywhere else. It was a
    blanket "on", which made raw + full the shape a launcher produced by
    answering its own defaults - and that pairing is refused.
    """
    from auto_loop import default_screen

    v = os.getenv("GTRADE_AR_SCREEN")
    if v is None:
        v = default_screen(_score_basis())
    return (v or "1").strip() not in ("0", "false", "False", "")


def _load_history(record):
    """The history bucket out of a read agent record, seeded via a DEEP copy so
    folding never mutates the PROGRESS_SEED module constant: dict(PROGRESS_SEED)
    only copies the top level, so history["assets"] would still be the exact
    same nested dict object as PROGRESS_SEED["assets"], and every fold on a
    fresh progress file would then permanently corrupt the seed for the rest of
    the process.

    Also discards a legacy FLAT assets bucket - {"USDJPY": [7200], ...}, asset
    name straight to a list of numbers, from before per-unit-kind keying (see
    _progress_fold_unit) - rather than migrating it: those samples are exactly
    the cross-population contamination keying exists to prevent, so keeping
    them around under any key would just re-poison the estimate they are
    meant to feed.
    """
    history = record.get("history") or copy.deepcopy(PROGRESS_SEED)
    assets = history.get("assets")
    if not isinstance(assets, dict) or any(not isinstance(v, dict) for v in assets.values()):
        assets = {}
    history["assets"] = assets
    return history


def _progress_publish(phase, step=None, pending_units=None):
    """Publish where the run is. Fail-safe: progress must never break the run."""
    try:
        from core import ar_progress
        record = ar_progress.read_agent()
        history = _load_history(record)
        ar_progress.write_agent({
            "run_started": record.get("run_started") or datetime.utcnow().isoformat(),
            "phase": phase,
            "step": step or {},
            "pending_units": pending_units or [],
            "results": ar_memory.findings_summary(),
            "history": history,
        })
        if step:
            print("[progress] {}: step {}/{} ({})".format(phase, step.get("i"), step.get("n"), step.get("kind") or ""))
        else:
            print(f"[progress] {phase}")
    except Exception:
        pass


_NO_MARK = object()


def _progress_unit_marker():
    """The training-unit file's 'started' stamp right now (None if there is no
    unit file yet), for a caller to capture BEFORE an evaluation and pass back
    to _progress_fold_unit as since=. An unchanged stamp after the evaluation
    means no unit_begin ran, i.e. the evaluation was satisfied entirely from
    cache and the unit file was never touched."""
    try:
        from core import ar_progress
        return ar_progress.read_unit().get("started")
    except Exception:
        return None


def _progress_unit_done_since(mark):
    """The unit file's current per-asset done pairs, but only if a fresh
    unit_begin happened since `mark` (see _progress_unit_marker); None when the
    stamp is unchanged, because then the done list on disk belongs to whatever
    training ran before, not to the one the caller is trying to measure."""
    try:
        from core import ar_progress
        rec = ar_progress.read_unit()
        if rec.get("started") == mark:
            return None
        return rec.get("done") or []
    except Exception:
        return None


def _progress_fold_unit(unit_kind, seconds, since=_NO_MARK, done_pairs=_NO_MARK):
    """Record a finished unit: its wall time, and each asset's own service
    time, both keyed under history["assets"][unit_kind] so populations with
    wildly different per-asset costs (a screen unit finishes an asset in
    seconds; a holdout/tier unit takes hours for that very same asset) never
    share a bucket. A holdout estimate only ever reads
    history["assets"]["holdout_14"]; nothing a screen or tier unit folds can
    land there, so keying by kind is what makes folding every unit's per-asset
    times safe. (Earlier this was gated by a fold_assets=False flag that kept
    screen units out of one flat history["assets"] dict entirely; keying
    replaced that gate structurally, and the flag was removed.)

    since, when given (see _progress_unit_marker), is the unit file's 'started'
    stamp captured right BEFORE the evaluation ran; if it is unchanged
    afterward, no training actually happened (a cache hit) and the fold is
    skipped entirely - neither the wall time nor any per-asset time - because
    folding a cache hit would otherwise record a near-zero wall time and
    re-absorb the PREVIOUS unit's per-asset samples under this unit_kind's
    label. The wall time is also floored to at least one second so a
    sub-second measurement can never enter a median.

    done_pairs, when given (see _heldout_eval's done_out), overrides the live
    unit-file read for the per-asset fold: use exactly this list (possibly
    empty or None) instead of ar_progress.read_unit()["done"], because for a
    _heldout_eval-driven fold the live file may already belong to the CB-only
    train that ran after the one this measurement is actually for.
    """
    try:
        from core import ar_progress
        if since is not _NO_MARK and ar_progress.read_unit().get("started") == since:
            return
        record = ar_progress.read_agent()
        history = _load_history(record)
        if unit_kind and seconds:
            history[unit_kind] = (history.get(unit_kind) or [])[-(PROGRESS_KEEP - 1):] + [max(1, int(seconds))]
        if unit_kind:
            pairs = ar_progress.read_unit().get("done") or [] if done_pairs is _NO_MARK else done_pairs or []
            bucket = history["assets"].setdefault(unit_kind, {})
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                asset, took = pair
                if not took:
                    continue
                prior = bucket.get(asset) or []
                bucket[asset] = prior[-(PROGRESS_KEEP - 1):] + [int(took)]
        record["history"] = history
        ar_progress.write_agent(record)
    except Exception:
        pass


def screen_env(env):
    """A copy of the candidate env with the CB-only screen flag set."""
    return {**env, "GTRADE_SCREEN_ONLY": "1"}


def _passes_screen(axis, selected, train_fn, screen_base, screen_min, objective="mean"):
    """Cheap CB-only screen of one candidate. Returns (passed, proxy_delta). A failed
    screen train (empty rows) returns (True, 0.0) - never drop a candidate on a screen
    infra failure, fall through to the full eval."""
    srows = train_fn(selection_assets(), screen_env(axis.to_env(selected)))
    if not srows:
        return True, 0.0
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in screen_base}
    delta, _ = _objective_delta(srows, base_score, objective)
    return delta > screen_min, delta


# --- tier ladder (axis J): a cheap mini-eval between the CB screen and the ----
# full train. 4 assets at roughly half epochs; drop rule mirrors the screen
# (clearly-negative candidates only; an infra failure passes through).

TIER_ENV = {"GTRADE_EPOCHS_LSTM": "45", "GTRADE_EPOCHS_TF": "30",
            "GTRADE_EPOCHS_TCN": "25"}


def tier_on():
    return (os.getenv("GTRADE_AR_TIER", "1") or "1").strip() not in (
        "0", "false", "False")


# Wall-clock stop. The budget counts GENOMES, and a genome costs anywhere from
# 43s (screened out) to 33min (a full held-out eval), so a budget of 30 buys a
# run somewhere between twenty minutes and sixteen hours. When what is actually
# scarce is the night, count the night.
_RUN_STARTED = time.time()


def _time_budget_s():
    try:
        h = float(os.getenv("GTRADE_AR_TIME_BUDGET_H") or 0.0)
    except ValueError:
        return 0.0
    return h * 3600.0 if h > 0 else 0.0


def out_of_time():
    """Whether the run has spent its wall-clock budget. False when there is none.

    Checked BETWEEN candidates, never inside one: a half-trained candidate is
    not a cheaper candidate, it is a candidate whose value nobody knows, and
    everything already finished is journalled either way."""
    budget = _time_budget_s()
    return bool(budget) and (time.time() - _RUN_STARTED) >= budget


def illum_full():
    """Whether the QD search illuminates on REAL nets (GTRADE_AR_ILLUM=full).

    "cb" is the historical cheap screen: 10 assets, every neural member replaced
    by a constant 0.5. That makes the archive - and therefore every elite the run
    ever proposes - a pure CatBoost selection, which is why mutate(ops=["nets"])
    is absent from the bandit arms and why a net basis scores every genome
    identically here. On "full" the search trains the tier assets with the tier's
    reduced epochs instead, so the nets are real and the active basis (net_auc
    etc.) actually means something during illumination. Costs roughly 12x per
    genome.

    The DEFAULT now follows the basis (auto_loop.default_illum): full on a net
    basis, cb on raw and neural where net training does not reproduce here. It
    used to be a blanket "cb", so a net-basis run started without the variable
    searched on CatBoost and let the basis re-score only the final gate."""
    from auto_loop import default_illum

    return (os.getenv("GTRADE_AR_ILLUM")
            or default_illum(_score_basis())).strip().lower() == "full"


def tier_assets():
    return os.getenv("GTRADE_AR_TIER_ASSETS") or "SP500,BTC,EURUSD,GOLD"


def tier_env(env):
    return {**env, **TIER_ENV}


def _rekeyed(rows, basis_name="the active basis", basis=None):
    """rekey_rows, but None when the rows predate the column the basis needs.

    The mini-run cache stores RAW rows so one training run is reusable across
    bases. A set trained before Ens_AUC/Net_AUC existed still loads fine and
    re-keys to an EMPTY list, which would sail through the tier as "no opinion"
    and spend a full evaluation on a candidate nobody measured. Empty-out means
    stale, not neutral: the caller must retrain.
    """
    out = rekey_rows(rows, basis)
    return out if out or not rows else None


def _gate_stats(base_full, var_full, obj):
    """`holdout_stats` for the adoption gate, on the ACTIVE basis.

    Only the `neural` basis ever had its own branch at the three call sites, so
    net_auc / net_gain / ens_auc fell through to the raw ensemble Score - while
    `adopt_floor` had ALREADY switched to AUC units for exactly those bases. The
    verdict then compared a value in Score against a floor in AUC. Measured
    2026-08-16: a search that had found genomes worth +0.065 Net_AUC, thirteen
    times that floor, was rejected on a money Score nobody asked it to optimize,
    and the neural finding was never tested at all.

    `rekey_rows` is the identity on the raw basis, so that path is unchanged.

    Returns None when the CANDIDATE rows carry no column for this basis: empty
    means stale, not neutral (see `_rekeyed`), and the caller must not read a
    vacuous pass out of it. A stale BASE is retrained instead, the way
    `_tier_base` handles the same trap.
    """
    base = _rekeyed(base_full)
    if base is None:
        base = rekey_rows(train_env(heldout_assets(), {}))
    var = _rekeyed(var_full)
    if var is None:
        return None
    return holdout_stats(base, var, obj)


def _tier_base(base_fn):
    """Mini-tier BASE rows (cached via base_fn = train_base_cached), on the
    ACTIVE basis - the candidate side is re-keyed the same way in _passes_tier."""
    rows = base_fn(tier_assets(), tier_env({}))
    out = _rekeyed(rows)
    if out is None:
        # cached from before this basis existed: bypass the cache and train
        out = rekey_rows(train_env(tier_assets(), tier_env({})))
    return out


def tier_neural_floor():
    """How far the nets may fall on the TIER before a candidate is dropped.

    Looser than neural_floor() on purpose, twice it by default. The tier is four
    assets at half epochs and two of them sit outside the set where the stacker
    leans on the nets, so the reading is noisy and biased towards zero. A false
    pass costs one held-out evaluation; a false reject costs a genome nobody
    looks at again. Override with GTRADE_AR_TIER_NEURAL_MIN.
    """
    try:
        v = os.getenv("GTRADE_AR_TIER_NEURAL_MIN")
        return float(v) if v else 2.0 * neural_floor()
    except ValueError:
        return 2.0 * neural_floor()


def _tier_neural_base(base_fn):
    """The BASE's neural contribution on the tier assets as {asset: score}.

    A dict, not rows: _objective_delta takes its base side keyed by asset, the
    same shape the held-out path builds at its own neural_lift call. Both trains
    go through base_fn, so they are cached for the whole run the way _tier_base's
    is - the base pays for this once.
    """
    full = base_fn(tier_assets(), tier_env({}))
    cb = base_fn(tier_assets(), tier_env(screen_env({})))
    if not full or not cb:
        return None
    return neural_contribution(full, cb)


def _tier_neural_ok(env, cache_key, base_contrib, train_fn=None):
    """(ok, lift) - is this candidate paying for its CatBoost gain in nets?

    The question adopt_ok asks with neural_lift, asked at TIER cost instead of
    holdout cost. The search fitness is the CatBoost-only screen, so a genome is
    free to win by starving the sequence members, and until now the only thing
    that noticed was the final gate. Measured 2026-08-29: two elites cleared
    significance (p=0.018 and p=0.003) and the effect floor (0.03 and 0.04
    against 0.005) and were both refused for neural_lift -1.15 and -1.29 - after
    each had spent a full held-out evaluation, 5438 s of nets plus 197 s of
    CatBoost on 14 assets.

    Here the full tier rows are already in the cache _passes_tier just filled,
    so the only new work is the CatBoost-only counterpart on the same four
    assets. An unmeasurable lift never blocks, the same convention adopt_ok
    uses for neural_lift=None.
    """
    if not base_contrib:
        return True, None
    if train_fn is None:
        train_fn = train_env
    full = ar_memory.cache_get(
        ar_memory.genome_key(tier_assets(), cache_key, "mini"))
    if not full:
        return True, None
    key_cb = ar_memory.genome_key(tier_assets(), cache_key, "mini_cb")
    cb = ar_memory.cache_get(key_cb)
    if cb is None:
        cb = train_fn(tier_assets(), tier_env(screen_env(env)))
        if cb:
            ar_memory.cache_put(key_cb, cb)
    if not cb:
        return True, None
    contrib = [{"Asset": a, "Score": c}
               for a, c in neural_contribution(full, cb).items()]
    lift, measured = _objective_delta(contrib, dict(base_contrib), "mean")
    if not measured:
        return True, None
    return lift > tier_neural_floor(), round(lift, 4)


def _tier_key(axis, cand):
    """Stable cache identity for an axis candidate (genomes use genome_sig)."""
    if axis.sig is not None:
        cands = cand if isinstance(cand, list) else [cand]
        return "|".join(axis.sig(c)[1] for c in cands)
    return json.dumps(cand, sort_keys=True, default=str)


def _passes_tier(env, cache_key, tier_base, objective="mean", train_fn=None):
    """(passed, delta). Candidate mini rows are cached by cache_key (kind
    "mini") so a resumed or repeated search reuses them, like the regate.
    train_fn defaults to train_env, resolved at call time (like _passes_screen,
    the caller's injected trainer flows through)."""
    if not tier_base:
        return True, 0.0
    if train_fn is None:
        train_fn = train_env
    key = ar_memory.genome_key(tier_assets(), cache_key, "mini")
    rows = ar_memory.cache_get(key)
    if rows is None:
        rows = train_fn(tier_assets(), tier_env(env))
        if rows:
            ar_memory.cache_put(key, rows)
    if not rows:
        return True, 0.0
    # Cache stores RAW rows (basis-independent, so a cached mini run stays valid
    # across bases); the basis is applied on read. Rows older than the basis
    # column re-key to nothing - drop them and train rather than pass vacuously.
    out = _rekeyed(rows)
    if out is None:
        rows = train_fn(tier_assets(), tier_env(env))
        if rows:
            ar_memory.cache_put(key, rows)
        out = rekey_rows(rows)
        if not out:
            return True, 0.0
    rows = out
    base_score = {r["Asset"]: r.get("Score", 0.0) for r in tier_base}
    d, deltas = _objective_delta(rows, base_score, objective)
    try:
        tier_min = float(os.getenv("GTRADE_AR_TIER_MIN") or "0.0")
    except ValueError:
        tier_min = 0.0
    return (not deltas) or d >= tier_min, d


_STATE_PATH = os.path.join(BASE, "_auto_research_state.json")
_LOG_PATH = os.path.join(BASE, "_auto_research_log.json")


def load_state():
    """The persisted loop state (cached base, kept set, log) for resume, or {}."""
    if not os.path.exists(_STATE_PATH):
        return {}
    try:
        with open(_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)
    # keep the human-readable log file in sync
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(state.get("log", []), f, ensure_ascii=True, indent=2)


@dataclass
class Axis:
    """One search axis. `propose(log)` returns candidate dicts; `to_env(selected)`
    turns the selected candidate (additive: the kept+new list; select_best: a single
    candidate) into training env overrides; `kind` is "additive" or "select_best";
    `validate(cand, selected)` and `prescreen(cand, screen_df)` default to accepting;
    `sig(cand)` returns (kind, signature) for the permanent tried-registry; None = not registered.

    cb_blind marks an axis whose lever the CatBoost-only screen cannot see at all,
    so the screen must not judge it. Under GTRADE_SCREEN_ONLY the nets are replaced
    by a neutral 0.5 and never train, which makes every net-only env key inert -
    the screen then compares two identical CatBoost runs and passes or fails the
    candidate on training noise. The 2026-08 log shows exactly that: all three
    `weighting` candidates screened out, with {cb_uniqueness} and
    {net_uniqueness, cb_uniqueness} scoring byte-identical deltas (adding the net
    flag changed nothing) while the certified no-op {net_uniqueness} alone read
    -1.71. A cb_blind axis skips the screen and is filtered by the tier ladder
    instead, which does train the nets."""
    name: str
    propose: object
    to_env: object
    kind: str = "additive"
    validate: object = None
    prescreen: object = None
    sig: object = None
    cb_blind: bool = False

    def ok(self, cand, selected, screen_df):
        v = self.validate is None or self.validate(cand, selected)
        p = self.prescreen is None or self.prescreen(cand, screen_df)
        return v and p


def run_axis(axis, budget, base_rows, train_fn, screen_df=None, prior_log=None, persist=None,
             screen_base=None, screen_min=0.0, tier_base=None,
             tier_neural_base=None):
    """Generalized search over one axis, sharing the mean-delta gate.

    additive: forward-selection - keep the running list when the cumulative mean
    Score delta improves. select_best: evaluate each proposed candidate against the
    base and keep the single best whose delta beats the base. `persist(log)` is
    called after each iteration (None = no-op).
    budget counts NEW iterations for THIS run (the persisted log only sets the
    resume point, it does not consume budget)."""
    def _mark_tried(cands):
        if axis.sig is None:
            return
        for c in (cands if isinstance(cands, list) else [cands]):
            kind, s = axis.sig(c)
            ar_memory.tried_add(kind, s)

    base_score = {r["Asset"]: r.get("Score", 0.0) for r in base_rows}
    objective = _objective()
    log = list(prior_log or [])
    persist = persist or (lambda _log: None)
    # Per-axis entries always carry axis == axis.name; filter explicitly rather than
    # relying incidentally on the caller (_persist) bucketing the log per axis.
    axis_log = [e for e in log if e.get("axis") in (None, axis.name)]

    if axis.kind == "additive":
        kept = [c for e in axis_log if e.get("accepted") for c in e.get("cand", [])]
        kept_delta = max((e.get("cand_mean_delta", 0.0) for e in axis_log if e.get("accepted")), default=0.0)
        start = len(axis_log)
        for i in range(start, start + budget):
            if out_of_time():
                print("[auto-research] axis %s: time budget spent after %d of %d "
                      "candidates; gating what was found." % (axis.name, i - start, budget))
                break
            proposed = axis.propose(log)
            new = [c for c in proposed if axis.ok(c, kept, screen_df)]
            if not new:
                log.append({"axis": axis.name, "iter": i, "cand": [], "note": "no valid/screened candidate"})
                persist(log)
                continue
            cand = kept + new
            if screen_base is not None and not axis.cb_blind:
                passed, sdelta = _passes_screen(axis, cand, train_fn, screen_base, screen_min, objective)
                if not passed:
                    entry = {"axis": axis.name, "iter": i, "cand": new,
                             "screen_delta": sdelta, "note": "screened out"}
                    log.append(entry)
                    _mark_tried(new)
                    persist(log)
                    continue
            if tier_base is not None:
                tpassed, tdelta = _passes_tier(axis.to_env(cand),
                                               _tier_key(axis, new), tier_base, objective,
                                               train_fn=train_fn)
                if not tpassed:
                    log.append({"axis": axis.name, "iter": i, "cand": new,
                                "tier_delta": tdelta, "note": "tiered out"})
                    _mark_tried(new)
                    persist(log)
                    continue
                nok, nlift = _tier_neural_ok(axis.to_env(cand),
                                             _tier_key(axis, new),
                                             tier_neural_base, train_fn=train_fn)
                if not nok:
                    log.append({"axis": axis.name, "iter": i, "cand": new,
                                "tier_neural_lift": nlift,
                                "note": "nets pay for it"})
                    _mark_tried(new)
                    persist(log)
                    continue
            rows = score_rows(selection_assets(), axis.to_env(cand), train_fn)
            delta, _ = _objective_delta(rows, base_score, objective)
            entry = {"axis": axis.name, "iter": i, "cand": new,
                     "cand_mean_delta": delta, "score": delta - kept_delta}
            if delta - kept_delta > 1e-9:
                kept, kept_delta = cand, delta
                entry["accepted"] = True
            log.append(entry)
            _mark_tried(new)
            persist(log)
        return {"axis": axis.name, "kept": kept, "kept_delta": kept_delta, "log": log}

    # select_best
    tried = {json.dumps(e["cand"], sort_keys=True) for e in axis_log
             if "cand" in e and isinstance(e["cand"], dict)}
    accepted = [e for e in axis_log if e.get("accepted")]
    if accepted:
        _be = max(accepted, key=lambda e: e.get("cand_mean_delta", 0.0))
        best, best_delta = _be["cand"], _be.get("cand_mean_delta", 0.0)
    else:
        best, best_delta = None, 0.0
    proposed = [c for c in axis.propose(log) if axis.ok(c, best, screen_df)]
    i = len([e for e in axis_log if "iter" in e])
    stop = i + budget
    for cand in proposed:
        if i >= stop:
            break
        if out_of_time():
            print("[auto-research] axis %s: time budget spent; gating what was found."
                  % axis.name)
            break
        if json.dumps(cand, sort_keys=True) in tried:
            continue
        if screen_base is not None and not axis.cb_blind:
            passed, sdelta = _passes_screen(axis, cand, train_fn, screen_base, screen_min, objective)
            if not passed:
                log.append({"axis": axis.name, "iter": i, "cand": cand,
                            "screen_delta": sdelta, "note": "screened out"})
                _mark_tried(cand)
                persist(log)
                i += 1
                continue
        if tier_base is not None:
            tpassed, tdelta = _passes_tier(axis.to_env(cand),
                                           _tier_key(axis, cand), tier_base, objective,
                                           train_fn=train_fn)
            if not tpassed:
                log.append({"axis": axis.name, "iter": i, "cand": cand,
                            "tier_delta": tdelta, "note": "tiered out"})
                _mark_tried(cand)
                persist(log)
                i += 1
                continue
            nok, nlift = _tier_neural_ok(axis.to_env(cand),
                                         _tier_key(axis, cand),
                                         tier_neural_base, train_fn=train_fn)
            if not nok:
                log.append({"axis": axis.name, "iter": i, "cand": cand,
                            "tier_neural_lift": nlift, "note": "nets pay for it"})
                _mark_tried(cand)
                persist(log)
                i += 1
                continue
        rows = score_rows(selection_assets(), axis.to_env(cand), train_fn)
        delta, _ = _objective_delta(rows, base_score, objective)
        entry = {"axis": axis.name, "iter": i, "cand": cand, "cand_mean_delta": delta}
        if delta > best_delta + 1e-9:
            best, best_delta = cand, delta
            entry["accepted"] = True
        log.append(entry)
        _mark_tried(cand)
        persist(log)
        i += 1
    return {"axis": axis.name, "best": best, "best_delta": best_delta, "log": log}


def make_features_axis(base_features):
    """The existing feature-DSL search as an axis (behavior-preserving). validate
    grows the available-column set with the names of already-kept specs, matching the
    old forward-selection."""
    proposer = _select_proposer()

    def _validate(cand, kept):
        cols = set(base_features) | {s["name"] for s in (kept or [])}
        return validate_spec(cand, cols)

    return Axis(
        name="features",
        propose=lambda log: proposer(log, base_features,
                                     avoid=ar_memory.tried_recent("spec", 30)),
        to_env=lambda selected: _feature_env(selected, [s["name"] for s in selected]),
        kind="additive",
        validate=_validate,
        prescreen=lambda cand, screen_df: True if screen_df is None else _prescreen_ok(cand, screen_df),
        sig=lambda c: ("spec", json.dumps(_spec_signature(c))),
    )


LABEL_WINDOWS = (20, 30, 60)


def _label_env(cand):
    """Env overrides for one labeling candidate. triple_barrier's window is its
    horizon H (same convention as the genome's label_window)."""
    if cand["mode"] == "triple_barrier":
        return {"GTRADE_LABEL_MODE": "triple_barrier",
                "GTRADE_LABEL_HORIZON": str(cand["window"])}
    return {"GTRADE_LABEL_MODE": cand["mode"],
            "GTRADE_LABEL_WINDOW": str(cand["window"])}


def make_labeling_axis():
    """Sweep the alternative label modes (direction is the base): rel_median windows
    and triple_barrier horizons. select_best: keep the single best candidate whose
    mean Score delta beats the base. Unlike the 2026-07-11 manual A/B, a
    triple_barrier candidate here does NOT bundle uniqueness weighting - that is a
    separate net-hygiene gene/axis, so the two effects are measured apart."""
    def _propose(log):
        tried = {(e["cand"].get("mode"), e["cand"].get("window")) for e in log
                 if isinstance(e.get("cand"), dict) and "window" in e["cand"]}
        cands = ([{"mode": "rel_median", "window": w} for w in LABEL_WINDOWS]
                 + [{"mode": "triple_barrier", "window": h} for h in TB_HORIZONS])
        cands = [c for c in cands if (c["mode"], c["window"]) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "label", json.dumps(c, sort_keys=True))]

    return Axis(
        name="labeling",
        propose=_propose,
        to_env=_label_env,
        kind="select_best",
        sig=lambda c: ("label", json.dumps(c, sort_keys=True)),
    )


# --- model-hyperparameter axis (relative deltas/multipliers, see Genome) -----

HYPER_CANDIDATES = (
    {"cb_depth_delta": -1}, {"cb_depth_delta": 1},
    {"cb_lr_mult": 0.5}, {"cb_lr_mult": 2.0},
    {"cb_iter_mult": 1.5}, {"cb_iter_mult": 0.7},
    {"lookback_delta": -5}, {"lookback_delta": 5}, {"lookback_delta": 10},
)

_HYPER_ENV_KEYS = {
    "cb_depth_delta": "GTRADE_CB_DEPTH_DELTA",
    "cb_lr_mult": "GTRADE_CB_LR_MULT",
    "cb_iter_mult": "GTRADE_CB_ITER_MULT",
    "lookback_delta": "GTRADE_LOOKBACK_DELTA",
}


def hyper_env(cand):
    """Env overrides for one hyperparameter candidate dict (one or more genes)."""
    return {_HYPER_ENV_KEYS[k]: str(v) for k, v in cand.items()}


def make_hyper_axis():
    """One-gene-at-a-time sweep of the relative model-hyperparameter overrides
    (train_hybrid applies them on top of each asset's optuna baseline).
    select_best: keep the single best override whose delta beats the base."""
    def _propose(log):
        tried = {json.dumps(e["cand"], sort_keys=True) for e in log
                 if isinstance(e.get("cand"), dict)}
        cands = [c for c in HYPER_CANDIDATES
                 if json.dumps(c, sort_keys=True) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "hyper", json.dumps(c, sort_keys=True))]

    return Axis(
        name="hyper",
        propose=_propose,
        to_env=hyper_env,
        kind="select_best",
        sig=lambda c: ("hyper", json.dumps(c, sort_keys=True)),
    )


# --- net-hygiene axis (levers as searchable candidates) ------------------

NETS_CANDIDATES = (
    {"seeds": 3},
    {"calibrate": 1},
    {"seeds": 3, "calibrate": 1},
)

_NETS_ENV_KEYS = {"seeds": "GTRADE_NET_SEEDS", "uniqueness": "GTRADE_NET_UNIQUENESS",
                  "calibrate": "GTRADE_NET_CALIBRATE"}


def nets_env(cand):
    return {_NETS_ENV_KEYS[k]: str(v) for k, v in cand.items()}


def make_nets_axis():
    """Sweep the net-hygiene levers (seed-averaging, per-net calibration).
    Uniqueness weighting is intentionally NOT proposed here - it lives on its own
    `weighting` axis because it is a SHARED lever (CatBoost takes the same weights),
    and it stays a no-op until the label is multi-bar; the QD genome can still
    combine it with a triple_barrier label."""
    def _propose(log):
        tried = {json.dumps(e["cand"], sort_keys=True) for e in log
                 if isinstance(e.get("cand"), dict)}
        cands = [c for c in NETS_CANDIDATES
                 if json.dumps(c, sort_keys=True) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "nets", json.dumps(c, sort_keys=True))]

    return Axis(
        name="nets",
        propose=_propose,
        to_env=nets_env,
        kind="select_best",
        sig=lambda c: ("nets", json.dumps(c, sort_keys=True)),
        cb_blind=True,   # seeds/calibrate do nothing when the nets are stubbed out
    )


# --- weighting axis (label-uniqueness sample weights, per learner) -----------

WEIGHTING_CANDIDATES = (
    {"net_uniqueness": 1},
    {"cb_uniqueness": 1},
    {"net_uniqueness": 1, "cb_uniqueness": 1},
)

_WEIGHTING_ENV_KEYS = {"net_uniqueness": "GTRADE_NET_UNIQUENESS",
                       "cb_uniqueness": "GTRADE_CB_UNIQUENESS"}


def weighting_env(cand):
    return {_WEIGHTING_ENV_KEYS[k]: str(v) for k, v in cand.items()}


def _weighting_sig(cand):
    """Registry signature for a weighting candidate, qualified by the active label
    mode. The same on/off combination is a genuinely different experiment under a
    multi-bar label than under the next-bar default, where the uniqueness weights
    are all-ones and every candidate is a no-op. Without the qualifier, one no-op
    run would mark all three candidates tried forever and the real run would
    propose nothing."""
    mode = (os.getenv("GTRADE_LABEL_MODE") or "direction").strip()
    return mode + "|" + json.dumps(cand, sort_keys=True)


def make_weighting_axis():
    """Sweep LdP label-uniqueness sample weights per learner: nets only, CatBoost
    only, or both. This is the one axis whose lever is genuinely SHARED - both
    learners consume the identical weight array - so read it on the raw Score basis.
    On the neural_lift basis a change that helps both learners equally cancels out by
    construction and reads as zero.

    Pair it with a multi-bar label. Under the next-bar base label every span is 1,
    the weights are all-ones, and all three candidates are honest no-ops; the QD
    genome is what combines this gene with a triple_barrier label. Running this axis
    in a list next to labeling does NOT combine them either: run_axis evaluates each
    axis independently against the same shared base environment, so axes in a list
    never compose.

    cb_blind because the axis is a select_best over all three candidates and one
    of them ({net_uniqueness}) is invisible to the CatBoost screen by
    construction. Screening the CB-visible candidates while the net-only one dies
    on noise means the axis can never select the net variant, which is the
    comparison the axis exists to make."""
    def _propose(log):
        tried = {json.dumps(e["cand"], sort_keys=True) for e in log
                 if isinstance(e.get("cand"), dict)}
        cands = [c for c in WEIGHTING_CANDIDATES
                 if json.dumps(c, sort_keys=True) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "weighting", _weighting_sig(c))]

    return Axis(
        name="weighting",
        propose=_propose,
        to_env=weighting_env,
        kind="select_best",
        sig=lambda c: ("weighting", _weighting_sig(c)),
        cb_blind=True,
    )


# --- threshold axis (margin + neutral band over the tuned per-asset values) --

THRESHOLD_CANDIDATES = (
    {"thr_margin": 0.02}, {"thr_margin": 0.05},
    {"band_delta": -0.01}, {"band_delta": 0.01}, {"band_delta": 0.02},
    {"thr_margin": 0.02, "band_delta": 0.01},
)

_THRESHOLD_ENV_KEYS = {"thr_margin": "GTRADE_THR_MARGIN",
                       "band_delta": "GTRADE_BAND_DELTA"}


def thresholds_env(cand):
    return {_THRESHOLD_ENV_KEYS[k]: str(v) for k, v in cand.items()}


def make_thresholds_axis():
    """One-candidate-at-a-time sweep of the relative threshold overrides
    (train_hybrid shifts each asset's own tuned thresholds/band). select_best."""
    def _propose(log):
        tried = {json.dumps(e["cand"], sort_keys=True) for e in log
                 if isinstance(e.get("cand"), dict)}
        cands = [c for c in THRESHOLD_CANDIDATES
                 if json.dumps(c, sort_keys=True) not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "thresholds", json.dumps(c, sort_keys=True))]

    return Axis(
        name="thresholds",
        propose=_propose,
        to_env=thresholds_env,
        kind="select_best",
        sig=lambda c: ("thresholds", json.dumps(c, sort_keys=True)),
    )


# --- regime axis (selection-time regime-filter variants) ---------------------


def regime_env(cand):
    return {"GTRADE_REGIME_MODE": cand["regime_mode"]}


def make_regime_axis():
    """Sweep the selection-time regime-filter modes ("both" is the base and is
    not proposed). Measures whether the SMA200/Taleb filter earns its keep."""
    def _propose(log):
        tried = {e["cand"].get("regime_mode") for e in log
                 if isinstance(e.get("cand"), dict)}
        cands = [{"regime_mode": m} for m in REGIME_MODES
                 if m != "both" and m not in tried]
        return [c for c in cands if not ar_memory.tried_seen(
            "regime", json.dumps(c, sort_keys=True))]

    return Axis(
        name="regime",
        propose=_propose,
        to_env=regime_env,
        kind="select_best",
        sig=lambda c: ("regime", json.dumps(c, sort_keys=True)),
    )


def make_pruning_axis(base_features):
    """Backward-elimination over the active candidate features. select-drop additive:
    propose dropping one feature at a time, keep a drop when it lifts the cumulative
    mean Score delta. The active set and the floor are read once at build time."""
    from core.features import active_candidate_features
    active = list(active_candidate_features())
    prune_min = int(os.getenv("GTRADE_AR_PRUNE_MIN", "8"))

    def _propose(log):
        # one candidate per round: the next droppable feature not yet tried in any
        # prior round (accepted or rejected), matching the additive contract that
        # run_axis treats a proposer's whole result as ONE increment.
        tried = {c["drop"] for e in log for c in e.get("cand", [])
                 if isinstance(c, dict) and "drop" in c}
        remaining = [f for f in active if f not in tried
                     and not ar_memory.tried_seen("drop", f)]
        return [{"drop": remaining[0]}] if remaining else []

    def _validate(cand, kept):
        # dropping one more must leave at least prune_min active features
        return len(active) - len(kept or []) - 1 >= prune_min

    return Axis(
        name="pruning",
        propose=_propose,
        to_env=lambda selected: {"GTRADE_DROP_FEATURES": ",".join(c["drop"] for c in selected)},
        kind="additive",
        validate=_validate,
        sig=lambda c: ("drop", c["drop"]),
    )


def _try_sample_frame():
    """Best-effort: build one asset's engineered frame so proposals can be
    univariate-screened before training. Returns None on any problem (screening off)."""
    try:
        import pandas as pd
        from sqlalchemy import create_engine

        # build_features, not a hand-rolled chain: this copy silently went stale
        # once already, and screening on a different feature space than the
        # trainer's is a screen that measures the wrong thing.
        from core.features import build_features
        from core.track_record import _table_name
        engine = create_engine("sqlite:///" + os.path.join(BASE, "market.db"))
        table = _table_name(selection_assets().split(",")[0])
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        return build_features(df, table, engine)[0]
    except Exception:
        return None


def build_axes(names, base_features):
    """Map axis names to Axis objects; unknown names are skipped with a warning."""
    builders = {
        "features": lambda: make_features_axis(base_features),
        "labeling": make_labeling_axis,
        "pruning": lambda: make_pruning_axis(base_features),
        "hyper": make_hyper_axis,
        "nets": make_nets_axis,
        "weighting": make_weighting_axis,
        "thresholds": make_thresholds_axis,
        "regime": make_regime_axis,
    }
    axes = []
    for n in names:
        n = n.strip()
        if n in builders:
            axes.append(builders[n]())
        elif n:
            logger.warning("unknown auto-research axis skipped: %s", n)
    return axes


def _persist(axis_name, log):
    """Persist a per-axis log into the shared state, keyed by axis."""
    state = load_state()
    by_axis = state.get("by_axis", {})
    by_axis[axis_name] = log
    state["by_axis"] = by_axis
    state["log"] = [e for entries in by_axis.values() for e in entries]
    save_state(state)


def _gate_verdict(ok, replicated, clears, neural_lift=None, significant=None):
    """Console verdict string shared by the axis and QD gates. A candidate that
    cleared the Score bar and was stopped only by the neural floor says so - the
    old bare "not adoptable" next to a strong dScore reads like a stats failure."""
    if not ok:
        if (significant and neural_lift is not None
                and neural_lift <= neural_floor()):
            return "not adoptable (neural floor: nets pay for it)"
        return "not adoptable"
    if replicated:
        return "REPLICATED-ADOPTABLE (%d clears)" % clears
    return "ADOPTABLE (BH), 1st clear - awaiting replication"


def _winner_sig(axis_name, winner):
    """Stable cross-run signature of an axis winner, independent of temp-file env
    paths. A features winner is a list of spec dicts (name-agnostic); a pruning
    winner is a list of {'drop': f}; a labeling winner is a dict."""
    def _item(it):
        if isinstance(it, dict) and "drop" in it:
            return "drop:" + it["drop"]
        if isinstance(it, dict) and "op" in it:
            return "spec:" + json.dumps(_spec_signature(it))
        return json.dumps(it, sort_keys=True)
    if isinstance(winner, list):
        body = ",".join(sorted(_item(it) for it in winner))
    else:
        body = json.dumps(winner, sort_keys=True)
    return axis_name + ":" + body


# The two axis-candidate keys that are spelled differently on the Genome. Every
# other key an axis proposes is already a gene name, which is why this is short.
_AXIS_GENE_ALIASES = {"seeds": "net_seeds", "calibrate": "net_calibrate",
                      "uniqueness": "net_uniqueness"}


def _axis_genome(axis_name, winner):
    """Literal translation of one axis winner into a Genome. May raise."""
    if axis_name == "features":
        return Genome(extra=[s for s in winner if isinstance(s, dict)])
    if axis_name == "pruning":
        return Genome(drops=[c["drop"] for c in winner
                             if isinstance(c, dict) and "drop" in c])
    if axis_name == "labeling":
        return Genome(label_mode=winner["mode"],
                      label_window=int(winner["window"]))
    return Genome(**{_AXIS_GENE_ALIASES.get(k, k): v
                     for k, v in winner.items()})


def _env_without_specs(env):
    """GTRADE_DSL_SPECS names a per-call temp file, so it can never compare
    equal between two calls and says nothing about what was trained. The spec
    NAMES still ride along in GTRADE_EXTRA_FEATURES, which does compare."""
    return {k: v for k, v in env.items() if k != "GTRADE_DSL_SPECS"}


def moved_genes(genome):
    """The gene names this genome sets away from the production default.

    Used to answer "what did the finding actually change", which is not the same
    question as "what does the finding's genome say": a bare genome states a
    value for all fifteen genes, and fourteen of them are just the defaults.

    Sound because no axis proposes a candidate equal to its own base - the
    labeling axis never offers `direction`, the regime axis never offers `both` -
    so a gene sitting on its default was not moved by the axis.
    """
    ref = asdict(_canon_genome(Genome()))
    got = asdict(_canon_genome(Genome(**genome))) if isinstance(genome, dict) \
        else asdict(_canon_genome(genome))
    return sorted(k for k in ref if got.get(k) != ref[k])


def compose_genomes(base, overlay, genes):
    """`base`, with only `genes` taken from `overlay`, canonicalised."""
    names = {f.name for f in fields(Genome)}
    merged = {k: v for k, v in dict(base).items() if k in names}
    merged.update({g: overlay[g] for g in genes if g in overlay and g in names})
    return _canon_genome(Genome(**merged))


def compose_with_reference(overlay, ref_genome):
    """The finding applied ON TOP of what is running, or None when meaningless.

    The A/B measures a candidate against the adopted reference, so the candidate
    that answers the adoption question is "the reference, with this change",
    not "this change alone on top of nothing". The difference is not academic:
    the 2026-08-18 adoption replaced a genome carrying four feature drops, seven
    DSL features and a threshold margin with a bare two-gene genome, because the
    bare form was the only one ever offered, and the A/B duly reported that the
    bare form beat the full one on the campaign basis.

    None when there is no reference, or when composing changes nothing (the
    reference already has these genes), or when the result IS the reference.
    """
    if not ref_genome:
        return None
    genes = moved_genes(overlay)
    if not genes:
        return None
    composed = compose_genomes(ref_genome, overlay, genes)
    if genome_sig(composed) in (genome_sig(_canon_genome(Genome(**ref_genome))),
                                genome_sig(_canon_genome(Genome(**overlay)))):
        return None
    return composed


def genome_from_axis(axis, winner):
    """The axis winner as a Genome, or None when it has no genome form.

    An axis finding is a verdict about an ENV; the A/B and the adoption speak
    only Genome. Without this join an adoptable axis winner is unadoptable by
    construction: it clears the held-out gate, is written to the journal, and
    nothing downstream can reach it. That is not hypothetical - a labeling
    winner cleared the gate seven times between 2026-08-17 and 08-18 while
    auto_loop.next_action, finding no genome to test, kept searching.

    The mapping is CHECKED rather than trusted: the genome is kept only when it
    composes back to the same training env the gate actually measured. A wrong
    or stale alias here would send the A/B off to train something else and then
    file the result under this winner's evidence, which is worse than not
    offering the candidate at all.
    """
    if winner is None:
        return None
    try:
        g = _canon_genome(_axis_genome(axis.name, winner))
        same = (_env_without_specs(genome_to_env(g))
                == _env_without_specs(axis.to_env(winner)))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return g if same else None


def main():
    import argparse

    # Before anything trains, and before the regate branch too: both paths train.
    apply_manual_load_profile()
    if refuse_contradictory_campaign():
        return 1
    p = argparse.ArgumentParser(description="auto-research")
    p.add_argument("--regate", action="store_true",
                   help="re-gate stored candidate genomes under the current gate")
    p.add_argument("--regate-k", type=int, default=8)
    p.add_argument("--regate-screen", action="store_true",
                   help="optional cheap CB-only prefilter before the full re-gate eval")
    # parse_known_args (not parse_args): main() is called directly by the existing
    # test suite under pytest, whose own argv (test paths, -k, -o, ...) must not
    # make an un-flagged `ar.main()` call SystemExit(2). Real CLI invocations are
    # unaffected since there are no extra args to discard.
    args, _ = p.parse_known_args()
    # GTRADE_AR_MODE=regate is the same run as --regate, reachable by an unattended
    # caller that only passes an environment. It spends the cycle re-testing what
    # the search already flagged instead of finding something new, which is the
    # only way a finding gets its SECOND independent clear on purpose: the
    # replication gate needs two, and until now the search had to stumble over
    # the same genome twice by chance. 29 flagged against 16 replicated is the
    # size of that backlog.
    by_env = (os.getenv("GTRADE_AR_MODE") or "").strip().lower() == "regate"
    if args.regate or by_env:
        k, screen = args.regate_k, args.regate_screen
        if by_env and not args.regate:
            # The env caller has no flags, so its two settings come from the env
            # too. The CLI path keeps its own defaults untouched.
            try:
                k = int(os.getenv("GTRADE_AR_REGATE_K") or k)
            except ValueError:
                pass
            screen = _screen_on()
        regate(k=k, screen=screen)
        return
    base_features = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_z", "rsi",
                     "macd_hist", "bb_pos", "trend_strength", "atr"]
    names = os.getenv("GTRADE_AR_AXES", "features,labeling").split(",")
    if "qd" in [n.strip() for n in names]:
        run_qd()
        return
    axes = build_axes(names, base_features)
    # Only additive axes (e.g. features) consult screen_df for prescreening; skip the
    # DB read + feature pipeline entirely when no such axis is selected.
    screen_df = _try_sample_frame() if any(a.kind == "additive" for a in axes) else None
    print("[auto-research] axes: %s | budget: %d | prescreen: %s" % (
        ",".join(a.name for a in axes), BUDGET, "on" if screen_df is not None else "off"))

    base_rows = score_rows(selection_assets(), {}, train_base_cached)  # shared base
    screen_base = train_base_cached(selection_assets(), {"GTRADE_SCREEN_ONLY": "1"}) if _screen_on() else None
    tier_base = _tier_base(train_base_cached) if tier_on() else None
    tier_neural_base = (_tier_neural_base(train_base_cached)
                        if tier_on() else None)
    obj = _objective()
    basis = _score_basis()
    winners = []   # (axis_name, winner, p, value, tag, neural_lift)
    ho_base_full = ho_base_contrib = None
    for axis in axes:
        try:
            prior = load_state().get("by_axis", {}).get(axis.name)
            res = run_axis(axis, BUDGET, base_rows, train_env, screen_df=screen_df,
                           prior_log=prior, persist=lambda log, a=axis.name: _persist(a, log),
                           screen_base=screen_base, screen_min=SCREEN_MIN,
                           tier_base=tier_base, tier_neural_base=tier_neural_base)
            winner = res.get("kept") or res.get("best")
            if not winner:
                print(f"[auto-research] axis {axis.name}: nothing beat the base.")
                continue
            winner_env = axis.to_env(winner)
            if ho_base_full is None:
                ho_base_full, ho_base_contrib = _heldout_eval(
                    heldout_assets(), {}, train_base_cached)
            var_full, var_contrib = _heldout_eval(heldout_assets(), winner_env, train_env)
            base_contrib = {r["Asset"]: r["Score"] for r in ho_base_contrib}
            nl, _d = _objective_delta(var_contrib, base_contrib, "mean")
            nl = round(nl, 4) if _d else None
            if basis == "neural":
                p, value, _d, tag = holdout_stats(ho_base_contrib, var_contrib, obj)
            else:
                _st = _gate_stats(ho_base_full, var_full, obj)
                if _st is None:
                    print("[gate] skipped: the candidate's holdout rows carry no "
                          "column for basis %s" % basis)
                    continue
                p, value, _d, tag = _st
            winners.append((axis, winner, p, value, tag, nl))
        except RuntimeError as exc:
            print(f"[auto-research] axis {axis.name}: LLM proposer unavailable, skipping ({exc})")
            continue

    flags = benjamini_hochberg([w[2] for w in winners])
    ts = datetime.utcnow().isoformat()
    finding_winners = []
    for (axis, winner, p, value, tag, nl), s in zip(winners, flags):
        name = axis.name
        ok = adopt_ok(s, value, obj, nl)
        replicated = clears = None
        if ok:
            wsig = _winner_sig(name, winner)
            replicated = ar_memory.replication_seen(wsig)
            clears = ar_memory.replication_add(wsig, ts)
            if ar_wiki.wiki_on() and clears >= 2:
                ar_wiki.note_replicated(wsig, "replicated (%d clears)" % clears)
        # Recorded for every winner, not only the adoptable ones: an axis result
        # that misses the bar today can still be re-gated later, and a finding
        # without its genome is a dead end no matter what its verdict was.
        g = genome_from_axis(axis, winner)
        finding_winners.append({"axis": name, "p": p, "value": value, "tag": tag,
                                "adoptable": ok, "neural_lift": nl,
                                "genome": asdict(g) if g else None,
                                "replicated": bool(replicated), "clears": clears or 0})
        verdict = _gate_verdict(ok, bool(replicated), clears, nl, s)
        nl_str = "" if nl is None else f" | neural_lift {nl:+.2f}"
        print(f"[auto-research] axis {name}: {verdict} | {tag}{nl_str}")
        if ok and g is None:
            print("[auto-research] axis %s: adoptable, but this winner has no "
                  "genome form, so no A/B can be built from it. It stays a note "
                  "in the journal." % name)
    ar_memory.findings_append({
        "ts": ts, "mode": "axes", "basis": basis,
        "axes": [a.name for a in axes], "budget": BUDGET,
        "winners": finding_winners})
    if ar_wiki.wiki_on():
        ar_wiki.compile_wiki()
    mem = ar_memory.findings_summary()
    print("[auto-research] memory: %d experiments tried, %d adoptable, %d replicated so far."
          % (mem["experiments"], mem["adoptable"], mem["replicated"]))
    print("[auto-research] nothing adopted automatically; review _auto_research_log.json.")


if __name__ == "__main__":
    main()
