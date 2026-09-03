"""Assemble and run an A/B of an archive elite against what is actually live.

The reference arm is the ADOPTED genome, not production defaults. Once something
is adopted, "better than defaults" is a number about a configuration that no
longer exists anywhere: a candidate can beat the naked base and still be worse
than what is running, and adopting it would be a regression the report concealed.

Two steps on purpose. A run is tens of hours, so what is about to be measured
should be visible before it starts.

Usage:
  python ab_build.py                 # list elites, suggest a holdout, write the config
  python ab_build.py --show          # what the pending config says
  python ab_build.py --run           # execute it
  python ab_build.py --assets A,B,C  # an explicit holdout instead of the suggestion
  python ab_build.py --n 20          # a different holdout size
"""
import argparse
import datetime
import glob
import json
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

CONFIG_PATH = os.path.join(BASE, "_ab_config.json")
DB_PATH = os.path.join(BASE, "market.db")
MAX_CANDIDATES = 3
# Far apart so two rolls never share a derived per-model seed. ab_noise.py has
# used 1000/2000/3000 by hand since the determinism work; this keeps that shape.
SEED_STRIDE = 1000


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _adopted_record():
    """Indirected so tests can substitute a record without touching a file."""
    from core import adopted
    return adopted.load()


def reference():
    """What every candidate is measured against: {label, sig, env}.

    The env comes from auto_research.genome_to_env, NOT core.adopted.env_overrides.
    env_overrides omits GTRADE_DSL_SPECS because production reads the adopted
    specs from the adopted file, but training children are forced unadopted, so a
    child given only env_overrides would receive the feature NAMES and compute
    none of the columns. genome_to_env materialises the specs to a temp file.
    """
    rec = _adopted_record()
    if not rec:
        return {"label": "base", "sig": None, "env": {}}
    import auto_research as ar
    g = ar.Genome(**rec["genome"])
    return {"label": "adopted:{}".format(rec.get("label", "?")),
            "sig": ar.genome_sig(g), "env": ar.genome_to_env(g)}


def is_reference(cand, ref):
    """True when this candidate IS the thing it would be measured against."""
    return bool(ref.get("sig")) and cand.get("sig") == ref["sig"]


def previous_holdouts(base=None, for_sigs=None):
    """Holdouts a NEW gate must avoid.

    The rule used to be "every asset of every past A/B, forever", and the cost
    was measured on 2026-09-03: 254 of 847 assets consumed and the whole forex
    class gone, which is what made a class hypothesis untestable.

    The failure it was written for was narrower than that. Re-gating the SAME
    genome on the same assets compared a result with itself and produced dScore
    0.00, p=1.000 - both arms read the same cached rows. A DIFFERENT genome
    trains a new candidate arm on those assets, so nothing is compared with
    itself; what remains is ordinary multiple testing, which alpha and the
    replication step already handle.

    So a run is excluded only when it measured one of `for_sigs`. Passing None
    keeps the old blanket behaviour, which is what a caller with no signature in
    hand should get.
    """
    out = []
    want = {s for s in (for_sigs or []) if s}
    for path in sorted(glob.glob(os.path.join(base or BASE,
                                              "_ab_genomes_*.json"))):
        data = _read_json(path)
        if not isinstance(data, dict) or not data.get("holdout"):
            continue
        if want:
            sigs = {(r or {}).get("sig") for r in (data.get("results") or {}).values()}
            sigs.add(data.get("reference_sig"))
            if not (sigs & want):
                continue
        out.append(data["holdout"])
    rec = _adopted_record()
    held = ((rec or {}).get("evidence") or {}).get("holdout")
    if held:
        out.append(held)
    return out


def bar_counts(db_path=None):
    """Rows per asset in market.db. A missing table counts as zero."""
    from config import FULL_ASSET_MAP
    path = db_path or DB_PATH
    out = {}
    if not os.path.exists(path):
        raise SystemExit(f"market.db not found at {path}; run data_engine first.")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        for asset in FULL_ASSET_MAP:
            try:
                out[asset] = cur.execute(
                    f'SELECT COUNT(*) FROM "{asset.lower()}"').fetchone()[0]
            except sqlite3.Error:
                out[asset] = 0
    finally:
        con.close()
    return out


def build_config(candidates, assets, ref, floor, alpha, seed, objective):
    """The run description, with the reference written down.

    floor and alpha are frozen here rather than read at run time, so a later
    change to either constant cannot silently reinterpret a pending run. The
    BASIS is frozen for the same reason and belongs with them: it decides the
    units the floor is even in, so a config built on net_auc and run after the
    environment fell back to raw would compare a Score against an AUC floor.

    `basis` is the DECISION basis, the one this verdict is read in. The search
    basis is recorded beside it rather than instead of it: the two are the same
    until a campaign names both, and when they differ, which one produced the
    candidate is part of the evidence.
    """
    import auto_research as ar
    return {
        "holdout": ",".join(assets),
        "objective": objective,
        "basis": ar.decision_basis(),
        "search_basis": ar._score_basis(),
        "floor": floor,
        "alpha": alpha,
        "seed": seed,
        "reference": ref["label"],
        "reference_sig": ref["sig"],
        "candidates": [{"label": c["label"], "sig": c.get("sig"),
                        "genome": c["genome"]} for c in candidates],
    }


def last_spread(base=None):
    """The per-asset spread the most recent A/B actually observed, or None.

    Recorded from 2026-08-21 onward as `sd_raw`. Older runs did not keep it, so
    a repository with only those answers None rather than a guess.
    """
    import glob
    want = len(seed_roll())
    for path in sorted(glob.glob(os.path.join(base or BASE,
                                              "_ab_genomes_*.json")),
                       reverse=True):
        data = _read_json(path) or {}
        # Only runs averaged the same number of times. Projecting an r=1 run
        # from an r=4 spread promises power the run will not have.
        if int(data.get("ab_seeds") or 1) != want:
            continue
        for res in (data.get("results") or {}).values():
            if isinstance(res, dict) and res.get("sd_raw"):
                return float(res["sd_raw"])
    return None


def _allow_underpowered():
    return (os.getenv("GTRADE_AB_ALLOW_UNDERPOWERED") or "").strip() in ("1", "true", "True")


def projected_power(n, floor, base=None):
    """What a run of `n` assets could resolve, before it is paid for.

    The point of doing this at configure time: a holdout that cannot resolve
    its own floor takes the same hours to run as one that can, and answers
    nothing. On 2026-08-21 the spread was 3.74 in Score units, which means even
    the whole 207-asset universe resolves only about +0.65, so the honest move
    is to raise the floor or lower the noise rather than to spend the hours.
    """
    sd = last_spread(base)
    if not sd or not n or floor <= 0:
        return ""
    mde = Z_SUM * sd / (n ** 0.5)
    needed = int((Z_SUM * sd / floor) ** 2 + 0.999)
    if mde <= floor:
        return ("  power: at the last measured spread %.3g, %d assets resolve "
                "%+.3g, which clears the floor %+.3g." % (sd, n, mde, floor))
    return ("  power: at the last measured spread %.3g, %d assets resolve only "
            "%+.3g against a floor of %+.3g. %d assets would be needed. Raise "
            "--n, raise the floor, or reduce the noise; running it as "
            "configured cannot answer the question."
            % (sd, n, mde, floor, needed))


def write_config(cfg, path=None):
    with open(path or CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)


def read_config(path=None):
    return _read_json(path or CONFIG_PATH)


def _candidate_pool():
    """Every archive elite, measured ones included.

    A previous measurement was against whatever was live THEN. Hiding an elite
    because it once missed would make it permanently invisible after the adoption
    changes or is reverted, which is the same reasoning error this tool exists to
    remove: a number is only meaningful beside the reference it was taken against.
    """
    import adopt_genome
    return adopt_genome.candidates()


def _gate_by_sig():
    """Latest held-out gate verdict per genome signature, from the findings journal.

    The picker lists archive elites by their SEARCH fitness, measured on the ten
    selection assets. The number a decision is actually made on is the gate delta
    on the held-out set, and it lives in a different file. Without this join the
    same genome appears as two unrelated numbers and the reader has to match
    genomes by hand to find the candidate a run just flagged.
    """
    import auto_research as ar
    from core import ar_memory

    ref_genome = (_adopted_record() or {}).get("genome")
    out = {}
    for rec in ar_memory.findings_all() or []:
        ts = rec.get("ts") or ""
        for w in rec.get("winners", []):
            gd = w.get("genome")
            if not isinstance(gd, dict):
                continue
            try:
                sig = ar.genome_sig(ar.Genome(**gd))
            except (TypeError, ValueError):
                continue
            verdict_ = {"ts": ts, "tag": w.get("tag"),
                        "adoptable": bool(w.get("adoptable")),
                        "clears": w.get("clears") or 0}
            sigs = [sig]
            # The same verdict also stands for this change applied ON TOP of the
            # reference, which is the candidate the pool offers beside the bare
            # one. The gate measured the CHANGE, and the composed genome is that
            # change; without this the composed arm can never be auto-picked,
            # since auto_picks only takes what a gate has flagged.
            try:
                composed = ar.compose_with_reference(gd, ref_genome)
            except (TypeError, ValueError):
                composed = None
            if composed is not None:
                sigs.append(ar.genome_sig(composed))
            for s in sigs:
                prev = out.get(s)
                if prev is None or ts >= prev["ts"]:
                    out[s] = verdict_
    return out


def _gate_note(cand, gates):
    """The gate line for one candidate, or empty when it never reached a gate."""
    import auto_research as ar

    try:
        sig = ar.genome_sig(ar.Genome(**cand["genome"]))
    except (TypeError, ValueError):
        return ""
    hit = gates.get(sig)
    if not hit:
        return ""
    flag = "ADOPTABLE" if hit["adoptable"] else "not adoptable"
    rep = ", replicated" if hit["clears"] >= 2 else ""
    return "\n      gate {}: {} [{}{}]".format(
        hit["ts"][:16], hit["tag"], flag, rep)


def tested_against(ref_sig, base=None):
    """Genome signatures already measured against THIS reference.

    Provenance is not a verdict: adopt_genome marks a candidate "measured" the
    moment it appears in any A/B file, but a result taken against an earlier
    reference is a number about a configuration that is no longer live, so it
    stays retestable. Only a run whose reference_sig matches the live one has
    settled the question. An unattended caller needs that distinction to
    terminate: without it it would rebuild the same config every cycle.
    """
    out = set()
    for path in sorted(glob.glob(os.path.join(base or BASE,
                                              "_ab_genomes_*.json"))):
        data = _read_json(path)
        if not isinstance(data, dict) or data.get("reference_sig") != ref_sig:
            continue
        for res in (data.get("results") or {}).values():
            if isinstance(res, dict) and res.get("sig"):
                out.add(res["sig"])
    return out


def auto_picks(pool, ref, gates, tested, limit=MAX_CANDIDATES):
    """The candidates an unattended run should test, strongest evidence first.

    Only genomes the held-out gate already flagged adoptable. The pool also holds
    every other archive elite, ranked by SEARCH fitness, and search fitness
    shrinks: genome A scored 5.30 in the search and 1.63 on a fresh holdout. A
    picker running without a human would otherwise spend a day of training per
    arm on a number that was never measured on data it had not seen.
    """
    ranked, seen = [], set()
    for cand in pool:
        sig = cand.get("sig")
        # candidates() yields one entry per A/B file, so a genome measured in
        # several past runs appears several times; the first is enough.
        if not sig or sig in seen or sig in tested or is_reference(cand, ref):
            continue
        hit = gates.get(sig)
        if not hit or not hit["adoptable"]:
            continue
        seen.add(sig)
        ranked.append((hit["clears"], hit["ts"], cand))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [cand for _, _, cand in ranked[:limit]]


def search_gate(n):
    """The search's own gate list, grown to n and printed for the launcher.

    Two different holdouts exist and only one of them was ever sizeable from
    outside. auto_research gates candidates on GTRADE_AR_HELDOUT before any of
    them reaches an A/B, so that list decides what is even offered - and it was
    a hardcoded fourteen. This grows it, keeping every asset already in it so
    earlier measurements stay comparable, and excluding the search and tier
    sets so the gate never scores an asset the search selected on.
    """
    import auto_research as ar
    from config import ASSET_TYPES, FULL_ASSET_MAP
    from core import holdout
    from core.backtesting import price_resolution_ok

    current = [a.strip() for a in ar.heldout_assets().split(",") if a.strip()]
    ex = holdout.excluded([ar.SELECTION_ASSETS, ar.tier_assets()], [])
    counts = bar_counts()
    elig = holdout.eligible(list(FULL_ASSET_MAP), counts, ex)
    # A gate is only as honest as its series: an asset whose price is quoted too
    # coarsely to carry a one-bar sign contributes a label that is mostly ties.
    elig = [a for a in elig if price_resolution_ok(_closes(a))[0]]
    return holdout.grow(current, elig, ASSET_TYPES, n)


def _closes(asset):
    import sqlite3

    from core.track_record import _table_name
    try:
        with sqlite3.connect(os.path.join(BASE, "market.db")) as con:
            return [r[0] for r in con.execute(
                'SELECT Close FROM "%s" ORDER BY Date' % _table_name(asset))]
    except Exception:
        return []


def _suggest_assets(n, seed, for_sigs=None):
    """A holdout for a gate, preferring assets whose noise is already known small.

    `for_sigs` are the genome signatures about to be measured: only runs that
    already measured one of them make their assets off limits (see
    previous_holdouts). Everything else is drawable again, which is what puts
    the 254 consumed assets - the whole forex class among them - back in play.
    """
    import auto_research as ar
    from config import ASSET_TYPES, FULL_ASSET_MAP
    from core import holdout
    # heldout_assets(), not the HELDOUT_ASSETS alias: the alias is frozen to
    # PROD_HELDOUT at import, so a run with GTRADE_AR_HELDOUT set would exclude
    # the wrong list and hand the A/B assets the search had already gated on.
    # auto_research says as much beside the alias - "live code calls
    # heldout_assets()" - and this was the one live caller that did not.
    ex = holdout.excluded([ar.SELECTION_ASSETS, ar.heldout_assets(),
                           ar.tier_assets()], previous_holdouts(for_sigs=for_sigs))
    elig = holdout.eligible(list(FULL_ASSET_MAP), bar_counts(), ex)
    quiet = quiet_first(elig)
    return holdout.suggest(quiet, ASSET_TYPES, n=n, seed=seed), elig


NOISE_PATH = os.path.join(BASE, "_asset_noise.json")


def record_asset_noise(assets, deltas, path=None):
    """Bank each asset's own seed noise, so later holdouts can prefer the quiet.

    The required holdout grows with the SQUARE of the spread, and the spread is
    wildly uneven: measured 2026-09-02, se runs from 0.53 (EVRG) to 8.87 (SHY).
    Selecting on it cuts the noise part of 2.75 and leaves the 1.83 that is a
    real difference between assets, which is why this helps and cannot solve.

    A variance is far more stable than a mean, and this one is not the quantity
    any verdict is read on, so reusing it to choose assets is not the selection
    trap that reusing an EFFECT would be.
    """
    path = path or NOISE_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            book = json.load(fh)
    except (OSError, ValueError):
        book = {}
    n = deltas.shape[1] if getattr(deltas, "ndim", 0) == 2 else 0
    if n < 2:
        return book
    for i, asset in enumerate(assets):
        row = deltas[i]
        se = float(row.std(ddof=1) / (n ** 0.5))
        prev = book.get(asset) or {}
        book[asset] = {"se": se, "n_seeds": n,
                       "measured": datetime.date.today().isoformat(),
                       "times": int(prev.get("times", 0)) + 1}
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(book, fh, ensure_ascii=False, indent=2, sort_keys=True)
    except OSError:
        pass
    return book


def quiet_first(assets, path=None):
    """`assets` with the measured-quiet ones first and the unmeasured after.

    Unmeasured assets are NOT pushed to the back as if they were noisy: nothing
    is known about them, and a holdout of only previously-measured assets would
    keep re-drawing the same handful. They keep their order behind the quiet
    ones, so the draw still spreads across classes.
    """
    try:
        with open(path or NOISE_PATH, encoding="utf-8") as fh:
            book = json.load(fh)
    except (OSError, ValueError):
        return list(assets)
    known = [(a, float((book.get(a) or {}).get("se", 0))) for a in assets
             if isinstance(book.get(a), dict) and book[a].get("se") is not None]
    quiet = [a for a, _ in sorted(known, key=lambda x: x[1])]
    rest = [a for a in assets if a not in set(quiet)]
    return quiet + rest


def _holdout_default_n():
    from core import holdout
    return holdout.DEFAULT_N


def _configure(args):
    import adopt_genome
    import auto_research as ar
    from core import holdout

    ref = reference()
    pool = _candidate_pool()
    if not pool:
        print("No unvalidated elites in the archive to test.")
        return
    print("Measuring against: {}\n".format(ref["label"]))
    gates = _gate_by_sig()
    if args.auto:
        chosen = auto_picks(pool, ref, gates, tested_against(ref["sig"]))
        if not chosen:
            print("Nothing to test: no gate-adoptable elite is left that has not "
                  "already been measured against {}.".format(ref["label"]))
            return
        for c in chosen:
            print("  auto-picked %s%s" % (adopt_genome.describe(c),
                                          _gate_note(c, gates)))
        suggested, elig = _suggest_assets(args.n, args.seed)
        assets = ([a.strip() for a in args.assets.split(",") if a.strip()]
                  or suggested)
        problems = holdout.validate(assets, elig)
        if problems:
            print("\nThis holdout cannot be used:")
            for p in problems:
                print(f"  - {p}")
            return
        cfg = build_config(chosen, assets, ref,
                           ar._adopt_floor(args.objective, basis=ar.decision_basis()),
                           args.alpha, args.seed, args.objective)
        write_config(cfg)
        print(f"\nWrote {os.path.basename(CONFIG_PATH)}")
        _print_config(cfg)
        return
    for i, c in enumerate(pool, 1):
        notes = []
        if is_reference(c, ref):
            notes.append("this IS the reference")
        if c["kind"] == "measured":
            notes.append("already measured against a previous reference")
        suffix = "  <- " + "; ".join(notes) if notes else ""
        print("  %d. %s%s%s" % (i, adopt_genome.describe(c), suffix,
                                _gate_note(c, gates)))
    raw = input("\nWhich numbers (comma separated, blank to cancel)? ").strip()
    picks = []
    for p in raw.split(","):
        p = p.strip()
        # Keyed on the resolved index, not the typed token: "1,01" is one pick.
        if p.isdigit() and 1 <= int(p) <= len(pool) and int(p) not in picks:
            picks.append(int(p))
    chosen = [pool[i - 1] for i in picks]
    if not chosen:
        print("Cancelled.")
        return
    if len(chosen) > MAX_CANDIDATES:
        # The old text here said 8 to 11 hours an arm. That predated the GPU
        # environment and the 2026-08-04 miner cleanup; re-measured 2026-08-13
        # from ar_progress.json a 14-asset holdout arm is 1997s. Quoting hours
        # where the truth is minutes is how a run nobody could afford gets
        # refused and one nobody checked gets waved through.
        print("At most %d per run: each arm is a full training of the holdout, "
              "about %d min at %d assets (measured 2026-08-13: 1997s for 14), "
              "plus the reference. A later run draws a fresh holdout, so its "
              "reference arm is trained again from scratch - picking fewer now "
              "does not save that."
              % (MAX_CANDIDATES, round(1997 * args.n / 14 / 60), args.n))
        return

    suggested, elig = _suggest_assets(args.n, args.seed)
    if args.assets:
        assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    else:
        print("\nSuggested holdout (%d): %s" % (len(suggested),
                                                ",".join(suggested)))
        typed = input("Enter to accept, or paste your own list: ").strip()
        assets = [a.strip() for a in typed.split(",") if a.strip()] or suggested

    problems = holdout.validate(assets, elig)
    if problems:
        print("\nThis holdout cannot be used:")
        for p in problems:
            print(f"  - {p}")
        return

    cfg = build_config(chosen, assets, ref,
                           ar._adopt_floor(args.objective, basis=ar.decision_basis()),
                       args.alpha, args.seed, args.objective)
    write_config(cfg)
    print(f"\nWrote {os.path.basename(CONFIG_PATH)}")
    _print_config(cfg)
    print("\nNext: python ab_build.py --run   (do not start it while a retrain "
          "is running)")


def _print_config(cfg):
    print("  reference : {}".format(cfg["reference"]))
    print("  holdout   : {}".format(cfg["holdout"]))
    print("  objective : {}   floor {:+.4g}   alpha {:.3f}   basis {}".format(
        cfg["objective"], cfg["floor"], cfg["alpha"], cfg.get("basis", "raw")))
    for c in cfg["candidates"]:
        g = c["genome"]
        print("  candidate : %-8s %d drops, %d extra, label %s/%s"
              % (c["label"], len(g.get("drops") or []),
                 len(g.get("extra") or []), g.get("label_mode", "direction"),
                 g.get("label_window", 30)))
    # Said BEFORE the hours are spent. A holdout that cannot resolve its own
    # floor costs exactly as much to run as one that can, and answers nothing.
    pw = projected_power(len((cfg["holdout"] or "").split(",")), cfg["floor"])
    if pw:
        print(pw)


def seed_roll():
    """The GTRADE_SEED values one A/B arm is trained under.

    Reseeding is the only lever measured to cut this gate's noise. A per-asset
    dScore has sd 3.74 against an adoption floor of 0.5, and that noise is
    per-TRAINING: a retrain moves every fold of an asset together, which is why
    fold-averaging bought 10 percent and reseeding falls as sqrt(r). At r=4 the
    holdout needed to resolve +0.5 drops from 347 assets to about 87, which the
    208-asset universe can actually supply.

    Four by default, because one cannot decide anything here. Measured
    2026-08-24 with ab_noise on the tier unit: reseeding ALONE, same config and
    same data, moves the objective by 1.917 against an adoption floor of 0.5.
    Over the 40-asset gate at the observed per-asset spread of 3.03, the mean
    carries a noise of 0.48 at r=1, which is the floor itself, and 0.24 at r=4.
    Set GTRADE_AB_SEEDS=1 to opt out and pay one training instead of four.
    """
    from core.net_hygiene import seed_base
    try:
        r = int(os.getenv("GTRADE_AB_SEEDS") or 4)
    except ValueError:
        r = 4
    return [seed_base() + i * SEED_STRIDE for i in range(max(1, r))]


def _mean_rows(rolls):
    """Average per-asset rows across rolls, numeric columns only.

    Only assets EVERY roll scored. A training that dropped one would otherwise
    leave that asset averaged over fewer seeds than its neighbours, so the arm
    would carry a different amount of noise per row with nothing on the row to
    say so. Non-numeric columns (Fold_Scores) are taken from the first roll
    unchanged, so nothing downstream loses a column it reads.
    """
    per = [{r["Asset"]: r for r in roll} for roll in rolls]
    common = set(per[0]).intersection(*[set(p) for p in per[1:]])
    out = []
    for row in rolls[0]:
        asset = row["Asset"]
        if asset not in common:
            continue
        merged = dict(row)
        for key in row:
            vals = [p[asset].get(key) for p in per]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in vals):
                merged[key] = sum(vals) / len(vals)
        out.append(merged)
    return out


def _heldout_eval(subset, env, fn, **kw):
    """Indirected so the tests can see which trainer the reference chose.

    Also where r-seed averaging happens, and it happens for BOTH arms because
    both come through this one function. Averaging the candidate but not the
    reference would put a quiet number against a noisy one and read the
    difference as an effect.
    """
    import auto_research as ar
    seeds = seed_roll()
    if len(seeds) == 1:
        return ar._heldout_eval(subset, env, fn, **kw)
    was = os.environ.get("GTRADE_SEED")
    fulls, contribs = [], []
    try:
        for seed in seeds:
            os.environ["GTRADE_SEED"] = str(seed)
            full, contrib = ar._heldout_eval(subset, env, fn, **kw)
            if full:
                fulls.append(full)
                contribs.append(contrib)
    finally:
        if was is None:
            os.environ.pop("GTRADE_SEED", None)
        else:
            os.environ["GTRADE_SEED"] = was
    if not fulls:
        return [], []
    _bank_noise(fulls)
    return _mean_rows(fulls), _mean_rows(contribs)


def _bank_noise(fulls):
    """Record each asset's seed spread from the rolls this arm just trained.

    One ARM's own standard error, not the delta's - the delta of two arms is
    about sqrt(2) times larger. That constant does not matter for what the book
    is used for, which is ranking assets from quiet to noisy, and keeping the
    raw quantity means the number stays meaningful if the use ever changes.
    """
    import numpy as _np

    try:
        per = [{r["Asset"]: float(r["Score"]) for r in roll
                if isinstance(r, dict) and isinstance(r.get("Score"), (int, float))}
               for roll in fulls]
        common = sorted(set(per[0]).intersection(*[set(p) for p in per[1:]]))
        if not common or len(per) < 2:
            return
        record_asset_noise(common,
                           _np.array([[p[a] for p in per] for a in common]))
    except Exception:
        pass  # a bookkeeping aid must never take a gate down


def train_reference(subset, ref):
    """Train the arm every candidate is compared against.

    Cached BY SIGNATURE when a genome is adopted: base_key hashes the env dict,
    and the reference env holds a per-run temp spec path, so it would miss every
    run. Caching it under the plain base key would be worse - it would serve a
    naked-base result and this run would report it as a comparison against the
    adopted genome.
    """
    import auto_research as ar

    if ref["sig"]:
        def _candidate_train_cached_by_sig(sub, env, _sig=ref["sig"]):
            return ar._candidate_train_cached(sub, env, _sig)
        return _heldout_eval(subset, ref["env"], _candidate_train_cached_by_sig)
    return _heldout_eval(subset, {}, ar.train_base_cached)


def evaluate(cand, subset, ref_full, ref_contrib, objective):
    """One candidate against the reference. Returns the statistics dict."""
    import auto_research as ar

    g = ar.Genome(**cand["genome"])
    sig = cand.get("sig") or ar.genome_sig(g)

    def _fn(sub, env, _sig=sig):
        return ar._candidate_train_cached(sub, env, _sig)

    var_full, var_contrib = _heldout_eval(subset, ar.genome_to_env(g), _fn)
    # Both sides onto the ACTIVE basis before comparing. The floor this verdict
    # is checked against comes from ar._adopt_floor(), which already switches to
    # AUC units on net_auc / net_gain / ens_auc - so an unre-keyed raw Score
    # (~0.4) would be compared against a floor of 0.005 and PASS everything.
    # rekey_rows is the identity on the raw basis. _rekeyed returns None when the
    # rows predate the column: unmeasurable, which verdict() reads as FAILED
    # rather than inventing a number right before an adoption decision.
    # The DECISION basis, not the search basis. They are the same constant until
    # a campaign names both, and the day they came apart the A/B was reporting a
    # mean that had no relationship to what the retrain would then do.
    basis = ar.decision_basis()
    ref_scored = ar._rekeyed(ref_full, basis=basis)
    var_scored = ar._rekeyed(var_full, basis=basis)
    if ref_scored is None or var_scored is None:
        return {"sig": sig, "p": None, "value": None, "n": 0,
                "p_neural": None, "value_neural": None,
                "promoted": 0, "demoted": 0, "p_promotion": 1.0}
    p, value, deltas, _tag = ar.holdout_stats(ref_scored, var_scored, objective)
    p_n, value_n, _d2, _t2 = ar.holdout_stats(ref_contrib, var_contrib,
                                              objective)
    # The decision the retrain would then make, counted on the SAME rows and on
    # the raw Score, whatever basis the verdict above is read in. It costs
    # nothing here and it is the only number that is about production.
    promo = ar.promotion_stats(ref_full, var_full)
    import statistics as _st
    sd = float(_st.stdev(deltas)) if len(deltas) > 1 else 0.0
    return {"sig": sig, "p": p, "value": value, "n": len(deltas), "sd": sd,
            "p_neural": p_n, "value_neural": value_n,
            "promoted": promo["promoted"], "demoted": promo["demoted"],
            "p_promotion": promo["p"]}


def ar_promotion_tag(stats):
    """The promotion counts as one line, or empty when nothing was comparable."""
    import auto_research as ar
    return ar.promotion_tag({"promoted": stats.get("promoted", 0),
                             "demoted": stats.get("demoted", 0),
                             "n": stats.get("promoted", 0) + stats.get("demoted", 0),
                             "p": stats.get("p_promotion", 1.0)})


# One-sided alpha 0.05 plus 80 percent power, as z scores: 1.645 + 0.842.
# Kept as one constant because both numbers are conventions, and a reader who
# wants different ones should see where they are.
Z_SUM = 2.487


def power(stats, floor):
    """What this A/B could have seen, given the spread it actually observed.

    A FAILED verdict is two different results wearing one word. Either the
    candidate was measured and did not clear the floor, or the holdout was
    never able to resolve an effect that size and the run says nothing. On
    2026-08-21 the second was the true state of every campaign A/B: per-asset
    deltas carry a standard deviation of about 3.74 in Score units, so a
    14-asset holdout can only resolve about +2.2, while the adoption floor is
    +0.5. Reporting the verdict without this number is reporting a null result
    that was never possible to disprove.

    Returns the minimum detectable effect at the observed spread, whether the
    run was powered for the floor it was judged against, and the holdout size
    that floor would need.
    """
    n = stats.get("n") or 0
    sd = stats.get("sd")
    if not n or sd is None or sd <= 0.0:
        return {"mde": None, "powered": None, "n_needed": None}
    mde = Z_SUM * sd / (n ** 0.5)
    needed = int((Z_SUM * sd / floor) ** 2 + 0.999) if floor > 0 else None
    return {"mde": mde, "powered": bool(mde <= floor), "n_needed": needed}


def power_tag(stats, floor):
    """The power reading as one line, or empty when it cannot be computed."""
    pw = power(stats, floor)
    if pw["mde"] is None:
        return ""
    if pw["powered"]:
        return "powered: could resolve %+.4g, floor %+.4g" % (pw["mde"], floor)
    return ("UNDERPOWERED: could only resolve %+.4g against a floor of %+.4g; "
            "%d assets would be needed, this run had %d"
            % (pw["mde"], floor, pw["n_needed"], stats.get("n") or 0))


def verdict(stats, floor, alpha):
    """PASSED only when the candidate beats the reference by the floor, on
    enough assets to mean anything.

    A candidate that beats production defaults but loses to the live genome must
    read as a failure: adopting it would be a regression. The sample is checked
    here and not only when the holdout was chosen: an arm that lost assets to a
    failed training leaves fewer deltas than the holdout had, and a handful of
    mildly positive ones reaches significance easily.

    The promotion count is a VETO, not a fourth criterion to average in. A
    candidate that would take a champion away from more assets than it wins is
    not an improvement to production whatever the mean of the basis says, and
    the mean is exactly what hid it on 2026-08-18: the passing arm carried 3
    promotions against 10 demotions in the rows the verdict was computed from.
    Deliberately a plain majority rather than a significant one - at fourteen
    assets significance is a high bar, and this only has to stop a candidate
    from being adopted while pointing the wrong way.
    """
    from core import holdout

    p, value = stats.get("p"), stats.get("value")
    n = stats.get("n") or 0
    if p is None or value is None:
        return "FAILED"
    if n < holdout.MIN_N:
        return "FAILED"
    if stats.get("demoted", 0) > stats.get("promoted", 0):
        return "FAILED"
    return "PASSED" if (p <= alpha and value >= floor) else "FAILED"


def write_result(cfg, results, base=None):
    """Write the run in the shape adopt_genome.candidates() already parses."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(base or BASE, f"_ab_genomes_{stamp}.json")
    payload = {
        "holdout": cfg["holdout"],
        "objective": cfg["objective"],
        # How many trainings each arm was averaged over. Recorded because sd_raw
        # is in different units without it: an r=4 spread is half an r=1 spread
        # on the same genome, and a reader comparing the two across files would
        # read the reseeding as an effect.
        "ab_seeds": len(seed_roll()),
        "floor": cfg["floor"],
        "alpha": cfg["alpha"],
        "reference": cfg["reference"],
        "reference_sig": cfg["reference_sig"],
        "results": {label: {"sig": st["sig"], "value_raw": st["value"],
                            "p_raw": st["p"], "n_raw": st["n"],
                            "sd_raw": st.get("sd"),
                            "mde": power(st, cfg["floor"])["mde"],
                            "powered": power(st, cfg["floor"])["powered"],
                            "value_neural": st["value_neural"],
                            "p_neural": st["p_neural"], "label": label,
                            "promoted": st.get("promoted"),
                            "demoted": st.get("demoted"),
                            "p_promotion": st.get("p_promotion")}
                    for label, st in results.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def run(cfg):
    """Train the reference, then every candidate, then report and write.

    Refuses when the adoption moved since the config was written: the env would
    come from the new adoption while reference_sig still named the old one, so
    the run would measure one thing and record another.
    """
    subset = cfg["holdout"]
    live = reference()
    if live["sig"] != cfg["reference_sig"]:
        print("The adoption changed since this config was written.")
        print("  config expects: %s" % (cfg["reference"] or "base"))
        print("  live now      : {}".format(live["label"]))
        print("Rebuild it: python ab_build.py")
        return
    ref = {"label": cfg["reference"], "sig": cfg["reference_sig"],
           "env": live["env"]}
    from core import ar_memory
    if not os.path.exists(DB_PATH):
        # data_fingerprint connects without mode=ro and would create an empty
        # file here, so the run would train on nothing and say so late.
        print(f"market.db not found at {DB_PATH}; run data_engine first.")
        return
    # Refuse an underpowered gate instead of only warning about it. The warning
    # has existed since 2026-08-21 and was printed at configure time; on
    # 2026-09-02 a 12-asset run was started anyway, spent nine hours, and
    # reported that it could resolve +1.74 against a floor of +0.5. The hours
    # are the same whether the question is answerable or not.
    pw = projected_power(len(subset.split(",")), cfg["floor"])
    if pw and "cannot answer" in pw and not _allow_underpowered():
        print(pw)
        print("Refusing to start. Add assets, raise the floor, or set "
              "GTRADE_AB_ALLOW_UNDERPOWERED=1 to run it anyway.")
        return
    fp_start = ar_memory.data_fingerprint(subset)
    print("Reference: {}   holdout: {}   data {}".format(ref["label"], subset, fp_start))
    seeds = seed_roll()
    if len(seeds) > 1:
        print("Averaging each arm over %d seeds (%s): %dx the trainings, the "
              "per-asset spread falls as sqrt(%d)."
              % (len(seeds), ",".join(str(s) for s in seeds), len(seeds),
                 len(seeds)))
    ref_full, ref_contrib = train_reference(subset, ref)
    if not ref_full:
        print("The reference arm produced no rows; stopping.")
        return
    results = {}
    for cand in cfg["candidates"]:
        print("\nTraining candidate {} ...".format(cand["label"]))
        results[cand["label"]] = evaluate(cand, subset, ref_full, ref_contrib,
                                          cfg["objective"])
    print("\n%s" % ("=" * 66))
    for label, st in results.items():
        v = verdict(st, cfg["floor"], cfg["alpha"])
        p_txt = "{:.4f}".format(st["p"]) if st["p"] is not None else "n/a"
        # %.4g, not %.2f: on an AUC basis the whole adoption floor is 0.005 and
        # would print as a reassuring "+0.01", with a real +0.065 finding shown
        # as "+0.07". Same reason ab_noise prints its residual this way.
        v_txt = "{:+.4g}".format(st["value"]) if st["value"] is not None else "n/a"
        print("  %-8s %s over %s   p=%s  n=%s   %s (floor %+.4g, alpha %.3f)"
              % (label, v_txt, cfg["reference"], p_txt, st["n"], v,
                 cfg["floor"], cfg["alpha"]))
        # Printed beside the verdict, not folded into it: a PASS with more
        # demotions than promotions is exactly the shape that cost a ten-hour
        # retrain on 2026-08-18, and the reader has to see both numbers.
        tag = ar_promotion_tag(st)
        if tag:
            print("  %-8s %s" % ("", tag))
        # And what the run could have seen at all. Without it a FAILED reads as
        # "no effect" when the honest reading is often "not measurable here".
        pw = power_tag(st, cfg["floor"])
        if pw:
            print("  %-8s %s" % ("", pw))
    if ar_memory.data_fingerprint(subset) != fp_start:
        print("\nWARNING: market.db changed while this run was in progress. The "
              "arms were measured over different windows, so the comparison is "
              "unreliable; rerun it on a quiet database.")
    path = write_result(cfg, results)
    print(f"\nWrote {os.path.basename(path)}")
    if all(verdict(st, cfg["floor"], cfg["alpha"]) != "PASSED"
           for st in results.values()):
        queue_per_asset(cfg)
    print("Next: python adopt_genome.py")


QUEUE_PATH = os.path.join(BASE, "_per_asset_queue.json")


def queue_per_asset(cfg, path=None):
    """After a gate that adopted nothing, say which assets it DID help.

    A failed verdict is a mean over a heterogeneous effect, and twice out of
    twice the failing run contained an asset that later survived a replication:
    RTX out of a run that read -0.30 overall, AUDCAD out of one that read +0.63.
    Leaving that on the floor throws away the expensive half of the run.

    Nothing is trained or adopted here. The picks are written down with the
    command that would confirm them, because confirming costs hours and this
    also runs inside an unattended loop that stops before the retrain by design.
    """
    try:
        import ab_confirm
        import ab_per_asset

        assets, deltas, label, result_file, _rec, _got = ab_per_asset.original()
        picks, why = ab_confirm.picks_from_scan(assets, deltas)
        if not picks:
            return None
        entry = {"result": result_file, "candidate": label, "picks": picks,
                 "why": why,
                 "confirm": "python ab_confirm.py --assets " + ",".join(picks)}
        with open(path or QUEUE_PATH, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False, indent=2)
        print()
        print("Nothing adopted, but the run was not empty. Per asset, %s:" % why)
        print("  " + ", ".join(picks))
        print("  " + entry["confirm"])
        return entry
    except (Exception, SystemExit) as exc:
        # SystemExit too: ab_per_asset REFUSES with one when it cannot identify
        # the arms, and SystemExit is not an Exception - so a refusal here would
        # have killed the gate's process after the hours were already spent.
        print("Per-asset follow-up skipped: %s" % exc)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-gate", type=int, default=None,
                    metavar="N",
                    help="the search's own gate list grown to N assets, for "
                         "GTRADE_AR_HELDOUT. Use --out: this module prints a "
                         "campaign banner at import, so stdout is not clean "
                         "enough for a launcher to read a variable from.")
    ap.add_argument("--out", default=None, metavar="FILE",
                    help="with --search-gate, write the list here instead of "
                         "to stdout")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--assets", default="")
    # core.holdout owns the size: a second default here silently overrode it,
    # so raising DEFAULT_N did nothing to the path the launcher actually uses.
    ap.add_argument("--n", type=int, default=_holdout_default_n())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--objective", default="mean")
    ap.add_argument("--auto", action="store_true",
                    help="pick the gate-adoptable elites and the suggested "
                         "holdout without asking; for auto_loop.py")
    args = ap.parse_args()

    if args.search_gate:
        names = search_gate(args.search_gate)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(",".join(names))
            print("%d asset(s) written to %s" % (len(names), args.out))
        else:
            print(",".join(names))
        return
    if args.show:
        cfg = read_config()
        if not cfg:
            print("No pending config. Run python ab_build.py first.")
        else:
            _print_config(cfg)
        return
    if args.run:
        cfg = read_config()
        if not cfg:
            print("No pending config. Run python ab_build.py first.")
            return
        run(cfg)
        return
    _configure(args)


def _apply_process_defaults():
    """Environment this PROCESS should run under, filled in where it is empty.

    Called from the entry point only, never from main() or run(). Both of these
    write to os.environ, which outlives the call: with the campaign basis and the
    load profile applied inside run(), one test that exercised an A/B left
    GTRADE_AR_TRAIN_CHUNK and GTRADE_AR_TRAIN_JOBS set for the whole pytest
    session, and every later test file trained for real through the chunked path.
    A process-wide default belongs to the process, not to a function an importer
    can call.

    basis first, then load: the basis decides what is measured (the floor, the
    re-keying, the verdict), the profile only how fast.
    """
    import auto_loop

    frozen = auto_loop.apply_campaign_basis()
    if frozen:
        print("campaign: %s" % " ".join("%s=%s" % kv for kv in sorted(frozen.items())))
    took = auto_loop.apply_load_profile()
    if took:
        print("load profile: %s" % " ".join("%s=%s" % kv for kv in sorted(took.items())))


if __name__ == "__main__":
    _apply_process_defaults()
    main()
