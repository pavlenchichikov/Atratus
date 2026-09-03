"""Is the A/B's per-asset spread real heterogeneity, or seed noise?

The gate reports one mean over the holdout and a promotion count (14 promote,
24 demote on 2026-09-02) and then decides ONE genome for every asset. If the
genome's effect genuinely differs by asset, that count is a signal and per-asset
adoption is worth building; if the per-asset deltas are just seed noise around a
common effect, the count is coin flips and the mechanism would select noise.

Nothing here trains. Every arm of the last A/B is already on disk: ab_build
averages each arm over GTRADE_AB_SEEDS seeds, and each (seed, chunk) training
was cached separately in _ar_eval_cache.json, so the four per-seed measurements
of each asset survive even though only their mean reached the verdict.

Identification is by TIME, not by cache key: a key hashes data_fingerprint, and
market.db has moved since the run, so the keys no longer reproduce. gtrade.log
stamps every training phase with the genome signature it was about, which gives
the phase windows; a cache entry belongs to the phase whose window contains its
timestamp. That is an inference, so it is checked rather than trusted: the mean
of the recovered deltas has to reproduce the value_raw the run recorded, and the
script refuses to report anything if it does not.

    python ab_per_asset.py          (menu: [PA] then 1)

Step 1 of the per-asset adoption workflow, so it ships with the project rather
than living beside it like the throwaway ab_* harnesses: the menu offers it, and
a menu entry pointing at a file that is not in the repository is not an offer.
"""

import glob
import json
import os
import re
import sys

# timezone.utc below, not datetime.UTC: this runs in both environments and the
# training one is pinned to Python 3.10 (TF 2.10), where that alias does not
# exist. ruff's UP017 does not know that, hence the suppressions.
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "_ar_eval_cache.json")
LOG = os.path.join(BASE, "gtrade.log")
CONFIG = os.path.join(BASE, "_ab_config.json")

FDR_Q = 0.10          # Benjamini-Hochberg level for the per-asset selection
FLOOR = 0.5           # the adoption floor: significance alone is not enough
TOL = 1e-9            # how exactly the recovered mean must match the recorded one

# "[regate full {"drops": ["] 6 chunks on 2 processes: ABT,..." - the signature is
# truncated in the log, but the only thing needed from it is whether the drops
# list is empty, which is what separates the two arms of this run.
PHASE_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*\[regate (full|cb) (\{.*?)\] \d+ chunks on")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _utc_offset_seconds():
    """Local clock minus UTC, in seconds.

    ar_memory.cache_put stamps entries with datetime.utcnow() while gtrade.log
    is written in local time, so the two are three hours apart on this box and a
    naive comparison files every entry under the wrong phase. Taken from the
    machine rather than hardcoded; Moscow has no DST, so one offset holds for the
    whole run.
    """
    now = datetime.now()
    utc = datetime.now(timezone.utc).replace(tzinfo=None)  # noqa: UP017
    return round((now - utc).total_seconds() / 60.0) * 60


def phases(log_path=None, since=None, until=None):
    """Training phases in order: (start, kind, arm). arm is 'ref' or 'cand'.

    Timestamps are returned in UTC to match the cache. The window of a phase runs
    to the start of the next one, so an entry is placed by the phase it falls
    after.

    `until` matters once anything else trains this genome again - a confirmation
    run writes the same '[regate full ...]' lines and its own cache entries, and
    without an upper bound they would be read as more seeds of the original A/B.

    The path is resolved on the CALL, not bound as a default: a default is
    evaluated once at import, so a caller pointing LOG somewhere else - a test,
    or a second checkout - would have gone on reading the original file.

    ROTATED files are read too, oldest first. The log rotates at a few megabytes
    and one training run writes megabytes, so a gate finished yesterday can have
    its phases in gtrade.log.1 while gtrade.log starts this morning - which is
    exactly what happened on 2026-09-03, and the tool answered "no training
    phases found" as if the run had never existed.
    """
    paths = [log_path] if log_path else _log_files()
    offset = _utc_offset_seconds()
    out = []
    for one in paths:
        out += _phases_in(one, offset, since, until)
    out.sort()
    return out


def _log_files(log_path=None):
    """The log and its rotated siblings, oldest first."""
    base = log_path or LOG
    rotated = sorted(glob.glob(base + ".*"),
                     key=lambda p: -_suffix_number(p))
    return [p for p in rotated if os.path.exists(p)] + [base]


def _suffix_number(path):
    tail = path.rsplit(".", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _phases_in(log_path, offset, since, until):
    out = []
    if not os.path.exists(log_path):
        return out
    with open(log_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = PHASE_RE.match(line)
            if not m:
                continue
            # Normalised to the cache's UTC ISO form BEFORE any comparison: the
            # log separates date and time with a space, which sorts BELOW 'T', so
            # a raw compare against an ISO cutoff drops the whole first day.
            local = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            start = (local - timedelta(seconds=offset)).isoformat()
            kind, sig = m.group(2), m.group(3)
            if (since and start < since) or (until and start > until):
                continue
            # '{"drops": ["' means the reference carries feature drops; the
            # candidate of this run drops nothing, so its fragment is '{"drops": []'.
            arm = "ref" if '"drops": ["' in sig else "cand"
            out.append((start, kind, arm))
    return out


def rolls(cache, holdout, phase_list):
    """{(arm, seed_index): {asset: Score}} for the full trainings only."""
    full = [p for p in phase_list if p[1] == "full"]
    if not full:
        raise SystemExit("no training phases found in the log")
    bounds = [p[0] for p in phase_list] + ["9999"]

    per_phase = {}
    for entry in cache.values():
        ts = str(entry.get("ts") or "")
        rows = [r for r in (entry.get("rows") or []) if isinstance(r, dict)]
        assets = {r.get("Asset") for r in rows}
        if not rows or not assets <= holdout or ts < bounds[0]:
            continue
        # the phase this entry falls inside
        idx = None
        for i, p in enumerate(phase_list):
            nxt = bounds[i + 1]
            if p[0] <= ts < nxt:
                idx = i
                break
        if idx is None or phase_list[idx][1] != "full":
            continue
        bucket = per_phase.setdefault(phase_list[idx][0], {})
        for r in rows:
            score = r.get("Score")
            if isinstance(score, (int, float)):
                bucket[r["Asset"]] = float(score)

    seen = {"ref": 0, "cand": 0}
    out = {}
    for start, _kind, arm in full:
        got = per_phase.get(start)
        if not got:
            continue
        out[(arm, seen[arm])] = got
        seen[arm] += 1
    return out


def paired(roll_map):
    """(assets, deltas) with deltas[i][s] = cand - ref for asset i, seed s.

    Only assets every roll scored, which is the rule _mean_rows already applies:
    an asset averaged over fewer seeds than its neighbours would carry a
    different amount of noise with nothing on the row to say so.
    """
    ref = [roll_map[k] for k in sorted(roll_map) if k[0] == "ref"]
    cand = [roll_map[k] for k in sorted(roll_map) if k[0] == "cand"]
    if not ref or not cand:
        raise SystemExit("one of the two arms was not recovered")
    n = min(len(ref), len(cand))
    ref, cand = ref[:n], cand[:n]
    common = set(ref[0])
    for r in ref[1:] + cand:
        common &= set(r)
    assets = sorted(common)
    deltas = np.array([[cand[s][a] - ref[s][a] for s in range(n)] for a in assets])
    return assets, deltas


def bh(pvals, q):
    """Benjamini-Hochberg: the boolean mask of rejections at level q."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        cut = np.max(np.nonzero(passed)[0])
        keep[order[: cut + 1]] = True
    return keep


def original(verify=True):
    """(assets, deltas, label, result_file) for the last A/B, from its cache.

    Shared with ab_confirm.py, which needs the same recovery to compare a
    replication against. `verify` is the whole reason the numbers are usable:
    the mean of the recovered deltas has to reproduce the value the run wrote
    down, or the identification is wrong and nothing is returned.
    """
    if not (os.path.exists(CACHE) and os.path.exists(LOG) and os.path.exists(CONFIG)):
        sys.exit("need _ar_eval_cache.json, gtrade.log and _ab_config.json in the repo root")
    cfg = _load(CONFIG)
    holdout = {a.strip() for a in cfg["holdout"].split(",") if a.strip()}
    cache = _load(CACHE)

    result_files = sorted(f for f in os.listdir(BASE)
                          if f.startswith("_ab_genomes_") and f.endswith(".json"))
    if not result_files:
        sys.exit("no _ab_genomes_*.json result to check against")
    result_file = result_files[-1]
    result = _load(os.path.join(BASE, result_file))
    label, rec = next(iter(result["results"].items()))
    recorded = rec.get("value_raw")

    # The run started when its config was written and ended when it wrote its
    # result: phases outside that belong to other campaigns, or to a later
    # confirmation run of this same genome.
    since = datetime.fromtimestamp(os.path.getmtime(CONFIG),
                                   timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")  # noqa: UP017
    until = datetime.fromtimestamp(os.path.getmtime(os.path.join(BASE, result_file)),
                                   timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")  # noqa: UP017
    ph = phases(since=since, until=until)
    roll_map = rolls(cache, holdout, ph)
    assets, deltas = paired(roll_map)
    got = float(deltas.mean(axis=1).mean())
    if verify and (recorded is None or abs(got - recorded) > TOL):
        sys.exit(f"MISMATCH: recomputed {got:+.10f} vs recorded {recorded}; "
                 "the arms were not identified correctly, refusing to report.")
    return assets, deltas, label, result_file, recorded, got


def adoption_state(assets):
    """(per-asset genome map, {asset: "own"|"global"|"-"}, adoption dates).

    The reference arm already carries every adoption: it runs through the same
    config.py, whose ADOPTED_ENV_KEYS resolves genome_for_assets. So a delta IS
    measured against what the asset is on today - but nothing on the row SAID
    so, and an asset on its own genome reads identically to one on the global
    default. They are different claims: -3.8 on an adopted asset means this
    pass would UNDO an adoption that was measured and replicated, and -3.8 on
    an unadopted one just means the candidate does not help there.
    """
    from core import adopted

    record = adopted.load() or {}
    own = adopted.per_asset(record)
    has_global = bool(record.get("genome"))
    where = {}
    for asset in assets:
        key = str(asset).strip().upper()
        where[asset] = "own" if key in own else ("global" if has_global else "-")
    dates = {k.strip().upper(): (v or {}).get("adopted")
             for k, v in (record.get("per_asset") or {}).items()
             if isinstance(v, dict)}
    dates["_global"] = record.get("adopted")
    return own, where, dates


def run_started():
    """When this A/B's reference arm can first have run: its config's mtime.

    original() bounds the arm search the same way, so this is the same clock
    and not a second, drifting opinion about when the run began.
    """
    if not os.path.exists(CONFIG):
        return None
    return datetime.fromtimestamp(os.path.getmtime(CONFIG),
                                  timezone.utc).strftime("%Y-%m-%d")  # noqa: UP017


def stale_baseline(started, dates):
    """Adoptions made AFTER the reference arm ran. They invalidate the deltas.

    The reference arm is only "the current adopted state" if it ran under it.
    Adopt something, then compare a candidate against a baseline measured
    before that adoption, and the candidate is credited with a gain that was
    already banked. The eval cache is keyed by data fingerprint and seed, not
    by genome, so a pre-adoption row survives the adoption and gets reused
    without saying anything.
    """
    if not started:
        return []
    return sorted(name for name, when in dates.items()
                  if when and str(when)[:10] > started)


def main():
    assets, deltas, label, result_file, recorded, got = original(verify=False)
    n_seeds = deltas.shape[1]
    per_asset = deltas.mean(axis=1)

    print(f"A/B result   : {result_file}  candidate {label}")
    print(f"recovered    : {len(assets)} assets x {n_seeds} seeds per arm")
    print(f"check        : recomputed mean {got:+.10f} vs recorded {recorded:+.10f}")
    if recorded is None or abs(got - recorded) > TOL:
        sys.exit("MISMATCH: the arms were not identified correctly; refusing to report.")
    print("               match, the arms are the ones the verdict was computed from")

    # Which genome each asset's REFERENCE arm was running, and whether anything
    # was adopted after that arm was measured.
    _own, where, dates = adoption_state(assets)
    stale = stale_baseline(run_started(), dates)
    n_own = sum(1 for a in assets if where.get(a) == "own")
    print(f"adoption     : {n_own} of {len(assets)} holdout assets are on a "
          f"genome of their own; the rest on the global one")

    # Per-asset paired statistics. se is the standard error of THIS asset's own
    # mean, so it says how much of that asset's delta is the seed re-roll.
    sd = deltas.std(axis=1, ddof=1)
    se = sd / np.sqrt(n_seeds)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, per_asset / se, 0.0)
    p_one = stats.t.sf(t, df=n_seeds - 1)

    # Variance decomposition: what the spread across assets is actually made of.
    within = float(np.mean(se ** 2))
    total = float(np.var(per_asset, ddof=1))
    between = total - within
    print()
    print(f"spread of the per-asset delta : sd {np.sqrt(total):.3f}")
    print(f"  seed noise in one asset's mean: {np.sqrt(within):.3f}")
    print(f"  genuine between-asset spread  : "
          f"{np.sqrt(between):.3f}" if between > 0 else
          "  genuine between-asset spread  : none, the noise explains all of it")
    if between > 0:
        print(f"  share of the variance that is real: {between / total:.0%}")

    # 80% power at one-sided 0.05 with n_seeds-1 df.
    factor = stats.t.ppf(0.95, n_seeds - 1) + stats.t.ppf(0.80, n_seeds - 1)
    print(f"per-asset detectable effect at 80% power: {factor * np.median(se):+.2f} "
          f"(median se {np.median(se):.2f}, floor {FLOOR:+.2f})")

    keep = bh(p_one, FDR_Q)
    print()
    print(f"Benjamini-Hochberg at q={FDR_Q:.2f}, one-sided:")
    order = np.argsort(-per_asset)
    print(f"  {'asset':<12}{'on':>7}{'delta':>9}{'se':>8}{'t':>7}{'p':>9}   verdict")
    for i in order:
        if not (keep[i] or per_asset[i] >= FLOOR or i in order[:5] or i in order[-3:]
                or where.get(assets[i]) == "own"):
            continue
        verdict = "ADOPT" if (keep[i] and per_asset[i] >= FLOOR) else (
            "significant, below floor" if keep[i] else
            "above floor, not significant" if per_asset[i] >= FLOOR else "")
        print(f"  {assets[i]:<12}{where.get(assets[i], '-'):>7}"
              f"{per_asset[i]:+9.3f}{se[i]:8.3f}{t[i]:7.2f}"
              f"{p_one[i]:9.4f}   {verdict}")
    n_adopt = int(np.sum(keep & (per_asset >= FLOOR)))
    print()
    print(f"assets that would take the genome: {n_adopt} of {len(assets)}")

    # Every asset already on a genome of its own, and what this pass did to it.
    # This is the question a pass is actually run to answer: the point of an
    # adoption is that it improves, so a pass that makes an adopted asset worse
    # is a pass that would undo measured, replicated work.
    adopted_idx = [i for i in order if where.get(assets[i]) == "own"]
    if adopted_idx:
        print()
        print("Against the genome each asset is ALREADY on:")
        for i in adopted_idx:
            gain = per_asset[i]
            reading = ("improves on it" if gain >= FLOOR and keep[i] else
                       "improves, but not past the noise" if gain > 0 else
                       "WOULD UNDO IT" if gain <= -FLOOR else "no change worth the name")
            print(f"  {assets[i]:<12}{gain:+9.3f}   {reading}"
                  f"   (adopted {dates.get(assets[i].upper()) or '?'})")
        harmed = [assets[i] for i in adopted_idx if per_asset[i] <= -FLOOR]
        if harmed:
            print()
            print("  Adopting this candidate GLOBALLY would demote: %s."
                  % ", ".join(harmed))
            print("  Those keep their own genome through core/adopted.py, so a")
            print("  global adoption does not touch them - but the same lever")
            print("  measured on them is a different lever from the one that won")
            print("  here, and this is where that shows.")
    elif where:
        print()
        print("No asset in this holdout is on a genome of its own, so every "
              "delta above is against the global adoption.")

    if stale:
        print()
        print("ADOPTED AFTER this run's reference arm was measured: %s."
              % ", ".join(stale))
        print("Read it one of two ways, and they are opposites:")
        print("  If this IS the run those adoptions were made from, the delta")
        print("  above is the DISCOVERY, not a fresh improvement on top of it.")
        print("  Reading it as the latter counts the same gain twice.")
        print("  If it is a LATER run, its baseline predates the adoption and")
        print("  the candidate is credited with a gain already banked. The eval")
        print("  cache is keyed by data fingerprint and seed, not by genome, so")
        print("  the stale row survives the adoption and is reused in silence.")
        print("  Re-measure the reference arm before believing the number.")
    if not n_adopt:
        print("Nothing survives the correction: at this many seeds the per-asset")
        print("delta cannot be told from its own re-roll, so a per-asset adoption")
        print("mechanism would be selecting noise.")


if __name__ == "__main__":
    main()
