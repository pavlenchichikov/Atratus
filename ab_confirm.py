"""Does a per-asset selection replicate on a fresh seed roll?

ab_per_asset.py finds the assets where the candidate genome beat the reference by
more than their own seed noise. Those assets were SELECTED on the very rolls that
measured them, so their effect is biased upward: pick the three largest of forty
noisy numbers and they are large partly because the noise was kind. The only
honest answer is a fresh measurement of the same assets under seeds the selection
never saw.

This is deliberately NOT ab_build.py. That tool refuses a holdout that reuses
assets ("not eligible: already seen") and refuses fewer than eight, both correct
for a GATE - a gate must not be run twice on the same assets. A replication is
the opposite: the same assets on purpose, and no verdict of its own.

    python ab_confirm.py --dry     # plan, cost, and every guard, trains nothing
    python ab_confirm.py           (menu: [PA] then 2)

Step 2 of the per-asset adoption workflow, and the one that produces the numbers
step 3 is allowed to adopt on. Adopts nothing itself and writes nothing except
the training cache.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
from scipy import stats

import ab_per_asset
from core.console_status import Status, hm

BASE = ab_per_asset.BASE
# The A/B rolled seed_base + i*SEED_STRIDE = 1000,2000,3000,4000. A confirmation
# seed must be outside that set or the cache answers it without training and the
# run "confirms" itself.
DEFAULT_SEEDS = [11000, 12000, 13000, 14000]
DEFAULT_ASSETS = "KLAC,EVRG,RTX,ROSN,LTC"
# Measured on the 2026-09-02 A/B: a 7-asset chunk took about 98 min on the
# candidate arm and 26 on the reference. Per asset, for a rough plan only.
MIN_PER_ASSET = {"cand": 98 / 7, "ref": 26 / 7}


def _arms(cfg):
    """(ref_label, ref_sig, ref_env), (cand_label, cand_sig, cand_env).

    The reference comes from the live adoption, checked against the signature the
    config recorded - the same guard ab_build.run() makes, and for the same
    reason: the env would come from the new adoption while the label still named
    the old one.
    """
    import ab_build
    import auto_research as ar

    live = ab_build.reference()
    if live["sig"] != cfg["reference_sig"]:
        sys.exit("The adoption moved since this A/B was configured; the arms "
                 "would not be the ones that were measured.")
    cand = cfg["candidates"][0]
    g = ar.Genome(**cand["genome"])
    return ((live["label"] or "base", live["sig"], live["env"]),
            (cand["label"], cand["sig"], ar.genome_to_env(g)))


def _score_map(subset, env, sig):
    """{asset: Score} for one arm at the seed currently in the environment."""
    import auto_research as ar
    rows = ar._candidate_train_cached(subset, env, sig)
    return {r["Asset"]: float(r["Score"]) for r in (rows or [])
            if isinstance(r, dict) and isinstance(r.get("Score"), (int, float))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--chunk", type=int, default=0,
                    help="assets per training chunk; 0 splits them over the two "
                         "training processes")
    ap.add_argument("--dry", action="store_true",
                    help="run every guard and print the plan, train nothing")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not assets or not seeds:
        sys.exit("need at least one asset and one seed")

    cfg = ab_per_asset._load(ab_per_asset.CONFIG)
    original_seeds = set()
    import ab_build
    original_seeds = set(ab_build.seed_roll())
    clash = sorted(set(seeds) & original_seeds)
    if clash:
        sys.exit(f"seeds {clash} were used by the A/B itself: the cache would "
                 "answer them and the run would confirm its own numbers. "
                 "Pick seeds outside {}.".format(sorted(original_seeds)))

    # What the original run measured for these assets, with its own control.
    o_assets, o_deltas, label, result_file, _rec, _got = ab_per_asset.original()
    idx = {a: i for i, a in enumerate(o_assets)}
    missing = [a for a in assets if a not in idx]
    if missing:
        sys.exit(f"not in the A/B holdout, nothing to confirm: {', '.join(missing)}")
    o_mean = {a: float(o_deltas[idx[a]].mean()) for a in assets}
    o_se = {a: float(o_deltas[idx[a]].std(ddof=1) / math.sqrt(o_deltas.shape[1]))
            for a in assets}

    (ref_label, ref_sig, ref_env), (_cand_label, cand_sig, cand_env) = _arms(cfg)
    chunk = args.chunk or max(1, math.ceil(len(assets) / 2))
    est = len(seeds) * math.ceil(len(assets) / chunk) / 2 * chunk * (
        MIN_PER_ASSET["cand"] + MIN_PER_ASSET["ref"])

    print(f"confirming   : {label} vs {ref_label}   from {result_file}")
    print(f"assets       : {', '.join(assets)}")
    print(f"seeds        : {seeds}   (A/B used {sorted(original_seeds)})")
    print(f"chunk        : {chunk} assets, so both training processes are used")
    print(f"rough cost   : {est / 60:.1f} h  ({len(seeds) * len(assets) * 2} trainings)")
    print()
    print(f"  {'asset':<10}{'original':>10}{'se':>8}")
    for a in assets:
        print(f"  {a:<10}{o_mean[a]:+10.3f}{o_se[a]:8.3f}")
    if args.dry:
        print("\n--dry: every guard passed, nothing trained.")
        return

    subset = ",".join(assets)
    os.environ["GTRADE_AR_TRAIN_CHUNK"] = str(chunk)
    # The children inherit this and keep their own progress bars off: two chunk
    # processes and this status line all write to the one inherited console.
    os.environ["GTRADE_NO_TICKER"] = "1"
    was = os.environ.get("GTRADE_SEED")
    rolls = []
    status = Status(total_units=len(seeds) * 2, plan_min=est)
    status.start()
    print()
    try:
        for si, seed in enumerate(seeds):
            os.environ["GTRADE_SEED"] = str(seed)
            arms = {}
            for ai, (name, env, sig) in enumerate(
                    (("ref", ref_env, ref_sig), ("cand", cand_env, cand_sig))):
                status.unit(si * 2 + ai + 1, f"{name}@{seed}")
                arms[name] = _score_map(subset, env, sig)
                status.unit_done()
            common = sorted(set(arms["ref"]) & set(arms["cand"]))
            if not common:
                status.say(f"  seed {seed}: no asset scored on both arms, skipped")
                continue
            done = {a: arms["cand"][a] - arms["ref"][a] for a in common}
            rolls.append(done)
            status.say("  seed %-6d %s   [%s elapsed]"
                       % (seed, "  ".join(f"{a} {done[a]:+.2f}" for a in common),
                          hm(time.time() - status.t0)))
    finally:
        status.stop()
        os.environ.pop("GTRADE_NO_TICKER", None)
        if was is None:
            os.environ.pop("GTRADE_SEED", None)
        else:
            os.environ["GTRADE_SEED"] = was

    if not rolls:
        sys.exit("nothing was measured")
    common = sorted(set.intersection(*[set(r) for r in rolls]))
    n = len(rolls)
    print()
    print(f"replication over {n} fresh seed(s)")
    print(f"  {'asset':<10}{'original':>10}{'replication':>13}{'se':>8}{'t':>7}"
          f"{'p':>9}   reading")
    for a in common:
        d = np.array([r[a] for r in rolls], dtype=float)
        m = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        # Tested in the direction the original claimed, not always upward: an
        # asset selected for being HURT (ROSN, -5.1 at t -6.5) replicates by
        # being hurt again, and a one-sided upward test would score that as a
        # failure to confirm.
        side = 1.0 if o_mean[a] >= 0 else -1.0
        t = m / se if se > 0 else float("nan")
        p = (float(stats.t.sf(t * side, df=n - 1))
             if not math.isnan(t) else float("nan"))
        big = abs(m) >= ab_per_asset.FLOOR and m * side > 0
        if big and p < 0.05:
            reading = "HOLDS" if side > 0 else "HOLDS (harmful)"
        elif big:
            reading = "same side, not significant"
        elif m * o_mean[a] > 0:
            reading = "same sign, below floor"
        else:
            reading = "DOES NOT HOLD, sign flipped"
        print(f"  {a:<10}{o_mean[a]:+10.3f}{m:+13.3f}{se:8.3f}{t:7.2f}{p:9.4f}"
              f"   {reading}")
    # Selection bias is the thing being tested, so name it in the output rather
    # than leaving the reader to remember it. Measured IN THE DIRECTION EACH
    # ASSET CLAIMED, not as a plain mean of the deltas: a run that selected both
    # tails has a winner and a loser moving opposite ways, and averaging them
    # reported -0.25 on 2026-09-02 where every claimed effect had actually lost
    # about 3.2 - the summary said "barely any curse" about a set that kept
    # under a third of what it promised.
    lost = []
    for a in common:
        side = 1.0 if o_mean[a] >= 0 else -1.0
        lost.append((o_mean[a] - float(np.mean([r[a] for r in rolls]))) * side)
    print()
    print("claimed effect lost to selection: %+.2f on average (positive means the"
          " replication came back smaller than the discovery)" % float(np.mean(lost)))


if __name__ == "__main__":
    main()
