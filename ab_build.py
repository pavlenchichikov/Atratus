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
import glob
import io
import json
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

CONFIG_PATH = os.path.join(BASE, "_ab_config.json")
DB_PATH = os.path.join(BASE, "market.db")
MAX_CANDIDATES = 3


def _read_json(path):
    try:
        with io.open(path, encoding="utf-8") as fh:
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
    return {"label": "adopted:%s" % rec.get("label", "?"),
            "sig": ar.genome_sig(g), "env": ar.genome_to_env(g)}


def is_reference(cand, ref):
    """True when this candidate IS what we would measure it against."""
    return bool(ref.get("sig")) and cand.get("sig") == ref["sig"]


def previous_holdouts(base=None):
    """The holdout of every earlier A/B run, so none of them is reused."""
    out = []
    for path in sorted(glob.glob(os.path.join(base or BASE,
                                              "_ab_genomes_*.json"))):
        data = _read_json(path)
        if isinstance(data, dict) and data.get("holdout"):
            out.append(data["holdout"])
    return out


def bar_counts(db_path=None):
    """Rows per asset in market.db. A missing table counts as zero."""
    from config import FULL_ASSET_MAP
    out = {}
    con = sqlite3.connect("file:%s?mode=ro" % (db_path or DB_PATH), uri=True)
    try:
        cur = con.cursor()
        for asset in FULL_ASSET_MAP:
            try:
                out[asset] = cur.execute(
                    'SELECT COUNT(*) FROM "%s"' % asset.lower()).fetchone()[0]
            except sqlite3.Error:
                out[asset] = 0
    finally:
        con.close()
    return out


def build_config(candidates, assets, ref, floor, alpha, seed, objective):
    """The run description, with the reference written down.

    floor and alpha are frozen here rather than read at run time, so a later
    change to either constant cannot silently reinterpret a pending run.
    """
    return {
        "holdout": ",".join(assets),
        "objective": objective,
        "floor": floor,
        "alpha": alpha,
        "seed": seed,
        "reference": ref["label"],
        "reference_sig": ref["sig"],
        "candidates": [{"label": c["label"], "sig": c.get("sig"),
                        "genome": c["genome"]} for c in candidates],
    }


def write_config(cfg, path=None):
    with io.open(path or CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)


def read_config(path=None):
    return _read_json(path or CONFIG_PATH)


def _candidate_pool():
    """Archive elites that no A/B has measured yet.

    candidates() reports an elite as "measured" once a result file references its
    signature, so filtering to "search" hides exactly the ones already paid for.
    Offering one again would buy a number the archive already holds.
    """
    import adopt_genome
    return [c for c in adopt_genome.candidates() if c["kind"] == "search"]


def _suggest_assets(n, seed):
    import auto_research as ar
    from config import ASSET_TYPES, FULL_ASSET_MAP
    from core import holdout
    ex = holdout.excluded([ar.SELECTION_ASSETS, ar.HELDOUT_ASSETS,
                           ar.tier_assets()], previous_holdouts())
    elig = holdout.eligible(list(FULL_ASSET_MAP), bar_counts(), ex)
    return holdout.suggest(elig, ASSET_TYPES, n=n, seed=seed), elig


def _configure(args):
    import auto_research as ar
    from core import holdout

    import adopt_genome

    ref = reference()
    pool = _candidate_pool()
    if not pool:
        print("No unvalidated elites in the archive to test.")
        return
    print("Measuring against: %s\n" % ref["label"])
    for i, c in enumerate(pool, 1):
        warn = "  <- this IS the reference" if is_reference(c, ref) else ""
        print("  %d. %s%s" % (i, adopt_genome.describe(c), warn))
    raw = input("\nWhich numbers (comma separated, blank to cancel)? ").strip()
    picks = [p.strip() for p in raw.split(",") if p.strip().isdigit()]
    chosen = [pool[int(p) - 1] for p in picks if 1 <= int(p) <= len(pool)]
    if not chosen:
        print("Cancelled.")
        return
    if len(chosen) > MAX_CANDIDATES:
        print("At most %d per run: each arm is a full training of the holdout, "
              "roughly 8 to 11 hours, plus the reference. Run the rest against "
              "the same reference afterwards and the cache serves it free."
              % MAX_CANDIDATES)
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
            print("  - %s" % p)
        return

    cfg = build_config(chosen, assets, ref, ar._adopt_floor(args.objective),
                       args.alpha, args.seed, args.objective)
    write_config(cfg)
    print("\nWrote %s" % os.path.basename(CONFIG_PATH))
    _print_config(cfg)
    print("\nNext: python ab_build.py --run   (do not start it while a retrain "
          "is running)")


def _print_config(cfg):
    print("  reference : %s" % cfg["reference"])
    print("  holdout   : %s" % cfg["holdout"])
    print("  objective : %s   floor %+.2f   alpha %.3f"
          % (cfg["objective"], cfg["floor"], cfg["alpha"]))
    for c in cfg["candidates"]:
        g = c["genome"]
        print("  candidate : %-8s %d drops, %d extra, label %s/%s"
              % (c["label"], len(g.get("drops") or []),
                 len(g.get("extra") or []), g.get("label_mode", "direction"),
                 g.get("label_window", 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--assets", default="")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--objective", default="mean")
    args = ap.parse_args()

    if args.show:
        cfg = read_config()
        if not cfg:
            print("No pending config. Run python ab_build.py first.")
        else:
            _print_config(cfg)
        return
    if args.run:
        print("The run half is not wired yet.")
        return
    _configure(args)


if __name__ == "__main__":
    main()
