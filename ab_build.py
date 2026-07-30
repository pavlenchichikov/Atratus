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


def _heldout_eval(subset, env, fn, **kw):
    """Indirected so the tests can see which trainer the reference chose."""
    import auto_research as ar
    return ar._heldout_eval(subset, env, fn, **kw)


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
    p, value, deltas, _tag = ar.holdout_stats(ref_full, var_full, objective)
    p_n, value_n, _d2, _t2 = ar.holdout_stats(ref_contrib, var_contrib,
                                              objective)
    return {"sig": sig, "p": p, "value": value, "n": len(deltas),
            "p_neural": p_n, "value_neural": value_n}


def verdict(stats, floor, alpha):
    """PASSED only when the candidate beats the reference by the floor.

    A candidate that beats production defaults but loses to the live genome must
    read as a failure: adopting it would be a regression.
    """
    p, value = stats.get("p"), stats.get("value")
    if p is None or value is None:
        return "FAILED"
    return "PASSED" if (p <= alpha and value >= floor) else "FAILED"


def write_result(cfg, results, base=None):
    """Write the run in the shape adopt_genome.candidates() already parses."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(base or BASE, "_ab_genomes_%s.json" % stamp)
    payload = {
        "holdout": cfg["holdout"],
        "objective": cfg["objective"],
        "floor": cfg["floor"],
        "alpha": cfg["alpha"],
        "reference": cfg["reference"],
        "reference_sig": cfg["reference_sig"],
        "results": {label: {"sig": st["sig"], "value_raw": st["value"],
                            "p_raw": st["p"], "n_raw": st["n"],
                            "value_neural": st["value_neural"],
                            "p_neural": st["p_neural"], "label": label}
                    for label, st in results.items()},
    }
    with io.open(path, "w", encoding="utf-8") as fh:
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
        print("  live now      : %s" % live["label"])
        print("Rebuild it: python ab_build.py")
        return
    ref = {"label": cfg["reference"], "sig": cfg["reference_sig"],
           "env": live["env"]}
    print("Reference: %s   holdout: %s" % (ref["label"], subset))
    ref_full, ref_contrib = train_reference(subset, ref)
    if not ref_full:
        print("The reference arm produced no rows; stopping.")
        return
    results = {}
    for cand in cfg["candidates"]:
        print("\nTraining candidate %s ..." % cand["label"])
        results[cand["label"]] = evaluate(cand, subset, ref_full, ref_contrib,
                                          cfg["objective"])
    print("\n%s" % ("=" * 66))
    for label, st in results.items():
        v = verdict(st, cfg["floor"], cfg["alpha"])
        p_txt = "%.4f" % st["p"] if st["p"] is not None else "n/a"
        v_txt = "%+.2f" % st["value"] if st["value"] is not None else "n/a"
        print("  %-8s %s over %s   p=%s  n=%s   %s (floor %+.2f, alpha %.3f)"
              % (label, v_txt, cfg["reference"], p_txt, st["n"], v,
                 cfg["floor"], cfg["alpha"]))
    path = write_result(cfg, results)
    print("\nWrote %s" % os.path.basename(path))
    print("Next: python adopt_genome.py")


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
        cfg = read_config()
        if not cfg:
            print("No pending config. Run python ab_build.py first.")
            return
        run(cfg)
        return
    _configure(args)


if __name__ == "__main__":
    main()
