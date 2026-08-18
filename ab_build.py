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


def previous_holdouts(base=None):
    """The holdout of every earlier A/B run, so none of them is reused.

    The adopted record's own holdout is included too: its result file can be
    moved or archived, and those assets must not quietly become drawable again.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(base or BASE,
                                              "_ab_genomes_*.json"))):
        data = _read_json(path)
        if isinstance(data, dict) and data.get("holdout"):
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
    """
    import auto_research as ar
    return {
        "holdout": ",".join(assets),
        "objective": objective,
        "basis": ar._score_basis(),
        "floor": floor,
        "alpha": alpha,
        "seed": seed,
        "reference": ref["label"],
        "reference_sig": ref["sig"],
        "candidates": [{"label": c["label"], "sig": c.get("sig"),
                        "genome": c["genome"]} for c in candidates],
    }


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
            prev = out.get(sig)
            if prev is None or ts >= prev["ts"]:
                out[sig] = {"ts": ts, "tag": w.get("tag"),
                            "adoptable": bool(w.get("adoptable")),
                            "clears": w.get("clears") or 0}
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


def _suggest_assets(n, seed):
    import auto_research as ar
    from config import ASSET_TYPES, FULL_ASSET_MAP
    from core import holdout
    ex = holdout.excluded([ar.SELECTION_ASSETS, ar.HELDOUT_ASSETS,
                           ar.tier_assets()], previous_holdouts())
    elig = holdout.eligible(list(FULL_ASSET_MAP), bar_counts(), ex)
    return holdout.suggest(elig, ASSET_TYPES, n=n, seed=seed), elig


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
        cfg = build_config(chosen, assets, ref, ar._adopt_floor(args.objective),
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
        print("At most %d per run: each arm is a full training of the holdout, "
              "roughly 8 to 11 hours, plus the reference. A later run draws a "
              "fresh holdout, so its reference arm is trained again from "
              "scratch - picking fewer now does not save that." % MAX_CANDIDATES)
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

    cfg = build_config(chosen, assets, ref, ar._adopt_floor(args.objective),
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
    # Both sides onto the ACTIVE basis before comparing. The floor this verdict
    # is checked against comes from ar._adopt_floor(), which already switches to
    # AUC units on net_auc / net_gain / ens_auc - so an unre-keyed raw Score
    # (~0.4) would be compared against a floor of 0.005 and PASS everything.
    # rekey_rows is the identity on the raw basis. _rekeyed returns None when the
    # rows predate the column: unmeasurable, which verdict() reads as FAILED
    # rather than inventing a number right before an adoption decision.
    ref_scored, var_scored = ar._rekeyed(ref_full), ar._rekeyed(var_full)
    if ref_scored is None or var_scored is None:
        return {"sig": sig, "p": None, "value": None, "n": 0,
                "p_neural": None, "value_neural": None}
    p, value, deltas, _tag = ar.holdout_stats(ref_scored, var_scored, objective)
    p_n, value_n, _d2, _t2 = ar.holdout_stats(ref_contrib, var_contrib,
                                              objective)
    return {"sig": sig, "p": p, "value": value, "n": len(deltas),
            "p_neural": p_n, "value_neural": value_n}


def verdict(stats, floor, alpha):
    """PASSED only when the candidate beats the reference by the floor, on
    enough assets to mean anything.

    A candidate that beats production defaults but loses to the live genome must
    read as a failure: adopting it would be a regression. The sample is checked
    here and not only when the holdout was chosen: an arm that lost assets to a
    failed training leaves fewer deltas than the holdout had, and a handful of
    mildly positive ones reaches significance easily.
    """
    from core import holdout

    p, value = stats.get("p"), stats.get("value")
    n = stats.get("n") or 0
    if p is None or value is None:
        return "FAILED"
    if n < holdout.MIN_N:
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
    fp_start = ar_memory.data_fingerprint(subset)
    print("Reference: {}   holdout: {}   data {}".format(ref["label"], subset, fp_start))
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
    if ar_memory.data_fingerprint(subset) != fp_start:
        print("\nWARNING: market.db changed while this run was in progress. The "
              "arms were measured over different windows, so the comparison is "
              "unreliable; rerun it on a quiet database.")
    path = write_result(cfg, results)
    print(f"\nWrote {os.path.basename(path)}")
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
    ap.add_argument("--auto", action="store_true",
                    help="pick the gate-adoptable elites and the suggested "
                         "holdout without asking; for auto_loop.py")
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
