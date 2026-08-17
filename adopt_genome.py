"""Pick a genome, adopt it, and print what to run next.

Two sources, and they are not equivalent. A candidate from an A/B run carries a
value MEASURED on a holdout it had never seen, with a p-value. An elite from the
search archive carries only its search fitness, which shrinks: genome A scored
5.30 in the search and 1.63 on a fresh holdout. So an unvalidated elite can only
be adopted with --unvalidated, on purpose.

The adopted file is gitignored: this repository is public and the genome is the
edge. Revert therefore keeps the previous adoption beside it rather than relying
on git history.

Usage:
  python adopt_genome.py                # list, pick, write, print next steps
  python adopt_genome.py --show         # what is live now, with its evidence
  python adopt_genome.py --revert       # restore the previous adoption
  python adopt_genome.py --unvalidated  # also allow a search-stage elite
"""
import argparse
import datetime
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

PROGRESS_FILES = ("_chunk_progress.txt", "_chunk_quality.json")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _archive_by_sig(base):
    """Every archive elite keyed by its genome signature.

    The join that recovers a full genome from an A/B result, whose stored
    signature carries the DSL specs without their names.
    """
    import auto_research as ar

    arch = _read_json(os.path.join(base, "_qd_archive.json")) or {}
    out = {}
    for key, entry in arch.items():
        genome = (entry or {}).get("genome")
        if not isinstance(genome, dict):
            continue
        try:
            sig = ar.genome_sig(ar.Genome(**genome))
        except Exception:
            continue
        out[sig] = {"bucket": key, "genome": genome,
                    "fitness": (entry or {}).get("fitness")}
    return out


def candidates(base=None):
    """Adoptable genomes, measured ones first, each tagged with its evidence."""
    base = base or BASE
    by_sig = _archive_by_sig(base)
    seen, out = set(), []
    for path in sorted(glob.glob(os.path.join(base, "_ab_genomes_*.json"))):
        data = _read_json(path) or {}
        holdout = data.get("holdout")
        alpha = data.get("alpha") or 0.05
        floor = data.get("floor") or 0.0
        for label, res in (data.get("results") or {}).items():
            if not isinstance(res, dict):
                continue
            hit = by_sig.get(res.get("sig"))
            if not hit:
                continue
            seen.add(res.get("sig"))
            p = res.get("p_raw")
            value = res.get("value_raw")
            # Coming from an A/B file is provenance, not a pass. The run records
            # its own alpha and floor; a candidate that missed either FAILED and
            # must not be offered as a decision.
            passed = (p is not None and value is not None
                      and p <= alpha and value >= floor)
            out.append({
                "kind": "measured", "validated": passed, "label": label,
                "genome": hit["genome"], "bucket": hit["bucket"],
                "value": value, "p": p,
                "n": res.get("n_raw"), "holdout": holdout,
                "source": os.path.basename(path),
                "neural": res.get("value_neural"),
                "alpha": alpha, "floor": floor,
                "sig": res.get("sig"),
            })
    for sig, hit in sorted(by_sig.items(), key=lambda kv: kv[1]["bucket"]):
        if sig in seen:
            continue
        out.append({
            "kind": "search", "validated": False, "label": hit["bucket"],
            "genome": hit["genome"], "bucket": hit["bucket"],
            "value": hit["fitness"], "p": None, "n": None, "holdout": None,
            "source": "_qd_archive.json", "neural": None,
            "sig": sig,
        })
    return out


def best_validated(cands):
    """The candidate an unattended adopt should take, or None.

    Only candidates that PASSED their own A/B: `validated` already carries the
    run's own alpha and floor, so this never re-decides significance, it only
    ranks what already cleared. Highest measured value wins, ties go to the
    smaller p. A search-archive elite can never be picked here - that stays
    --unvalidated, which is a human decision on purpose.
    """
    passed = [c for c in cands
              if c.get("validated") and c.get("value") is not None]
    if not passed:
        return None
    return max(passed, key=lambda c: (c["value"],
                                      -(c["p"] if c["p"] is not None else 1.0)))


def describe(cand):
    """One line for the picker. A search fitness is never called a gain."""
    g = cand["genome"]
    bits = ["%d drops" % len(g.get("drops") or []),
            "%d extra" % len(g.get("extra") or []),
            "label {}/{}".format(g.get("label_mode", "direction"),
                             g.get("label_window", 30))]
    # The tuning genes, shown only when they leave their default. Four elites of
    # the same feature family print an identical line without them, and the
    # picker then cannot express which one the reader means.
    if g.get("thr_margin"):
        bits.append("thr %.3f" % g["thr_margin"])
    if g.get("band_delta"):
        bits.append("band %+.4f" % g["band_delta"])
    if g.get("regime_mode", "both") != "both":
        bits.append("regime %s" % g["regime_mode"])
    # %.4g: on an AUC basis a +0.065 finding and a 0.005 floor both round to
    # two decimals as "+0.07" and "+0.01" on the one screen where the adoption
    # decision is actually taken.
    value = "{:+.4g}".format(cand["value"]) if cand["value"] is not None else "n/a"
    if cand["kind"] == "measured":
        p = "{:.4f}".format(cand["p"]) if cand["p"] is not None else "n/a"
        n = cand["n"] if cand["n"] is not None else "?"
        verdict = "PASSED" if cand["validated"] else "FAILED its own A/B"
        head = f"{verdict}  value {value}  p={p}  n={n}"
    else:
        head = f"search fitness {value}  (NOT validated on a fresh holdout)"
    return "%-6s %s | %s" % (cand["label"], head, ", ".join(bits))


def _caveat(cand):
    """The honest limitation of this candidate's evidence, recorded with it.

    Without this the record says what is live but not what the measurement did
    not cover, and the next reader cannot tell a replicated finding from a guess.
    """
    if cand["kind"] != "measured":
        return ("search fitness only; never validated on a holdout it had not "
                "seen. Search values shrink: genome A scored 5.30 in the search "
                "and 1.63 on a fresh holdout.")
    if not cand["validated"]:
        return ("this candidate FAILED its own A/B (p={} against alpha {}, value "
                "{} against floor {})".format(cand["p"], cand.get("alpha"),
                                          cand["value"], cand.get("floor")))
    holdout = cand.get("holdout") or ""
    return ("measured on %d held-out assets it had not seen (%s). "
            "Generalisation to the full asset list is an assumption, not a "
            "measurement." % (len([a for a in holdout.split(",") if a]), holdout))


def write_adoption(cand, path=None):
    """Write the adoption, keeping any current one as the previous."""
    from core import adopted as _adopted

    dest = path or _adopted.PATH
    prev = dest.replace(".json", ".prev.json")
    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as fh:
            body = fh.read()
        with open(prev, "w", encoding="utf-8") as fh:
            fh.write(body)
    record = {
        "adopted": datetime.date.today().isoformat(),
        "label": cand["label"],
        "evidence": {
            "kind": cand["kind"],
            "value": cand["value"],
            "p": cand["p"],
            "n": cand["n"],
            "holdout": cand["holdout"],
            "source": cand["source"],
            "bucket": cand["bucket"],
            "neural_value": cand["neural"],
            "alpha": cand.get("alpha"),
            "floor": cand.get("floor"),
            "caveat": _caveat(cand),
        },
        "genome": cand["genome"],
    }
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)


def revert(path=None):
    """Restore the previous adoption, or remove the file to go back to base."""
    from core import adopted as _adopted

    dest = path or _adopted.PATH
    prev = dest.replace(".json", ".prev.json")
    if os.path.exists(prev):
        with open(prev, encoding="utf-8") as fh:
            body = fh.read()
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.remove(prev)
        return True
    if os.path.exists(dest):
        os.remove(dest)
        return True
    return False


def reset_chunk_progress(base=None):
    """Clear the chunked trainer's resume state. Returns what was removed.

    A new genome means every asset must be retrained. A stale progress file would
    make the next run print "Nothing to do" and quietly leave a mixed-generation
    model set behind.
    """
    base = base or BASE
    removed = []
    for name in PROGRESS_FILES:
        path = os.path.join(base, name)
        if os.path.exists(path):
            os.remove(path)
            removed.append(name)
    return removed


def _fmt(key, value):
    """Effect sizes with enough digits to read. On an AUC basis the whole floor
    is 0.005 and two decimals would print it as a reassuring 0.01."""
    if key in ("value", "floor", "neural_value") and isinstance(value, (int, float)):
        return f"{value:+.4g}"
    if key == "p" and isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def report_lines(rec, prev=None):
    """The full adoption report: what is live, on what evidence, and the exact
    genome a retrain would run on.

    Returned as lines rather than printed so the unattended loop and `--show`
    render the same thing, and so it can be written beside the run instead of
    living only in console scrollback that a long night will bury.
    """
    from core import adopted as _adopted

    if not rec:
        return ["Nothing adopted: production defaults are in force."]
    out = ["=" * 72,
           "ADOPTED: {}    (on {})".format(rec.get("label"), rec.get("adopted")),
           "=" * 72, "", "EVIDENCE"]
    ev = rec.get("evidence") or {}
    labels = (("kind", "measurement"), ("value", "effect"), ("p", "p-value"),
              ("n", "assets"), ("alpha", "alpha"), ("floor", "floor"),
              ("neural_value", "neural"), ("holdout", "holdout"),
              ("source", "from"), ("bucket", "archive cell"))
    for key, title in labels:
        if ev.get(key) is not None:
            out.append("  %-13s %s" % (title, _fmt(key, ev[key])))
    if ev.get("caveat"):
        out.append("  %-13s %s" % ("caveat", ev["caveat"]))

    genome = rec.get("genome") or {}
    out += ["", "GENOME (genes that leave their default)"]
    changed = {k: v for k, v in sorted(genome.items())
               if k in _adopted._DEFAULTS and v != _adopted._DEFAULTS[k]}
    if changed:
        for k, v in changed.items():
            body = json.dumps(v, ensure_ascii=False)
            out.append("  %-15s %s" % (k, body if len(body) < 90
                                       else body[:87] + "..."))
    else:
        out.append("  none: this genome is the production default")

    out += ["", "TRAINING ENV a retrain will run under"]
    env = _adopted.env_overrides(genome)
    for k, v in sorted(env.items()):
        body = str(v)
        out.append("  %s=%s" % (k, body if len(body) < 90 else body[:87] + "..."))
    if not env:
        out.append("  (none)")

    if prev:
        pev = prev.get("evidence") or {}
        out += ["", "PREVIOUS ADOPTION (kept as adopted_genome.prev.json)",
                "  %s on %s, effect %s"
                % (prev.get("label"), prev.get("adopted"),
                   _fmt("value", pev.get("value")) if pev.get("value") is not None
                   else "n/a")]
    out += ["", "FULL GENOME (copy of what was written to adopted_genome.json)",
            json.dumps(genome, ensure_ascii=False, sort_keys=True)]
    return out


def _adopted_path():
    from core import adopted as _adopted
    return _adopted.PATH


def _previous(path=None):
    """The adoption this one replaced, or None."""
    return _read_json((path or _adopted_path()).replace(".json", ".prev.json"))


def _show():
    from core import adopted as _adopted

    for line in report_lines(_adopted.load(), _previous()):
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--unvalidated", action="store_true",
                    help="also offer search-archive elites")
    ap.add_argument("--auto", action="store_true",
                    help="adopt the best candidate that passed its own A/B "
                         "without asking; for auto_loop.py")
    args = ap.parse_args()

    if args.show:
        _show()
        return
    if args.auto:
        pick = best_validated(candidates())
        if not pick:
            print("Nothing passed its own A/B; adopting nothing.")
            return
        prev = _read_json(_adopted_path())
        print("Adopting: %s" % describe(pick))
        write_adoption(pick)
        # Not optional: a new genome means every asset must be retrained, and a
        # stale progress file would make the next run print "Nothing to do" and
        # leave a mixed-generation model set behind.
        removed = ", ".join(reset_chunk_progress()) or "none"
        print()
        for line in report_lines(_read_json(_adopted_path()), prev):
            print(line)
        print("\nRemoved chunk progress: %s" % removed)
        print("Next (manual): python train_chunked.py, then python predict.py")
        return
    if args.revert:
        if revert():
            print("Reverted. Removed chunk progress: %s"
                  % (", ".join(reset_chunk_progress()) or "none"))
            print("Next: python train_chunked.py")
        else:
            print("Nothing to revert.")
        return

    cands = [c for c in candidates() if c["validated"] or args.unvalidated]
    if not cands:
        print("No validated candidates. Run an A/B first, or pass --unvalidated")
        print("to adopt a search-archive elite on its unvalidated fitness.")
        return
    for i, c in enumerate(cands, 1):
        print("  %d. %s" % (i, describe(c)))
    raw = input("\nAdopt which number (blank to cancel)? ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(cands)):
        print("Cancelled.")
        return
    pick = cands[int(raw) - 1]
    write_adoption(pick)
    print("\nAdopted {}.".format(pick["label"]))
    print("Removed chunk progress: %s"
          % (", ".join(reset_chunk_progress()) or "none"))
    if not pick["validated"]:
        print("WARNING: this is a search fitness, not a measured holdout gain.")
    print("\nNext:")
    print("  python train_chunked.py     # chunked, resumable, champion-challenger")
    print("  python predict.py           # refresh signals")


if __name__ == "__main__":
    main()
