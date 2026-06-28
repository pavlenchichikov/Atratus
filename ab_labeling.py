"""A/B the labeling mode (direction vs rel_median) on the research subset.

Tracked in git (the script itself), but its output log (_ab_labeling_log.json)
is gitignored via the repo-wide "*.json" rule - this is a runtime artifact, not
config. Trains the subset twice via train_hybrid (the same LIGHT_ENV profile
auto_research uses) and reports the paired Score delta on the selection set
plus the canonical held-out adoption verdict. It never retrains production and
writes only a local log.

Run:  python ab_labeling.py
"""
import json
import os

import auto_research as ar

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ab_labeling_log.json")


def evaluate(base_rows, var_rows, selection, heldout):
    """Pure comparison: paired selection-set delta + held-out adopt verdict.

    A missing expected asset (trainer skipped it - insufficient history, data
    gap, per-asset crash) shrinks the base/variant intersection in a way that
    is otherwise indistinguishable from "no improvement". Surface that
    degraded-set case explicitly as verdict=INCONCLUSIVE with the missing
    asset names, rather than letting it silently compute as a false HOLD.
    """
    base_assets = {r["Asset"] for r in base_rows}
    var_assets = {r["Asset"] for r in var_rows}
    expected = set(selection) | set(heldout)
    missing = sorted(
        (expected - base_assets) | (expected - var_assets)
    )

    base_sel = {r["Asset"]: r.get("Score", 0.0) for r in base_rows if r["Asset"] in selection}
    sel_mean, _ = ar._mean_delta([r for r in var_rows if r["Asset"] in selection], base_sel)

    base_held = [r for r in base_rows if r["Asset"] in heldout]
    var_held = [r for r in var_rows if r["Asset"] in heldout]
    held_score = {r["Asset"]: r.get("Score", 0.0) for r in base_held}
    held_mean, _ = ar._mean_delta(var_held, held_score)
    ok, why = ar.is_adoptable(base_held, var_held, 1, 1)

    if missing:
        return {
            "selection_mean_delta": sel_mean,
            "heldout_mean_delta": held_mean,
            "verdict": "INCONCLUSIVE",
            "why": why,
            "missing": missing,
        }

    return {
        "selection_mean_delta": sel_mean,
        "heldout_mean_delta": held_mean,
        "verdict": "ADOPT" if ok else "HOLD",
        "why": why,
    }


def _rows_for(mode, subset):
    """Train one labeling mode's rows. Restores the prior GTRADE_LABEL_MODE
    afterward so this function is pure from the caller's perspective and a
    future in-process caller never inherits a leaked mode. Raises RuntimeError
    if the subprocess training produced no rows (e.g. crashed or wrote no
    quality_report.json), so a subprocess failure is surfaced loudly instead
    of silently flowing into evaluate() as a false HOLD."""
    prior = os.environ.get("GTRADE_LABEL_MODE")
    os.environ["GTRADE_LABEL_MODE"] = mode
    try:
        rows = ar.train_rows(subset, [], [])
    finally:
        if prior is None:
            os.environ.pop("GTRADE_LABEL_MODE", None)
        else:
            os.environ["GTRADE_LABEL_MODE"] = prior
    if not rows:
        raise RuntimeError(
            "training mode=%r on subset=%r produced no rows "
            "(subprocess failure or missing quality_report.json)" % (mode, subset)
        )
    return rows


def main():
    selection = ar.SELECTION_ASSETS.split(",")
    heldout = ar.HELDOUT_ASSETS.split(",")
    subset = ",".join(selection + heldout)

    base_rows = _rows_for("direction", subset)
    var_rows = _rows_for("rel_median", subset)

    result = evaluate(base_rows, var_rows, selection, heldout)
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print("[ab-labeling] %s | %s" % (result["verdict"], result["why"]))


if __name__ == "__main__":
    main()
