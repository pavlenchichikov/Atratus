"""Fit the GLOBAL live calibration layer from verified prediction outcomes.

The 2026-06-12..07-16 live stream was anti-calibrated at the extremes (the
0.9-1.0 probability bucket scored 32% accuracy), so serve-time probabilities
get a second isotonic layer fitted on what actually happened. prediction_log
stores the PRE-layer (raw) probability, so every refit trains raw -> P(up)
on a homogeneous history and REPLACES models/live_calib_global.pkl.

Run weekly:  python recalibrate_live.py [--days 90] [--min-n 300]
Rollback: delete models/live_calib_global.pkl (scoring degrades to identity).
"""

import argparse
import os
import sqlite3
from datetime import datetime

import numpy as np

from core.calibration import (
    PlattCalibrator,
    fit_calibrator,
    save_live_global,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
MODEL_DIR = os.path.join(BASE, "models")


def collect_pairs(days=90, db_path=None, with_dates=False):
    """(probs, went_up) from verified BUY/SELL rows. The stored probability is
    the ensemble P(up), so went_up = correct for BUY and NOT correct for SELL.

    `with_dates` adds the date of each row, which is what lets the fit be
    split in time instead of shuffled: a calibration map chosen on rows drawn
    from the same days it was fitted on has seen its own answer.
    """
    con = sqlite3.connect(db_path or DB_PATH)
    try:
        rows = con.execute(
            "SELECT signal, probability, correct, date FROM prediction_log "
            "WHERE correct IS NOT NULL AND signal IN ('BUY','SELL') "
            "AND probability IS NOT NULL AND date >= date('now', ?)",
            ("-%d days" % days,),
        ).fetchall()
    finally:
        con.close()
    probs, ups, dates = [], [], []
    for sig, p, c, d in rows:
        probs.append(p)
        ups.append(int(c) if sig == "BUY" else 1 - int(c))
        dates.append(d)
    if with_dates:
        return probs, ups, dates
    return probs, ups


def split_in_time(probs, ups, dates, frac=0.6):
    """(fit, held) split on the DATE, not on the row order.

    Rows from one day are correlated across assets, so a random split would put
    the same day on both sides and the held-out reading would flatter every
    candidate equally.
    """
    days = sorted(set(dates))
    if len(days) < 4:
        return None
    cut = days[int(len(days) * frac)]
    fit = [(p, u) for p, u, d in zip(probs, ups, dates) if d < cut]
    held = [(p, u) for p, u, d in zip(probs, ups, dates) if d >= cut]
    if len(fit) < MIN_SAMPLES_SPLIT or len(held) < MIN_SAMPLES_SPLIT:
        return None
    return fit, held, cut


# The guard is not about spread in the abstract. What matters is whether the
# calibrated output can still reach the thresholds serve decides on: a layer
# that maps every asset into a narrow band around the base rate never crosses
# them, the whole book prints WAIT, and that is a kill switch wearing the word
# calibration. Measured 2026-08-21: isotonic collapsed to a literal constant
# and Platt to a band 0.055 wide, which the old spread test of 0.02 let past.
TYPICAL_BUY_THR, TYPICAL_SELL_THR = 0.55, 0.45
MIN_CROSSING = 0.05   # share of rows that must still reach a threshold
MIN_SPREAD = 0.02   # calibrated-probability range a fit must still produce
MIN_SAMPLES_SPLIT = 200   # per side of the time split, below which it says so
# A candidate has to beat leaving the probabilities alone by this much in log
# loss on days it never saw. Small, but not zero: a layer that is level with
# identity is a layer nobody should be maintaining.
MIN_GAIN = 0.002


class _Identity:
    """Leaving the probabilities alone, as something with a .predict."""

    def predict(self, probs):
        import numpy as _np
        return _np.asarray(probs, dtype=float)


def crossing_share(model, probs):
    """Share of these probabilities that still reach a decision threshold.

    Run on the FITTED window's own inputs: if a map cannot make its own
    training data cross, it will not make anything cross.
    """
    import numpy as _np
    out = model.predict(_np.asarray(probs, dtype=float))
    reach = (out >= TYPICAL_BUY_THR) | (out <= TYPICAL_SELL_THR)
    return float(_np.mean(reach)) if len(out) else 0.0


def _fit_spread(iso, probs):
    """How much of a range the fitted map still produces over the observed
    probabilities. Zero means every input became the same output."""
    lo, hi = min(probs), max(probs)
    if hi <= lo:
        return 0.0
    grid = [lo + (hi - lo) * i / 20.0 for i in range(21)]
    out = iso.predict(grid)
    return float(max(out) - min(out))


def choose(fit_rows, held_rows):
    """Pick between leaving the probabilities alone, isotonic, and Platt.

    Every candidate is FITTED on the earlier days and SCORED on the later ones,
    on log loss, which is a proper scoring rule: unlike accuracy it cannot be
    improved by collapsing toward the base rate, which is exactly the failure
    the isotonic guard exists to catch.

    Honest about its own limit: the winner is selected on the same held-out
    days it is reported on, so the gain quoted is an upper bound. With 44
    trading days of live history there is not enough to hold a third window
    back, and saying so beats pretending.
    """
    from core.calibration import fit_platt, log_loss
    fp = [r[0] for r in fit_rows]
    fu = [r[1] for r in fit_rows]
    hp = [r[0] for r in held_rows]
    hu = [r[1] for r in held_rows]

    rows = [{"name": "identity", "model": None,
             "loss": log_loss(hp, hu), "spread": None}]
    for name, model in (("isotonic", fit_calibrator(fp, fu)),
                        ("platt", fit_platt(fp, fu))):
        if model is None:
            rows.append({"name": name, "model": None, "loss": None,
                         "spread": None, "why": "did not fit"})
            continue
        mapped = model.predict(np.asarray(hp, dtype=float))
        spread = _fit_spread(model, fp)
        cross = crossing_share(model, fp)
        row = {"name": name, "model": model, "spread": spread,
               "crossing": cross, "loss": log_loss(mapped, hu)}
        if spread < MIN_SPREAD:
            row["why"] = ("collapsed to a constant (spread %.4f): the stream is "
                          "anti-calibrated and this family cannot say so"
                          % spread)
            row["model"] = None
        elif cross < MIN_CROSSING:
            row["why"] = ("only %.1f%% of rows would still reach a threshold "
                          "(spread %.3f): installing this silences the book"
                          % (cross * 100.0, spread))
            row["model"] = None
        rows.append(row)
    base = rows[0]["loss"]
    live = [r for r in rows
            if r["model"] is not None and r["loss"] is not None]
    best = min(live, key=lambda r: r["loss"]) if live else None
    if best is not None and base - best["loss"] < MIN_GAIN:
        best = None
    return rows, best, base


def main(days=90, min_n=None, model_dir=None, db_path=None, install=False):
    if min_n is None:
        try:
            min_n = int(os.getenv("GTRADE_LIVE_RECAL_MIN_N") or "300")
        except ValueError:
            min_n = 300
    probs, ups, dates = collect_pairs(days, db_path, with_dates=True)
    if len(probs) < min_n:
        print("[recalibrate-live] only %d verified rows (< %d) - not fitting"
              % (len(probs), min_n))
        return None
    split = split_in_time(probs, ups, dates)
    if split is None:
        print("[recalibrate-live] not enough distinct days on both sides of a "
              "time split - nothing written. A layer fitted and judged on the "
              "same days would install itself on its own answer.")
        return None
    fit_rows, held_rows, cut = split
    rows, best, base = choose(fit_rows, held_rows)
    print("[recalibrate-live] fit on days before %s (%d rows), judged on %s "
          "onward (%d rows)" % (cut, len(fit_rows), cut, len(held_rows)))
    for r in rows:
        if r["loss"] is None:
            print("  %-9s %s" % (r["name"], r.get("why", "no reading")))
            continue
        mark = "  <- chosen" if best is not None and r is best else ""
        cross = ("" if r.get("crossing") is None
                 else "  crosses %.1f%%" % (r["crossing"] * 100.0))
        print("  %-9s held-out log loss %.5f%s%s%s"
              % (r["name"], r["loss"], cross,
                 "  (%s)" % r["why"] if r.get("why") else "", mark))
    if best is None:
        print("[recalibrate-live] nothing beats leaving the probabilities "
              "alone by %.3f on days it never saw. Nothing written."
              % MIN_GAIN)
        return None
    iso = best["model"]
    if isinstance(iso, PlattCalibrator) and iso.a < 0:
        print("[recalibrate-live] the fitted slope is NEGATIVE (a=%.3f): on "
              "this history the model's confidence points the wrong way, and "
              "the layer now says so instead of failing to fit." % iso.a)
    # Isotonic can only fit a NON-DECREASING map. When the live stream is
    # anti-calibrated (high probabilities scoring worse than low ones, which is
    # exactly why this layer exists) the best non-decreasing fit of that data is
    # a flat line at the base rate. It is a valid IsotonicRegression, so it
    # passes every check above and then maps every asset to one number: the
    # serve layer sees the same probability everywhere, no threshold is ever
    # crossed, and the whole book prints WAIT. Refuse to ship that.
    fp = [r[0] for r in fit_rows]
    raw_cross = crossing_share(_Identity(), fp)
    print("[recalibrate-live] %s fitted on %d outcomes, %.5f better than "
          "identity on held-out days."
          % (best["name"], len(fit_rows), base - best["loss"]))
    print("[recalibrate-live] it would change how much the system trades: "
          "%.1f%% of rows reach a threshold today, %.1f%% would after the "
          "layer, about %.0fx less."
          % (raw_cross * 100.0, best["crossing"] * 100.0,
             raw_cross / max(best["crossing"], 1e-9)))
    if not install:
        print("[recalibrate-live] NOTHING WRITTEN. This is a report by default: "
              "a layer chosen on 44 days of live history, on the same held-out "
              "window it is scored on, that cuts trading by an order of "
              "magnitude, is a deliberate act. Rerun with --install to take it.")
        return None
    path = save_live_global(
        iso, {"n": len(fit_rows), "fitted_at": datetime.utcnow().isoformat(),
              "days": days, "family": best["name"],
              "raw_crossing": raw_cross, "crossing": best["crossing"],
              "held_out_gain": base - best["loss"]}, model_dir or MODEL_DIR)
    print("[recalibrate-live] installed -> %s" % path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit the global live calibration layer")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--min-n", type=int, default=None)
    ap.add_argument("--install", action="store_true",
                    help="write the winning layer. Without it this only "
                         "measures and reports, which is the default because "
                         "the layer can change how much the system trades by "
                         "an order of magnitude.")
    args = ap.parse_args()
    main(days=args.days, min_n=args.min_n, install=args.install)
