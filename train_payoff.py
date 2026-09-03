"""Fit the empirical payoff table from prediction_log.

Two jobs in one artifact. It is the baseline the analyst agent has to beat, and
it is the prior every calibration cell starts from before the agent has logged
anything of its own.

NOT the raw historical average for a direction. `mean` counts only bars the
ENSEMBLE signalled on, so every cell is the average payoff conditional on the
ensemble having called that direction, not the asset's or asset class's
unconditional payoff for it. A reader who treats this as "what a BUY has
historically been worth on this asset" is over-trusting it; it is "what a BUY
has been worth on the bars the ensemble picked."

And a conditional payoff still contains the market. On the RU class over
2026-06-12..2026-09-02 the raw BUY mean was -0.116 ATR, which reads as a
verdict on the ensemble's long calls until the same window is measured
unsignalled: -0.101 of it was the drift, leaving -0.016 that the calls
themselves are responsible for. The class was not losing because of the calls,
it was losing because it was long a falling market - and every RU long the
analyst produced inherited the whole -0.116 as its prior, with a
"[the payoff table disagrees]" flag stamped on it.

So each cell also carries `drift` - what a position on that side earned over
the SAME rows for merely being open, signalled or not - and `excess`, the
difference. `mean` stays raw because that is what the file has always promised
and what the docstring above describes; `excess` is the number a decision
should use, and core/analyst/calibrate.py reads it.

Everything is stored in ATR units. A payoff measured on SBER in 2024 is not
comparable to one measured today in percent, because both the price and the
volatility have moved; in ATR units it is.

WRITES payoff_stats.json ON EVERY RUN. Never run this in the main checkout: a
ten-asset smoke run overwrites the full-universe artifact, which is exactly how
the 207-asset sizing evidence was destroyed once already.
"""

import datetime as _dt
import json
import os
import sqlite3
import statistics

from config import radar_category
from core.analyst.payoff import ret_atr
from core.levels import atr_series
from core.track_record import ohlc_series

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")
STATS_PATH = os.path.join(BASE_DIR, "payoff_stats.json")

SIDE = {"BUY": 1, "SELL": -1}
MIN_CELL = 5   # below this an asset cell is not written at all; the reader
               # falls back to the class prior rather than to a thin mean.
MIN_DRIFT = 30
# Below this the window is too short to estimate a drift from, and no
# adjustment is made rather than a noisy one: subtracting a five-bar average
# would move a cell by more than the effect being measured.


def _scale_by_date(asset, db_path):
    """{date: (atr, close)} for one asset, computed once over its full history."""
    bars = ohlc_series(asset, days=6000, db_path=db_path)
    atrs = atr_series(bars)
    return {b["date"]: (a, b["close"]) for b, a in zip(bars, atrs)}


def _cell(values, drift=None, drift_n=0, side=1):
    """One side's payoff distribution, and what the market gave it for free.

    `drift` arrives unsigned (the raw ATR-unit move of the thing) and is signed
    here, because the same falling market is a cost to a long and a credit to a
    short. A cell with too little of a window to estimate it carries no `drift`
    key at all, which is how the reader tells "no adjustment was possible" from
    "the adjustment was zero".
    """
    values = sorted(values)
    n = len(values)
    cell = {"n": n,
            "mean": statistics.fmean(values),
            "q10": values[max(0, n // 10 - 1)],
            "q90": values[min(n - 1, (9 * n) // 10)]}
    if drift is not None:
        cell["drift"] = side * drift
        cell["drift_n"] = drift_n
        cell["excess"] = cell["mean"] - cell["drift"]
    return cell


def build_table(db_path=None):
    """The payoff table, in ATR units, by asset and by asset class."""
    path = db_path or DB_PATH
    with sqlite3.connect(path) as con:
        # WAIT rows come back too, and only for the drift: the benchmark a
        # payoff has to beat is holding the thing over the whole window, not
        # holding it on the days the ensemble happened to speak.
        rows = con.execute(
            "SELECT date, asset, signal, actual_next_ret FROM prediction_log "
            "WHERE actual_next_ret IS NOT NULL "
            "ORDER BY asset, date").fetchall()

    by_asset, by_class, scales = {}, {}, {}
    drift_asset, drift_class = {}, {}
    for date, asset, signal, ret in rows:
        if asset not in scales:
            scales[asset] = _scale_by_date(asset, path)
        atr, close = scales[asset].get(str(date)[:10], (None, None))
        move = ret_atr(ret, atr, close, side=1)
        if move is None:
            continue
        cls = radar_category(asset)
        drift_asset.setdefault(asset, []).append(move)
        drift_class.setdefault(cls, []).append(move)
        if signal not in SIDE:
            continue
        value = SIDE[signal] * move
        by_asset.setdefault(asset, {}).setdefault(signal, []).append(value)
        by_class.setdefault(cls, {}).setdefault(signal, []).append(value)

    def _pack(groups, drifts):
        out = {}
        for key, sides in groups.items():
            pool = drifts.get(key, [])
            drift = statistics.fmean(pool) if len(pool) >= MIN_DRIFT else None
            cells = {s: _cell(v, drift, len(pool), SIDE[s])
                     for s, v in sides.items() if len(v) >= MIN_CELL}
            if cells:
                out[key] = cells
        return out

    return {"asset": _pack(by_asset, drift_asset),
            "class": _pack(by_class, drift_class),
            "built": _dt.datetime.utcnow().isoformat(timespec="seconds")}


def main():
    table = build_table()
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    n_assets = len(table["asset"])
    n_rows = sum(c["n"] for sides in table["asset"].values()
                 for c in sides.values())
    print(f"[payoff] {n_assets} assets, {n_rows} scored signals -> "
          f"{os.path.basename(STATS_PATH)}")
    # The class drift, printed because it is the half of every prior that has
    # nothing to do with the model. A big one means the WINDOW, not the
    # ensemble, is what any negative cell is mostly describing.
    for cls, sides in sorted(table["class"].items()):
        cell = next(iter(sides.values()))
        if "drift" not in cell:
            print(f"[payoff] {cls:<10} drift not measured, under {MIN_DRIFT} bars")
            continue
        # _cell signed the drift by side, so the BUY cell already holds the
        # long's version of it and the SELL cell holds its negation.
        long_drift = sides["BUY"]["drift"] if "BUY" in sides else -cell["drift"]
        detail = "  ".join(
            f"{s_}: raw {c['mean']:+.3f} excess {c['excess']:+.3f}"
            for s_, c in sorted(sides.items()))
        print(f"[payoff] {cls:<10} drift {long_drift:+.3f} ATR "
              f"over {cell['drift_n']} bars   {detail}")



if __name__ == "__main__":
    main()
