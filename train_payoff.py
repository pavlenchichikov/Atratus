"""Fit the empirical payoff table from prediction_log.

Two jobs in one artifact. It is the baseline the analyst agent has to beat, and
it is the prior every calibration cell starts from before the agent has logged
anything of its own.

NOT the raw historical average for a direction. build_table() reads only
`WHERE signal IN ('BUY','SELL')` - bars the ENSEMBLE chose to signal on - so
every cell here is the average payoff conditional on the ensemble having
called that direction, not the asset's or asset class's unconditional payoff
for it. A reader who treats this as "what a BUY has historically been worth
on this asset" is over-trusting it; it is "what a BUY has been worth on the
bars the ensemble picked."

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


def _scale_by_date(asset, db_path):
    """{date: (atr, close)} for one asset, computed once over its full history."""
    bars = ohlc_series(asset, days=6000, db_path=db_path)
    atrs = atr_series(bars)
    return {b["date"]: (a, b["close"]) for b, a in zip(bars, atrs)}


def _cell(values):
    values = sorted(values)
    n = len(values)
    return {"n": n,
            "mean": statistics.fmean(values),
            "q10": values[max(0, n // 10 - 1)],
            "q90": values[min(n - 1, (9 * n) // 10)]}


def build_table(db_path=None):
    """The payoff table, in ATR units, by asset and by asset class."""
    path = db_path or DB_PATH
    with sqlite3.connect(path) as con:
        rows = con.execute(
            "SELECT date, asset, signal, actual_next_ret FROM prediction_log "
            "WHERE actual_next_ret IS NOT NULL AND signal IN ('BUY','SELL') "
            "ORDER BY asset, date").fetchall()

    by_asset, by_class, scales = {}, {}, {}
    for date, asset, signal, ret in rows:
        if asset not in scales:
            scales[asset] = _scale_by_date(asset, path)
        atr, close = scales[asset].get(str(date)[:10], (None, None))
        value = ret_atr(ret, atr, close, side=SIDE[signal])
        if value is None:
            continue
        by_asset.setdefault(asset, {}).setdefault(signal, []).append(value)
        by_class.setdefault(radar_category(asset), {}).setdefault(
            signal, []).append(value)

    def _pack(groups):
        out = {}
        for key, sides in groups.items():
            cells = {s: _cell(v) for s, v in sides.items() if len(v) >= MIN_CELL}
            if cells:
                out[key] = cells
        return out

    return {"asset": _pack(by_asset), "class": _pack(by_class),
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


if __name__ == "__main__":
    main()
