"""analyst_log: the judgment table and the backfill that scores it.

Written before anything produces a judgment, on purpose. guru_log holds 636
verdicts and 14 scored outcomes, because filling outcomes lived in a script
somebody had to remember to run. Here the backfill is a loop_cycle step, so a
judgment that is never scored is a broken pipeline rather than a quiet habit.
"""

import os
import sqlite3

from core.analyst.payoff import ret_atr
from core.track_record import ohlc_series

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "market.db")

DDL = """
CREATE TABLE IF NOT EXISTS analyst_log (
    date TEXT, asset TEXT, horizon INTEGER,
    direction TEXT, conviction INTEGER, vol_regime TEXT,
    key_risk TEXT, thesis TEXT, evidence_json TEXT,
    dossier_hash TEXT, llm_model TEXT,
    forecast_pct REAL, lo_pct REAL, hi_pct REAL,
    atr_at_signal REAL, close_at_signal REAL,
    realized_ret REAL, realized_atr_units REAL,
    inside_interval INTEGER, abs_err_atr REAL,
    PRIMARY KEY (date, asset, horizon)
)
"""

_FIELDS = ["date", "asset", "horizon", "direction", "conviction", "vol_regime",
           "key_risk", "thesis", "evidence_json", "dossier_hash", "llm_model",
           "forecast_pct", "lo_pct", "hi_pct", "atr_at_signal",
           "close_at_signal"]

# forecast_pct/lo_pct/hi_pct are in PAYOFF space (what the POSITION earned;
# see core/analyst/payoff.py and train_payoff.py's SIDE map). A `down`
# judgment's forecast is a short's payoff, positive when the price fell. The
# backfill below must flip the RAW price return through the same side before
# comparing it against those payoff-space numbers, or every non-`up` call
# scores backwards. `flat` stays +1: a flat judgment is a claim about the RAW
# return being small, the same reasoning calibrate.py gives for why the flat
# branch reads the BUY prior.
_SIDE = {"up": 1, "down": -1, "flat": 1}


def _connect(db_path=None):
    return sqlite3.connect(db_path or DB_PATH)


def ensure_table(db_path=None):
    with _connect(db_path) as con:
        con.execute(DDL)


def write_judgment(row, db_path=None):
    """Insert one judgment. Outcome columns stay NULL until the backfill runs.

    REPLACE rather than INSERT: rerunning a day re-states that day's judgment
    instead of failing, and the primary key keeps one row per asset per day.
    """
    values = [row.get(f) for f in _FIELDS]
    placeholders = ",".join("?" * len(_FIELDS))
    with _connect(db_path) as con:
        con.execute(DDL)
        con.execute(
            f'INSERT OR REPLACE INTO analyst_log ({",".join(_FIELDS)}) '
            f'VALUES ({placeholders})', values)


def judged_with_hash(asset, dossier_hash, db_path=None):
    """Whether this exact dossier was already judged. The LLM cache key."""
    with _connect(db_path) as con:
        con.execute(DDL)
        row = con.execute(
            "SELECT 1 FROM analyst_log WHERE asset=? AND dossier_hash=? LIMIT 1",
            (asset, dossier_hash)).fetchone()
        return row is not None


def pending_count(db_path=None):
    with _connect(db_path) as con:
        con.execute(DDL)
        return con.execute(
            "SELECT COUNT(*) FROM analyst_log WHERE realized_ret IS NULL"
        ).fetchone()[0]


def scored_rows(db_path=None):
    with _connect(db_path) as con:
        con.execute(DDL)
        con.row_factory = sqlite3.Row
        cur = con.execute("SELECT * FROM analyst_log "
                          "WHERE realized_ret IS NOT NULL ORDER BY date")
        return [dict(r) for r in cur.fetchall()]


def backfill_outcomes(db_path=None, today=None):
    """Score every judgment whose horizon has now elapsed. Returns rows filled.

    Bars are loaded once per asset, not once per row: 208 assets against a log
    that grows daily, and the naive form reloads the same series hundreds of
    times.

    Two number spaces meet here. `realized_ret` and the bars underneath it are
    RAW-RETURN space (the price's own move; a falling price is negative).
    `forecast_pct`/`lo_pct`/`hi_pct` are PAYOFF space (what the POSITION
    earned; a profitable short is positive). `realized_atr_units` and
    `inside_interval` are compared against those payoff-space numbers, so both
    must be turned through the judgment's side (_SIDE) before comparing -
    `realized_ret` itself stays RAW, on purpose, because it is the price's own
    move and other readers may want it in that form.
    """
    with _connect(db_path) as con:
        con.execute(DDL)
        pending = con.execute(
            "SELECT date, asset, horizon, direction, forecast_pct, lo_pct, "
            "hi_pct, atr_at_signal, close_at_signal FROM analyst_log "
            "WHERE realized_ret IS NULL ORDER BY asset, date").fetchall()

        filled = 0
        bars_by_asset = {}
        for (date, asset, horizon, direction, fc, lo, hi, atr_sig, close_sig) in pending:
            if asset not in bars_by_asset:
                bars_by_asset[asset] = ohlc_series(asset, days=6000,
                                                   db_path=db_path)
            bars = bars_by_asset[asset]
            idx = next((i for i, b in enumerate(bars) if b["date"] == date), None)
            if idx is None:
                continue
            target = idx + int(horizon or 1)
            if target >= len(bars):
                continue          # the horizon has not elapsed yet
            if today is not None and bars[target]["date"] > today:
                continue

            start, end = bars[idx]["close"], bars[target]["close"]
            if not start:
                continue
            realized = (end - start) / start
            side = _SIDE.get(direction, 1)
            realized_atr = ret_atr(realized, atr_sig, close_sig, side=side)
            if realized_atr is None:
                continue

            fc_atr = ret_atr(fc, atr_sig, close_sig) if fc is not None else None
            err = None if fc_atr is None else abs(fc_atr - realized_atr)
            inside = None
            if lo is not None and hi is not None:
                inside = 1 if lo <= side * realized <= hi else 0

            con.execute(
                "UPDATE analyst_log SET realized_ret=?, realized_atr_units=?, "
                "inside_interval=?, abs_err_atr=? "
                "WHERE date=? AND asset=? AND horizon=?",
                (realized, realized_atr, inside, err, date, asset, horizon))
            filled += 1
        return filled
