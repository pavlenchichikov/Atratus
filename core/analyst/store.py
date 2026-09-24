"""analyst_log: the judgment table and the backfill that scores it.

Written before anything produces a judgment, on purpose. guru_log holds 636
verdicts and 14 scored outcomes, because filling outcomes lived in a script
somebody had to remember to run. Here the backfill is a loop_cycle step, so a
judgment that is never scored is a broken pipeline rather than a quiet habit.
"""

import datetime
import os
import sqlite3

from core.analyst.payoff import ret_atr
from core.track_record import ohlc_series, volume_series

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
    tool_calls_json TEXT,
    PRIMARY KEY (date, asset, horizon)
)
"""

# A judgment that consulted a source nobody recorded cannot be replayed, and an
# unreplayable judgment is the same defect as an unscored one. ensure_table
# adds the column to a table written before tools existed, because CREATE TABLE
# IF NOT EXISTS does not alter one that is already there.
_ADDED_COLUMNS = (("tool_calls_json", "TEXT"),)

_FIELDS = ["date", "asset", "horizon", "direction", "conviction", "vol_regime",
           "key_risk", "thesis", "evidence_json", "dossier_hash", "llm_model",
           "forecast_pct", "lo_pct", "hi_pct", "atr_at_signal",
           "close_at_signal", "tool_calls_json"]

# forecast_pct/lo_pct/hi_pct are in PAYOFF space (what the POSITION earned;
# see core/analyst/payoff.py and train_payoff.py's SIDE map). A `down`
# judgment's forecast is a short's payoff, positive when the price fell. The
# backfill below must flip the RAW price return through the same side before
# comparing it against those payoff-space numbers, or every non-`up` call
# scores backwards. `flat` stays +1: a flat judgment is a claim about the RAW
# return being small, the same reasoning calibrate.py gives for why the flat
# branch reads the BUY prior.
_SIDE = {"up": 1, "down": -1, "flat": 1}


def _weekdays(asset, loader, days, db_path):
    """`loader`'s last `days` rows, with MOEX weekend sessions dropped.

    MOEX has run weekend sessions since 2025: every Moscow name carries about
    100 of them, at a quarter of the weekday range and a sixteenth of the
    volume. Left in, a Friday one-bar call was scored on Saturday's drift, ATR
    and the volume norm were diluted by quiet bars, and SBER's 60-bar
    correlation to IMOEX (no weekend bars) paired returns off by up to two days.
    Dropped, the weekend move lands in Monday's gap like any overnight move.
    Moscow only, on purpose: crypto trades seven days for real and EGX30's
    Sunday is a full regular session.
    """
    from config import radar_category

    if radar_category(asset) != "ru":
        return loader(asset, days=days, db_path=db_path)
    rows = loader(asset, days=days * 7 // 5 + 10, db_path=db_path)
    return [r for r in rows
            if datetime.date.fromisoformat(r["date"]).weekday() < 5][-days:]


def horizon_bars(asset, horizon):
    """Bars a horizon of `horizon` trading days spans for this asset.

    The unit is the exchange trading day, 5 a week and 20 a month, the same for
    SBER as for NVDA once weekend sessions are dropped. Crypto has a bar every
    calendar day, so the same span is 7/5 as many bars: 5 -> 7, 20 -> 28.
    """
    from config import ASSET_TYPES

    h = int(horizon or 1)
    return round(h * 7 / 5) if asset in ASSET_TYPES["CRYPTO"] else h


def bars(asset, days, db_path=None):
    return _weekdays(asset, ohlc_series, days, db_path)


def volumes(asset, days, db_path=None):
    return _weekdays(asset, volume_series, days, db_path)


def _connect(db_path=None):
    return sqlite3.connect(db_path or DB_PATH)


def _migrate(con):
    """Add columns the DDL grew after this table was first written.

    CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a column
    added later never appears in a database that predates it and every write
    fails on an unknown column. Adding it here keeps that from being an
    operator's problem.
    """
    have = {r[1] for r in con.execute("PRAGMA table_info(analyst_log)")}
    for name, decl in _ADDED_COLUMNS:
        if name not in have:
            con.execute("ALTER TABLE analyst_log ADD COLUMN %s %s" % (name, decl))


def ensure_table(db_path=None):
    with _connect(db_path) as con:
        con.execute(DDL)
        _migrate(con)


def write_judgment(row, db_path=None):
    """Insert one judgment. Outcome columns stay NULL until the backfill runs.

    REPLACE rather than INSERT: rerunning a day re-states that day's judgment
    instead of failing, and the primary key keeps one row per asset per day.
    """
    values = [row.get(f) for f in _FIELDS]
    placeholders = ",".join("?" * len(_FIELDS))
    with _connect(db_path) as con:
        con.execute(DDL)
        _migrate(con)
        con.execute(
            f'INSERT OR REPLACE INTO analyst_log ({",".join(_FIELDS)}) '
            f'VALUES ({placeholders})', values)


def judged_with_hash(asset, dossier_hash, db_path=None, horizon=None):
    """Whether this exact dossier was already judged. The LLM cache key.

    `horizon` scopes it, because the same dossier asked over one day and over
    five is two different questions with two different answers - and the table's
    own primary key has said so since it was written, (date, asset, horizon).
    Left None the check spans every horizon, which is the old behaviour.
    """
    sql = "SELECT 1 FROM analyst_log WHERE asset=? AND dossier_hash=?"
    args = [asset, dossier_hash]
    if horizon is not None:
        sql += " AND horizon=?"
        args.append(int(horizon))
    with _connect(db_path) as con:
        con.execute(DDL)
        return con.execute(sql + " LIMIT 1", args).fetchone() is not None


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


def latest_judgment(asset, db_path=None):
    """The most recent judgment for one asset, scored or not, or None.

    The asset card wants the opinion itself, not only the number derived from
    it, and it wants it the day it is made rather than after the horizon has
    elapsed - so this deliberately does not filter on realized_ret.
    """
    with _connect(db_path) as con:
        con.execute(DDL)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM analyst_log WHERE asset=? ORDER BY date DESC LIMIT 1",
            (asset,)).fetchone()
        return dict(row) if row else None


def latest_judgments(asset, db_path=None):
    """The newest judgment for EACH horizon this asset has, shortest first.

    A run judges several horizons (1 and 20 trading days by default) and the
    card must show every one, labelled, not whichever was written last."""
    with _connect(db_path) as con:
        con.execute(DDL)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM (SELECT *, ROW_NUMBER() OVER ("
            " PARTITION BY horizon ORDER BY date DESC, rowid DESC) AS _rn"
            " FROM analyst_log WHERE asset=?) WHERE _rn = 1 ORDER BY horizon",
            (asset,)).fetchall()
        out = [dict(r) for r in rows]
        for r in out:
            r.pop("_rn", None)
        return out


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
                bars_by_asset[asset] = _weekdays(asset, ohlc_series, 6000,
                                                    db_path)
            bars = bars_by_asset[asset]
            idx = next((i for i, b in enumerate(bars) if b["date"] == date), None)
            if idx is None:
                continue
            target = idx + horizon_bars(asset, horizon)
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
