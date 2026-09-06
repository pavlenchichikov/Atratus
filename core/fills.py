"""What the ACCOUNT actually holds, as opposed to what the signal history implies.

Orders are placed by hand at the broker, so until now nothing in this repository
knew a real entry existed. `core/levels.py` derives the trailing stop by scanning
bars from a segment's `start_date`, and that segment came from
`core/positions.build_positions`, which reconstructs positions from the SIGNAL
history: the bar the signal first turned, at the close of that bar. A hand-placed
order fills on a different day at a different price, so every trailing stop on
the sheet was measured from an entry nobody took.

This module stores the fills and hands `levels_sheet` a segment shaped exactly
like the reconstructed one, so the sheet needs no second code path and the stop
is measured from the bar you actually bought on.

Deliberately NOT a position manager: no P&L, no sizing, no reconciliation
against risk_manager's book. It answers one question, "what is open and since
when", because that is the one the levels need.
"""
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "market.db")

SIDES = ("BUY", "SELL")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    asset      TEXT NOT NULL,
    side       TEXT NOT NULL,
    qty        REAL NOT NULL,
    price      REAL NOT NULL,
    entry_date TEXT NOT NULL,
    exit_date  TEXT,
    exit_price REAL,
    note       TEXT
)
"""
# One partial index rather than a UNIQUE column: an asset may be traded many
# times, but only one of those rows may be open at a time. Enforced by the
# database because the alternative is every caller remembering to check.
OPEN_INDEX = ("CREATE UNIQUE INDEX IF NOT EXISTS fills_one_open_per_asset "
              "ON fills(asset) WHERE exit_date IS NULL")


def _connect(db_path=None):
    """For WRITES: creates the table if this is the first fill ever recorded."""
    con = sqlite3.connect(db_path or DB_PATH)
    con.execute(SCHEMA)
    con.execute(OPEN_INDEX)
    return con


def _read_connect(db_path=None):
    """For READS: None when there is no database or no journal yet.

    Deliberately not _connect. Every levels sheet asks this module whether a
    fill exists, and a reader that runs CREATE TABLE brings a market.db into
    existence wherever it is called from - which is how the test suite grew a
    28 KB stub database in a checkout that has none.
    """
    path = db_path or DB_PATH
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(path)
    if not con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                       "AND name='fills'").fetchone():
        con.close()
        return None
    return con


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def record(asset, side, qty, price, entry_date=None, note=None, db_path=None):
    """Store one opening fill. Raises when that asset already has one open."""
    side = (side or "").strip().upper()
    if side not in SIDES:
        raise ValueError("side must be BUY or SELL, got %r" % (side,))
    if float(qty) <= 0 or float(price) <= 0:
        raise ValueError("qty and price must be positive")
    con = _connect(db_path)
    try:
        with con:
            cur = con.execute(
                "INSERT INTO fills (asset, side, qty, price, entry_date, note) "
                "VALUES (?,?,?,?,?,?)",
                (asset, side, float(qty), float(price),
                 entry_date or _today(), note))
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("%s already has an open fill; close it first" % asset)
    finally:
        con.close()


def close(asset, price, exit_date=None, db_path=None):
    """Close the open fill on `asset`. Returns the row id, or None if none open."""
    con = _connect(db_path)
    try:
        with con:
            cur = con.execute(
                "UPDATE fills SET exit_date = ?, exit_price = ? "
                "WHERE asset = ? AND exit_date IS NULL",
                (exit_date or _today(), float(price), asset))
        if not cur.rowcount:
            return None
        row = con.execute("SELECT id FROM fills WHERE asset = ? "
                          "ORDER BY id DESC LIMIT 1", (asset,)).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def open_fills(db_path=None):
    """{asset: row} for everything currently held, as plain dicts."""
    con = _read_connect(db_path)
    if con is None:
        return {}
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM fills WHERE exit_date IS NULL ORDER BY asset"
        ).fetchall()
        return {r["asset"]: dict(r) for r in rows}
    finally:
        con.close()


def history(limit=50, db_path=None):
    """Closed fills, newest first."""
    con = _read_connect(db_path)
    if con is None:
        return []
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM fills WHERE exit_date IS NOT NULL "
            "ORDER BY exit_date DESC, id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def segment_of(fill, today=None):
    """A fill in the shape core/positions.build_positions returns a segment.

    `bars` is a day count and not a bar count, which is the one honest
    difference: this module knows the calendar date you filled on and not which
    of the asset's bars that was. It feeds the sheet's "held" column, while the
    stop reads start_date, so the approximation stays out of the price.
    """
    if not fill:
        return None
    start = fill["entry_date"]
    try:
        days = (datetime.strptime(today or _today(), "%Y-%m-%d")
                - datetime.strptime(start, "%Y-%m-%d")).days
    except (TypeError, ValueError):
        days = 0
    return {"side": 1 if fill["side"] == "BUY" else -1,
            "start_date": start, "end_date": None,
            "bars": max(0, days), "ret": None, "open": True,
            "entry_price": fill["price"], "source": "fill"}


def open_segment(asset, today=None, db_path=None):
    """The open segment for `asset` from the fills journal, or None.

    None means "no fill recorded", never "flat": a caller that has its own
    reconstruction must fall back to it rather than treat this as an answer.
    """
    return segment_of(open_fills(db_path=db_path).get(asset), today=today)
