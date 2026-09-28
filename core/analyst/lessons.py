"""What the analyst learns from its own resolved calls.

One or two sentences per scored judgment, written by the `memory` brain after
the outcome is known. A lesson carries the date its outcome RESOLVED, and
lessons_for shows only those resolved before `today`, so a rewound dossier
never learns from its own future.
"""

import datetime
import sqlite3

from core.analyst import store
from core.analyst.agent import plain

DDL = """
CREATE TABLE IF NOT EXISTS analyst_lessons (
    asset TEXT, klass TEXT, source_date TEXT, horizon INTEGER,
    resolved_date TEXT, lesson TEXT, brain TEXT, created TEXT,
    PRIMARY KEY (asset, source_date, horizon)
)
"""

PROMPT = (
    "On %(date)s you judged %(asset)s over %(h)d trading day(s): %(direction)s, "
    "conviction %(conviction)s.\nYour thesis: %(thesis)s\nThe risk you named: "
    "%(key_risk)s\nWhat happened: the price moved %(ret)+.2f%%.\n\n"
    "In one or two plain sentences, what should you do differently next time on "
    "this asset or this kind of asset? No JSON, no preamble.")


def _klass(asset):
    from config import radar_category
    return radar_category(asset)


def resolved_date(asset, date, horizon):
    """The date of the horizon-th bar after `date`, or None if not there yet."""
    from core.analyst.project_tools import _bars
    later = sorted({r[0] for r in _bars(asset, None, 400) if r[0] > date})
    h = int(horizon or 1)
    return later[h - 1] if len(later) >= h else None


def _connect(db_path):
    con = sqlite3.connect(db_path or store.DB_PATH)
    con.execute(DDL)
    return con


def learn(call, limit=20, db_path=None):
    """Lessons for scored judgments that have none yet. Returns how many."""
    from core.analyst import brains
    from core.llm_proposer import ProviderUnavailable, TerminalCallError

    con = _connect(db_path)
    try:
        done = set(con.execute("SELECT asset, source_date, horizon FROM analyst_lessons"))
    finally:
        con.close()
    todo = [r for r in store.scored_rows(db_path=db_path)
            if (r["asset"], r["date"], int(r["horizon"] or 1)) not in done]
    # `limit` counts model calls. A row with no resolvable date costs nothing
    # and is passed over; a row the model answered with nothing is recorded as
    # tried (lesson NULL), so neither can hold the oldest slots for ever.
    written = calls = 0
    for r in todo:
        if calls >= limit:
            break
        rd = resolved_date(r["asset"], r["date"], r["horizon"])
        if rd is None:
            continue
        calls += 1
        try:
            text = call(PROMPT % {
                "date": r["date"], "asset": r["asset"], "h": int(r["horizon"] or 1),
                "direction": r.get("direction"), "conviction": r.get("conviction"),
                "thesis": (r.get("thesis") or "")[:800],
                "key_risk": (r.get("key_risk") or "")[:300],
                "ret": 100.0 * float(r["realized_ret"])})
        except (TerminalCallError, ProviderUnavailable):
            raise
        except Exception:
            text = ""
        lesson = plain(str(text or "")).strip()[:400] or None
        con = _connect(db_path)
        try:
            with con:
                con.execute("INSERT OR IGNORE INTO analyst_lessons VALUES (?,?,?,?,?,?,?,?)",
                            (r["asset"], _klass(r["asset"]), r["date"], int(r["horizon"] or 1),
                             rd, lesson, brains.label("memory"),
                             datetime.datetime.now().isoformat(timespec="seconds")))
        finally:
            con.close()
        written += lesson is not None
    return written


def lessons_for(asset, klass, today=None, k=3, db_path=None):
    """Up to k lessons: this asset's newest first, then its class's."""
    con = _connect(db_path)
    try:
        cond, extra = "", []
        if today:
            cond, extra = "AND resolved_date < ? ", [str(today)[:10]]
        order = "ORDER BY resolved_date DESC LIMIT ?"
        own = con.execute("SELECT asset, resolved_date, lesson FROM analyst_lessons "
                          "WHERE lesson IS NOT NULL AND asset=? " + cond + order, [asset, *extra, k]).fetchall()
        rest = con.execute("SELECT asset, resolved_date, lesson FROM analyst_lessons "
                           "WHERE lesson IS NOT NULL AND klass=? AND asset<>? " + cond + order,
                           [klass, asset, *extra, k]).fetchall()
    finally:
        con.close()
    return ["%s %s: %s" % (d, a, t) for a, d, t in (own + rest)[:k]]
