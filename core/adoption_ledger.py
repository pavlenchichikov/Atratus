"""What each adoption did to the LIVE number, recorded when it is made.

The gate measures a backtest Score. The only thing that can ever confirm the
whole apparatus is whether an adoption moved the live directional accuracy of
the assets it touched, and on 2026-09-03 that could not be answered at all:
49.1% over 8742 scored calls, with nothing tying any of it to the history of
adoptions.

So the "before" side is written at adoption time, when it is cheap and correct,
and the "after" side is computed on demand from `prediction_log`. It will say
nothing useful for months - a per-asset call arrives about once a day - which is
exactly why the recording has to start before anyone needs the answer.

Standard library only: adopt_genome must not pull the research machinery in.
"""

import datetime
import json
import os
import sqlite3

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "market.db")
PATH = os.path.join(BASE, "_adoption_ledger.json")


def live_accuracy(assets=None, since=None, until=None, db_path=None):
    """{n, correct, accuracy} over SCORED directional calls, or n=0.

    WAIT is never counted: it is not a recommendation, and counting it would
    dilute the number with the days the system declined to speak. A row counts
    only once its outcome is known (actual_next_ret is filled by the reconcile).
    """
    where = ["actual_next_ret IS NOT NULL", "signal IN ('BUY','SELL')"]
    args = []
    if assets:
        where.append("asset IN (%s)" % ",".join("?" * len(assets)))
        args += [a.strip().upper() for a in assets]
    if since:
        where.append("date >= ?")
        args.append(str(since))
    if until:
        where.append("date <= ?")
        args.append(str(until))
    sql = ("SELECT COUNT(*), SUM(CASE WHEN (signal='BUY' AND actual_next_ret>0) "
           "OR (signal='SELL' AND actual_next_ret<0) THEN 1 ELSE 0 END) "
           "FROM prediction_log WHERE " + " AND ".join(where))
    try:
        con = sqlite3.connect("file:%s?mode=ro" % (db_path or DB_PATH), uri=True)
        try:
            n, k = con.execute(sql, args).fetchone()
        finally:
            con.close()
    except Exception:
        return {"n": 0, "correct": 0, "accuracy": None}
    n, k = int(n or 0), int(k or 0)
    return {"n": n, "correct": k, "accuracy": (k / n) if n else None}


def _load(path=None):
    try:
        with open(path or PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def record(label, assets=None, note="", path=None, db_path=None, today=None):
    """Bank one adoption with the live accuracy its assets had BEFORE it.

    `assets` None means the adoption is global and the entry covers everything.
    Nothing here can fail loudly: an adoption must not be blocked by its own
    bookkeeping.
    """
    day = str(today or datetime.date.today().isoformat())
    entry = {
        "label": label,
        "assets": sorted(a.strip().upper() for a in assets) if assets else None,
        "adopted": day,
        "note": note,
        "before": live_accuracy(assets, until=day, db_path=db_path),
    }
    book = _load(path)
    book.append(entry)
    try:
        with open(path or PATH, "w", encoding="utf-8") as fh:
            json.dump(book, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return entry


def report_lines(path=None, db_path=None):
    """One line per adoption: what the live number was, and what it is since.

    The two sides are different samples, not a before/after of the same days, so
    the line says how many calls each rests on. With a few dozen calls the
    difference means nothing at all, and the count is what makes that visible
    instead of inviting a story.
    """
    book = _load(path)
    if not book:
        return ["No adoption has been recorded yet."]
    out = ["%-14s %-10s %-22s %-22s" % ("adopted", "label", "before", "since")]
    for e in book:
        after = live_accuracy(e.get("assets"), since=e.get("adopted"), db_path=db_path)
        before = e.get("before") or {}

        def fmt(d):
            acc = d.get("accuracy")
            return "n/a" if acc is None else "%.1f%% of %d" % (100 * acc, d["n"])

        out.append("%-14s %-10s %-22s %-22s%s"
                   % (e.get("adopted"), str(e.get("label"))[:10],
                      fmt(before), fmt(after),
                      "" if e.get("assets") is None
                      else "  " + ",".join(e["assets"][:4])))
    out.append("")
    out.append("Both sides are directional calls only, WAIT excluded. They are "
               "different days, not a paired comparison, so read the counts "
               "before the percentages.")
    return out
