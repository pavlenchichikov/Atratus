"""Remove the daily-spaced rows that a weekly fetch appended to *_weekly.

    python repair_weekly_tables.py                 # scan only
    python repair_weekly_tables.py --apply         # delete, then re-run [4]

Yahoo's interval=1wk appends a bar for the week IN PROGRESS, stamped with the
day of the request instead of the week's start, at every window length:

    20-day window -> Mon 08-17, Mon 08-24, Mon 08-31, Mon 09-07, Tue 09-08

Every daily update therefore wrote one row stamped that day. Measured
2026-09-08: 665 of the 850 weekly tables carry such rows, 14021 of them dated
2026 or later and 502 older, and aapl_weekly went from a 7-day median step to a
1-day one. data_engine.fetch_yahoo_weekly now drops that trailing bar, so this
is the one-off cleanup of what it already wrote.

Two signatures are cut, both described on first_bad_date: a run of daily-spaced
rows, and a trailing bar whose week has not finished yet. A gap of 8, 14 or 35
days deeper in the history is a delisting or a halt, not a defect, and stays.

Everything from the cut onward goes, not the bad rows alone: the refetch only
writes dates AFTER the table's newest row, so an interior hole would never be
filled. The 15-year window restores whatever the delete removed.
"""
import argparse
import datetime as dt
import os
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "market.db")
WEEK_DAYS = 7


def first_bad_date(con, table, today=None):
    """The date from which the table stops holding completed weekly bars.

    Two different defects, one cut point.

    A run of daily-spaced rows is the old daily-update damage: a gap under seven
    days can only mean the fetch appended an in-progress bar stamped with the
    day of the request.

    A trailing bar younger than seven days is the same defect caught fresh, and
    the gap test cannot see it. aapl_weekly held 08-17, 08-24, 08-31 and then
    09-08, an 8-day gap that a gap rule waves through, and 845 of the 850 tables
    carried exactly one such bar. Age is the honest test: a bar stamped at the
    start of week W is complete only once seven days have passed.

    A gap of 8, 14 or 35 days deeper in the history is NOT a defect. There are
    969 such rows before 2026: delistings, halts and holidays. Only the two
    signatures above are cut.
    """
    today = today or dt.date.today()
    rows = con.execute('SELECT Date FROM "%s" ORDER BY Date' % table).fetchall()
    dates = [str(r[0])[:10] for r in rows if r[0]]
    if len(dates) < 5:
        return None
    prev = dt.date.fromisoformat(dates[0])
    for d in dates[1:]:
        cur = dt.date.fromisoformat(d)
        if (cur - prev).days < WEEK_DAYS:
            return d
        prev = cur
    last = dt.date.fromisoformat(dates[-1])
    if (today - last).days < WEEK_DAYS:
        return dates[-1]
    return None


def cut_plan(db_path, today=None):
    """[(table, cut date, rows from there on)] for every table needing a cut."""
    read = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        plan = []
        for t in sorted(r[0] for r in read.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE '%_weekly'")):
            cut = first_bad_date(read, t, today=today)
            if not cut:
                continue
            n = read.execute('SELECT count(*) FROM "%s" WHERE Date >= ?' % t,
                             (cut,)).fetchone()[0]
            plan.append((t, cut, n))
        return plan
    finally:
        read.close()


def apply_cuts(db_path, plan):
    """Delete what cut_plan found. Returns rows removed."""
    write = sqlite3.connect(db_path)
    try:
        gone = sum(write.execute('DELETE FROM "%s" WHERE Date >= ?' % t,
                                 (cut,)).rowcount for t, cut, _n in plan)
        write.commit()
        return gone
    finally:
        write.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="delete the rows; without it nothing is changed")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    read = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    total = len([r[0] for r in read.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_weekly'")])
    read.close()
    plan = cut_plan(args.db)

    print("SCAN of %s" % args.db)
    print("  weekly tables            : %d" % total)
    print("  tables needing a cut     : %d" % len(plan))
    print("  rows to delete           : %d" % sum(n for _t, _c, n in plan))
    if plan:
        print("  earliest cut date        : %s" % min(c for _t, c, _n in plan))
        print("  worst tables:")
        for t, c, n in sorted(plan, key=lambda x: -x[2])[:8]:
            print("     %-22s from %s, %d rows" % (t, c, n))

    if not args.apply:
        print()
        print("scan only, nothing changed. Add --apply, then run [4] Data Update "
              "so the fixed fetch refills the weeks.")
        return 0

    gone = apply_cuts(args.db, plan)
    print()
    print("deleted %d rows from %d tables." % (gone, len(plan)))
    print("NOW RUN [4] Data Update: the tables are short by design until the "
          "fixed weekly fetch refills them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
