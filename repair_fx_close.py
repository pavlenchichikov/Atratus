"""Give Yahoo's daily forex bars their real close.

    python repair_fx_close.py            # scan only
    python repair_fx_close.py --apply    # rewrite

Yahoo's daily `=X` bars store the day's OPEN in the close field while high and
low cover the whole day. Measured 2026-09-26 on the 34 pairs: median
|close - open| / (high - low) of 0.00-0.02 since 2016 (SPY, BTC, GOLD: 0.43 to
0.63), and a stored close 23-32 bp away from the true daily close over ~495
days of hourly bars. The direction label then scores the very day whose range
the row already shows: (close - low) / (high - low) alone predicted it at AUC
0.87-0.90 on every pair, 0.49-0.58 on the controls.

The market trades round the clock, so a day's close is the next day's open:
1.2-2.5 bp from the hourly close over the same ~495 days, and with it the same
single feature falls to AUC 0.47-0.56. This rewrites close from the next row's
open and widens high/low to take it in (Yahoo's range window ends a little
before the close on about a quarter of the days, by a median 9 bp).

It also deletes rows that are not daily bars: Yahoo's live snapshot, stored on
a Saturday or Sunday, and with --newest a newest row whose close is still its
open - it has no successor yet, and the next fetch stores it once one exists
(data_engine._fx_real_close). Applied 2026-09-26 with --newest: 194649 closes
rewritten and 822 rows deleted over 34 tables, backup in
market.pre_fxclose_20260926.db. Running it again changes nothing: close(D) =
open(D+1) depends only on opens, and opens are never touched.
"""
import argparse
import datetime as dt
import itertools
import os
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "market.db")
# A newest row this close to its own open is the unrepaired kind. A real close
# lands this near the open on a few percent of days; deleting one of those only
# costs a refetch of that bar.
LEAKY_RATIO = 0.05


def fx_tables():
    """Daily tables of every asset Yahoo quotes as a `=X` pair."""
    from config import FULL_ASSET_MAP

    return sorted(k.lower().replace("^", "").replace(".", "").replace("-", "")
                  for k, v in FULL_ASSET_MAP.items() if str(v).endswith("=X"))


def plan_table(con, table, newest=False):
    """What --apply would change in one table: updates (close, high, low, date),
    weekend snapshot dates, and - with newest=True - an unfinished newest date.

    `newest` is for the one-off migration only (run 2026-09-26). A repaired
    newest row lands this near its open on a few percent of days too, so the
    every-run DATA HEALTH scan must not judge it: measured right after the
    migration, 2 of 34 tables would have lost their newest bar on every run."""
    rows = con.execute('SELECT Date, open, high, low, close FROM "%s" ORDER BY Date'
                       % table).fetchall()
    weekend = [r[0] for r in rows
               if dt.date.fromisoformat(str(r[0])[:10]).weekday() >= 5]
    skip = set(weekend)
    days = [r for r in rows if r[0] not in skip and None not in r]
    updates = []
    for (d, _o, h, low, c), nxt in itertools.pairwise(days):
        close = nxt[1]
        high, lo = max(h, close), min(low, close)
        if abs(close - c) > 1e-9 * abs(close) or high != h or lo != low:
            updates.append((close, high, lo, d))
    unfinished = []
    if newest and days:
        d, o, h, low, c = days[-1]
        if h > low and abs(c - o) / (h - low) < LEAKY_RATIO:
            unfinished = [d]
    return {"updates": updates, "weekend": weekend, "unfinished": unfinished}


def scan(db=DB, tables=None, newest=False):
    con = sqlite3.connect(db)
    try:
        have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {t: plan_table(con, t, newest) for t in (tables or fx_tables()) if t in have}
    finally:
        con.close()


def apply(db, plans):
    """Write the plans. Returns (closes rewritten, rows deleted)."""
    con = sqlite3.connect(db)
    upd = gone = 0
    try:
        for t, p in plans.items():
            con.executemany('UPDATE "%s" SET close = ?, high = ?, low = ? WHERE Date = ?' % t,
                            p["updates"])
            dead = p["weekend"] + p["unfinished"]
            con.executemany('DELETE FROM "%s" WHERE Date = ?' % t, [(d,) for d in dead])
            upd += len(p["updates"])
            gone += len(dead)
        con.commit()
    finally:
        con.close()
    return upd, gone


def totals(plans):
    upd = sum(len(p["updates"]) for p in plans.values())
    gone = sum(len(p["weekend"]) + len(p["unfinished"]) for p in plans.values())
    touched = sum(1 for p in plans.values()
                  if p["updates"] or p["weekend"] or p["unfinished"])
    return upd, gone, touched


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--newest", action="store_true",
                    help="also delete a newest row whose close is still its open "
                         "(the one-off migration; see plan_table)")
    args = ap.parse_args()
    plans = scan(args.db, newest=args.newest)
    for t, p in plans.items():
        if p["updates"] or p["weekend"] or p["unfinished"]:
            print("%-8s %5d close(s)  %3d weekend row(s)  %s"
                  % (t, len(p["updates"]), len(p["weekend"]),
                     "newest %s unfinished" % p["unfinished"][0] if p["unfinished"] else ""))
    upd, gone, touched = totals(plans)
    print("%d table(s): %d close(s) to set from the next open, %d row(s) to delete"
          % (touched, upd, gone))
    if args.apply and (upd or gone):
        print("applied: %d rewritten, %d deleted" % apply(args.db, plans))


if __name__ == "__main__":
    main()
