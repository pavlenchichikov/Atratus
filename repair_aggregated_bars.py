"""Find, and optionally repair, daily rows that are aggregated bars.

    python repair_aggregated_bars.py                 # scan only, offline
    python repair_aggregated_bars.py --sample 25     # scan + refetch a sample
    python repair_aggregated_bars.py --apply         # rewrite the rows

`data_engine.py:131` records the cause: "range=max with interval=1d returns
quarterly data for long histories". Some backfill took that path and stored the
aggregate as a daily row. AAPL 2025-12-01 claims a 17.1% range where the real
session was 2.6%, and its own open is correct while the high, low and close span
the whole month, so the row is wrong in a way nothing downstream can notice.

Two independent sources agree on the defect AND the fix. Aggregating that day's
hourly bars gives 2.6%, and refetching the same date with a BOUNDED period gives
2.6% as well, against the 17.1% stored. The vendor is fine; the original request
was not.

WHAT IS AND IS NOT A DEFECT. A wide day is not enough: 2020-03, 2022-03 and
2022-04 are full of real ones, and SBER 2022-02-24 shows a 74.1% range that the
hourly bars confirm exactly. Three conditions together, validated against the
hourly bars on 19 assets at 16 true positives and 0 false ones:

  1. the range exceeds five times the asset's own median,
  2. the row brackets the whole of the next month (a real crash day does not),
  3. the date is the first of a month, which is where an aggregate lands.

Condition 3 alone removes every false positive the first two produce: all six
were MOEX names on other days, and all sixteen confirmed defects were day one.
"""
import argparse
import datetime as dt
import os
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

import config
import net

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "market.db")
RANGE_MULT = 5.0          # of the asset's own median range
FORWARD = 20              # sessions the row must bracket to count as an aggregate


def suspects(db_path=None):
    """[(table, date, how many median ranges wide)] for rows meeting all three."""
    con = sqlite3.connect("file:%s?mode=ro" % (db_path or DB), uri=True)
    try:
        out = []
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
            if not r[0].endswith("_weekly")]
        for t in names:
            try:
                rows = con.execute(
                    'SELECT Date, open, high, low, close FROM "%s" ORDER BY Date' % t
                ).fetchall()
            except sqlite3.Error:
                continue
            g = [(str(d)[:10], o, h, lo, c) for d, o, h, lo, c in rows
                 if None not in (d, o, h, lo, c) and c]
            if len(g) < 300:
                continue
            med = st.median((h - lo) / c for _d, _o, h, lo, c in g)
            if med <= 0:
                continue
            for i, (d, _o, h, lo, c) in enumerate(g):
                if (h - lo) / c <= RANGE_MULT * med or d[8:10] != "01":
                    continue
                nxt = g[i + 1:i + 1 + FORWARD]
                if len(nxt) < 15:
                    continue
                if h >= max(x[2] for x in nxt) and lo <= min(x[3] for x in nxt):
                    out.append((t, d, (h - lo) / c / med))
        return out
    finally:
        con.close()


def history(symbol, years=20):
    """{date: bar} for a whole history, in ONE request. The correct path.

    `data_engine.fetch_yahoo_smart` already fetches exactly this way, with an
    explicit period1/period2 window rather than range=max, so the cause of these
    rows is ALREADY FIXED and they are pure legacy. Verified 2026-09-08: explicit
    windows return clean daily bars at 2, 10, 16 and 25 years (median gap 1.0
    days at every length), and a standard 15-year request returns the correct bar
    for every known-bad date.

    One request per asset rather than one per row: 52 against 2494. Twenty
    years, not the fifteen data_engine uses, because market.db holds rows from
    2011 that a fifteen-year window would leave unrepaired.

    The date comes from the payload's own exchange zone, not from the local
    clock. A daily stamp is the session open, so 00:00 in Tokyo is the previous
    UTC day, and reading it as UTC or as Moscow shifts every such bar by one.
    """
    now = int(dt.datetime.now(dt.UTC).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d"
           "&period1=%d&period2=%d" % (symbol, now - years * 365 * 86400, now - 300))
    res = net.http_get(url, route="auto", retries=2,
                       timeout=(15, 60)).json()["chart"]["result"][0]
    tz = ZoneInfo(res.get("meta", {}).get("exchangeTimezoneName") or "UTC")
    q = res["indicators"]["quote"][0]
    out = {}
    for i, ts in enumerate(res.get("timestamp") or []):
        if any(q[k][i] is None for k in ("open", "high", "low", "close")):
            continue
        d = dt.datetime.fromtimestamp(int(ts), tz=tz).strftime("%Y-%m-%d")
        out[d] = {k: float(q[k][i]) for k in ("open", "high", "low", "close")}
    return out


def is_phantom(hist, date, window=3):
    """True when the vendor covers this date's neighbourhood but has no session
    ON it. Then the stored row is not a wrong bar, it is a day that never traded.

    Measured 2026-09-08: 784 of the flagged rows fall on a Saturday, a Sunday or
    an exchange holiday. An aggregate is stamped on the first of the month, and
    the first is often not a trading day, so the row is pure phantom and the
    repair is deletion rather than a rewrite.

    The neighbourhood check is what keeps a data gap from being read as a
    holiday: bars on BOTH sides prove the vendor covers this stretch, so the
    absence is the calendar and not a hole. Crypto needs no special case, its
    weekends really do have bars.
    """
    if date in hist:
        return False
    d0 = dt.date.fromisoformat(date)
    before = after = False
    for k in hist:
        delta = (dt.date.fromisoformat(k) - d0).days
        if -window <= delta < 0:
            before = True
        elif 0 < delta <= window:
            after = True
    return before and after


def selftest():
    """The phantom rule, with its positive control. Deleting production rows on
    a rule nobody ever watched fail is how a cleanup becomes the next defect."""
    hist = {"2012-08-31": {}, "2012-09-04": {}}
    assert is_phantom(hist, "2012-09-01")                  # closed, covered
    assert not is_phantom({"2012-09-01": {}}, "2012-09-01")  # the bar exists
    assert not is_phantom({"2012-08-31": {}}, "2012-09-01")  # gap, not a holiday
    assert not is_phantom({"2012-09-04": {}}, "2012-09-01")  # gap on the left
    assert not is_phantom({"2012-08-20": {}, "2012-09-20": {}}, "2012-09-01")
    print("selftest ok")
    return 0


# The filter is RELATIVE: it compares a row against its own asset's median
# range. Removing the worst rows lowers that median and re-exposes the next
# layer, so one pass is never enough. Measured on this database: 2494 rows on
# the first pass, 168 on the second, 33 on the third, and 14 of those 15 that
# landed on already-retrained assets were still genuine defects. Running to a
# fixed point is the difference between a repair and a repair that has to be
# discovered again after the next retrain.
MAX_PASSES = 8


def one_pass(args, quiet=False):
    """Scan, repair, and report what moved. (fixed, dropped, skipped, confirmed)."""
    found = suspects(args.db)
    if not quiet:
        by_asset = Counter(t for t, _d, _m in found)
        print("  rows meeting all three conditions : %d" % len(found))
        print("  assets affected                   : %d" % len(by_asset))
        for t, n in by_asset.most_common(5):
            print("     %-14s %d rows" % (t, n))
    if not found:
        return 0, 0, 0, 0

    inv = {a.lower().replace("^", "").replace(".", "").replace("-", ""): a
           for a in config.FULL_ASSET_MAP}
    per_asset = defaultdict(list)
    for table, date, _mult in found:
        per_asset[table].append(date)
    tables = sorted(per_asset)
    if args.sample:
        tables = tables[:args.sample]

    write = sqlite3.connect(args.db) if args.apply else None
    read = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    fixed = dropped = skipped = confirmed = 0
    try:
        for table in tables:
            dates = per_asset[table]
            asset = inv.get(table)
            sym = config.FULL_ASSET_MAP.get(asset) if asset else None
            if not sym or asset in getattr(config, "MOEX_ASSETS", []):
                skipped += len(dates)
                continue
            try:
                hist = history(sym)
            except Exception as exc:
                print("  %-14s fetch failed: %s" % (table, str(exc)[:50]))
                skipped += len(dates)
                continue
            worst = 0
            done = gone = same = 0
            for date in dates:
                good = hist.get(date)
                if good is None:
                    if is_phantom(hist, date):
                        if args.apply:
                            write.execute('DELETE FROM "%s" WHERE Date LIKE ?'
                                          % table, (date + "%",))
                        gone += 1
                    else:
                        skipped += 1
                    continue
                old = read.execute(
                    'SELECT open, high, low, close FROM "%s" WHERE Date LIKE ?' % table,
                    (date + "%",)).fetchone()
                if not old or not old[3]:
                    skipped += 1
                    continue
                # A row the vendor confirms is a genuinely wide day, not a defect.
                # Counting it as repaired would keep the pass loop from settling,
                # because the filter re-flags it on every pass forever.
                if all(abs(old[i] - good[k]) < 1e-9 for i, k in
                       enumerate(("open", "high", "low", "close"))):
                    same += 1
                    continue
                before = 100 * (old[1] - old[2]) / old[3]
                after = 100 * (good["high"] - good["low"]) / good["close"]
                worst = max(worst, before - after)
                if args.apply:
                    write.execute(
                        'UPDATE "%s" SET open=?, high=?, low=?, close=? '
                        'WHERE Date LIKE ?' % table,
                        (good["open"], good["high"], good["low"],
                         good["close"], date + "%"))
                done += 1
            if args.apply:
                write.commit()
            fixed += done
            dropped += gone
            confirmed += same
            if done or gone:
                print("  %-14s rewrite %3d  drop %3d  of %3d, widest %5.1f pp"
                      % (table, done, gone, len(dates), worst))
    finally:
        read.close()
        if write:
            write.close()
    return fixed, dropped, skipped, confirmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the rows; without it nothing is changed")
    ap.add_argument("--sample", type=int, default=0,
                    help="limit to this many assets per pass")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    print("SCAN of %s" % args.db)
    if not args.apply:
        f, d, sk, cf = one_pass(args)
        print()
        print("would rewrite %d rows and drop %d phantom rows; %d confirmed wide, "
              "%d skipped (MOEX or no vendor cover)" % (f, d, cf, sk))
        print("nothing was changed. Add --apply to repair, which repeats until "
              "the scan stops finding anything.")
        return 0

    total_f = total_d = 0
    for n in range(1, MAX_PASSES + 1):
        print()
        print("PASS %d" % n)
        f, d, sk, cf = one_pass(args, quiet=(n > 1))
        total_f += f
        total_d += d
        print("  pass %d: rewrote %d, dropped %d, confirmed wide %d, skipped %d"
              % (n, f, d, cf, sk))
        if f == 0 and d == 0:
            print()
            print("settled after %d pass(es): rewrote %d rows, dropped %d phantom "
                  "rows in total." % (n, total_f, total_d))
            return 0
    print()
    print("STILL MOVING after %d passes (rewrote %d, dropped %d). Run again."
          % (MAX_PASSES, total_f, total_d))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
