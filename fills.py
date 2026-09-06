"""Record what you actually filled at the broker, so the sheet stops guessing.

    python fills.py add SBER BUY 100 315.4          # opened today
    python fills.py add SBER BUY 100 315.4 --date 2026-09-03
    python fills.py close SBER 322.1
    python fills.py list

Orders are placed by hand, so the trailing stop on /levels was being measured
from the bar the SIGNAL turned rather than from the bar you bought on. One open
position per asset; see core/fills.py for why that is a database constraint and
not a convention.
"""
import argparse

from core import fills


def _fmt(row):
    tail = ("open" if not row["exit_date"]
            else "closed %s at %g" % (row["exit_date"], row["exit_price"]))
    return ("%-10s %-4s qty %-10g entry %s at %-10g %s"
            % (row["asset"], row["side"], row["qty"], row["entry_date"],
               row["price"], tail))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="record an opening fill")
    a.add_argument("asset")
    a.add_argument("side", choices=("BUY", "SELL", "buy", "sell"))
    a.add_argument("qty", type=float)
    a.add_argument("price", type=float)
    a.add_argument("--date", help="fill date YYYY-MM-DD, default today")
    a.add_argument("--note")

    c = sub.add_parser("close", help="close the open fill on an asset")
    c.add_argument("asset")
    c.add_argument("price", type=float)
    c.add_argument("--date", help="exit date YYYY-MM-DD, default today")

    ls = sub.add_parser("list", help="what is open, and recent closed fills")
    ls.add_argument("--history", type=int, default=10)

    args = ap.parse_args()
    if args.cmd == "add":
        rid = fills.record(args.asset, args.side, args.qty, args.price,
                           entry_date=args.date, note=args.note)
        print("recorded fill %d" % rid)
    elif args.cmd == "close":
        rid = fills.close(args.asset, args.price, exit_date=args.date)
        print("closed fill %d" % rid if rid else
              "%s has no open fill" % args.asset)
    else:
        open_rows = fills.open_fills()
        print("OPEN (%d)" % len(open_rows))
        for row in open_rows.values():
            print("  " + _fmt(row))
        past = fills.history(limit=args.history)
        if past:
            print("CLOSED (last %d)" % len(past))
            for row in past:
                print("  " + _fmt(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
