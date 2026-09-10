"""What a round trip actually costs you, measured from your own fills.

    python cost_report.py              # the measurement, and what it implies
    python cost_report.py --cost 0.001 # what a hypothetical cost would imply

This is the highest-leverage number in the project and nobody has measured it.

Measured 2026-09-10 over 40 assets, the median absolute move and the accuracy it
demands at a given round trip:

    horizon    move      0.50%    0.20%    0.10%    0.05%
    day       0.665%     87.6%    65.0%    57.5%    53.8%
    week      1.716%     64.6%    55.8%    52.9%    51.5%
    month     3.735%     56.7%    52.7%    51.3%    50.7%

The pooled panel puts the feature set at 52.6% out of sample over 2.8M rows. So
at the 0.5% the backtest assumes, NO horizon pays and no model will fix that; at
0.1% the week already pays. The difference between a system that cannot work and
one that does is a constant that was guessed, and the live risk config guesses a
different one (fee_rate 0.0).

The measurement is simple: the backtest charges its cost on close-to-close bar
returns, so it assumes you trade at the close. Whatever your fill differs from
that close by, in the direction that hurts you, IS the slippage. Add the
broker's commission and you have the round trip.

Record fills with fills.py as you trade; twenty are enough to tell 0.5% from
0.1%.
"""
import argparse
import os
import sqlite3
import statistics as st

from core import backtesting, fills

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "market.db")

# Measured 2026-09-10 on 40 assets from intraday.db: the median absolute close
# to close move over each holding period. They set what any cost has to clear.
MOVES = {"day": 0.00665, "week": 0.01716, "month": 0.03735}
ACCURACY = 0.526          # pooled panel, 2.8M rows, out of sample, leak-free


def _table(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def close_on(con, asset, date):
    """The daily close the backtest would have traded at, or None."""
    try:
        row = con.execute('SELECT close FROM "%s" WHERE Date LIKE ?' % _table(asset),
                          (str(date)[:10] + "%",)).fetchone()
    except sqlite3.Error:
        return None
    return float(row[0]) if row and row[0] else None


def legs(rows, con):
    """[(asset, date, 'entry'|'exit', slippage as a fraction)] over every fill.

    Signed so that POSITIVE always means worse than the close. Buying above the
    close and selling below it both cost money; averaging the raw difference
    would let one cancel the other and report a slippage of zero.
    """
    out = []
    for r in rows:
        side = 1 if str(r["side"]).upper() in ("BUY", "LONG", "1") else -1
        ref = close_on(con, r["asset"], r["entry_date"])
        if ref and r["price"]:
            out.append((r["asset"], r["entry_date"], "entry",
                        side * (float(r["price"]) - ref) / ref))
        if r["exit_date"] and r["exit_price"]:
            ref = close_on(con, r["asset"], r["exit_date"])
            if ref:
                out.append((r["asset"], r["exit_date"], "exit",
                            -side * (float(r["exit_price"]) - ref) / ref))
    return out


def implied(round_trip):
    """{horizon: required accuracy} at this round trip."""
    return {h: min(0.5 * (1 + round_trip / m), 1.0) for h, m in MOVES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=None,
                    help="skip the fills and ask what this round trip would imply")
    ap.add_argument("--commission", type=float, default=None,
                    help="broker commission per leg, as a fraction; the fills "
                         "only carry slippage")
    args = ap.parse_args()

    assumed = backtesting.COMMISSION + backtesting.SLIPPAGE
    print("the backtest currently charges %.3f%% per leg, %.3f%% round trip"
          % (100 * (backtesting.COMMISSION + backtesting.SLIPPAGE),
             100 * 2 * assumed))

    measured = None
    if args.cost is None:
        rows = fills.history(limit=10000)
        if not rows:
            print()
            print("NO FILLS RECORDED. Nothing to measure, and this is the whole")
            print("problem: the number that decides whether any horizon pays has")
            print("never been observed. Record them as you trade:")
            print()
            print("    python fills.py add SBER BUY 100 315.4")
            print("    python fills.py close SBER 322.1")
            print()
            print("Twenty round trips separate 0.5% from 0.1%.")
        else:
            data = legs(rows, sqlite3.connect("file:%s?mode=ro" % DB, uri=True))
            if not data:
                print("\n%d fill(s) recorded, but none could be matched to a bar."
                      % len(rows))
            else:
                slips = [s for _a, _d, _k, s in data]
                comm = args.commission or 0.0
                measured = 2 * (st.median(slips) + comm)
                print()
                print("MEASURED from %d fill(s), %d leg(s)" % (len(rows), len(slips)))
                print("  median slippage per leg : %+.4f%%" % (100 * st.median(slips)))
                print("  mean   slippage per leg : %+.4f%%" % (100 * st.mean(slips)))
                print("  worst leg               : %+.4f%%" % (100 * max(slips)))
                if comm:
                    print("  commission per leg      : %.4f%%" % (100 * comm))
                print("  ROUND TRIP              : %.4f%%" % (100 * measured))
                if len(slips) < 20:
                    print("  (%d legs is thin; twenty round trips is the point "
                          "where 0.5%% and 0.1%% stop overlapping)" % len(slips))

    for label, cost in (("assumed", 2 * assumed),
                        ("measured", measured),
                        ("asked", args.cost)):
        if cost is None:
            continue
        need = implied(cost)
        print()
        print("at %s round trip of %.3f%%, against %.1f%% accuracy:"
              % ("an " + label if label == "assumed" else "a " + label,
                 100 * cost, 100 * ACCURACY))
        for h in ("day", "week", "month"):
            ok = need[h] <= ACCURACY
            print("  %-6s move %.3f%%  needs %5.1f%%   %s"
                  % (h, 100 * MOVES[h], 100 * need[h],
                     "PAYS" if ok else "does not pay"))
    print()
    print("break-even round trip at %.1f%% accuracy: %s" % (
        100 * ACCURACY,
        "  ".join("%s %.3f%%" % (h, 100 * m * (2 * ACCURACY - 1))
                  for h, m in MOVES.items())))
    print("Set a measured figure with GTRADE_COMMISSION and GTRADE_SLIPPAGE "
          "(per leg) so backtests stop guessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
