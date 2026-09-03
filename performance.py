"""What an asset returned over a period, and how that compares to its index.

    python performance.py --asset SBER
    python performance.py --asset NVDA --windows 1M,6M,1Y,5Y
    python performance.py --asset BTC --windows 90,180,365 --json

Reads market.db and nothing else, so it costs no network call and works with
the VPN down. The arithmetic and the three things it refuses to do are in
core/performance.py.
"""

import argparse
import json
import sys

from config import radar_category
from core.analyst.dossier import BENCHMARK_BY_CLASS
from core.performance import BASE_WINDOWS, summary, table


def benchmark_for(asset):
    """The index this asset is measured against, or None off the class map."""
    return BENCHMARK_BY_CLASS.get(radar_category(asset))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--windows", default=",".join(BASE_WINDOWS),
                    help="named (1M,3M,6M,YTD,1Y,3Y,5Y,MAX) or a day count")
    ap.add_argument("--benchmark", default=None,
                    help="asset code; defaults to the class index, "
                         "'none' to compare against nothing")
    ap.add_argument("--json", action="store_true",
                    help="the full dict per window, not the printed table")
    args = ap.parse_args(argv)

    asset = args.asset.strip().upper()
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    if args.benchmark is None:
        bmk = benchmark_for(asset)
    else:
        bmk = None if args.benchmark.lower() == "none" else args.benchmark.upper()

    if args.json:
        print(json.dumps([summary(asset, w, benchmark=bmk) for w in windows],
                         indent=2, ensure_ascii=False))
        return 0

    print()
    print("  %s   benchmark %s" % (asset, bmk or "none"))
    print()
    for line in table(asset, windows=windows, benchmark=bmk):
        print("  " + line)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
