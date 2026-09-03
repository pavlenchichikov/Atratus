"""Fill macro_calendar.json from the banks that publish their own schedules.

    python macro_calendar.py            # fetch, merge, show what changed
    python macro_calendar.py --dry-run  # show it without writing
    python macro_calendar.py --show     # what the file already holds

Nothing here invents a date. A source that cannot be reached contributes
nothing and is named in the output, and anything already in the file that the
sources do not carry is kept - see core/macro.py for why both rules exist.
"""

import argparse
import sys

from core import macro


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and print, write nothing")
    ap.add_argument("--show", action="store_true",
                    help="print the current file and exit, fetching nothing")
    ap.add_argument("--days", type=int, default=120,
                    help="how far ahead to list in the printout (default 120)")
    args = ap.parse_args(argv)

    existing = macro.load()
    if args.show:
        _print(macro.upcoming(existing, days=args.days), existing)
        return 0

    fetched, failed = macro.fetch()
    for name, why in sorted(failed.items()):
        print("[macro] %s contributed NOTHING: %s" % (name, why))
    if not fetched:
        print("[macro] no source answered, so the file is left exactly as it "
              "was. Nothing is invented here.")
        return 1

    merged = macro.merge(existing, fetched)
    added = len(merged) - len(existing)
    print("[macro] %d event(s) from %d source(s); file had %d, now %d (%+d)"
          % (len(fetched), len(macro.SOURCES) - len(failed), len(existing),
             len(merged), added))
    if not args.dry_run:
        macro.save(merged)
        print("[macro] wrote %s" % macro.PATH)
    else:
        print("[macro] --dry-run, nothing written")
    _print(macro.upcoming(merged, days=args.days), merged)
    return 0


def _print(rows, everything):
    print()
    if not rows:
        print("  nothing scheduled in this window (%d event(s) on file)"
              % len(everything))
        return
    print("  %-12s %-9s %-6s %s" % ("date", "importance", "region", "event"))
    for e in rows:
        print("  %-12s %-9s %-6s %s"
              % (e.get("date"), e.get("importance", ""), e.get("region", ""),
                 e.get("name", "")))
    print()
    print("  A source that failed is named above, not filled in with a guess.")


if __name__ == "__main__":
    sys.exit(main())
