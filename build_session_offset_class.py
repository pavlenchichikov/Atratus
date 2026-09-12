"""Assign assets to the session-offset class BY RULE, before any score is read.

Spec: docs/superpowers/specs/2026-09-12-session-offset-accuracy-preregistration.md

An asset is IN the class when its session opens after the US regular session of
the previous day has closed and before the next US open. The exchange timezone
comes from what the vendor stamped on the hourly bars (asset_tz), not from a
hand-written table, so nobody chooses membership by looking at the result.

Writes _session_offset_class.json and prints the counts. Reads no accuracy
number and imports nothing that does.
"""
import json
import os
import sqlite3

import config

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "_session_offset_class.json")
DB = os.path.join(BASE, "intraday.db")

US_CLOSE_UTC = 20          # 20:00 UTC, the US regular close outside winter
# 07:00 UTC, where European trading begins. NOT the next US open: a window that
# ran to 13:00 admitted every European and Russian open and put 366 assets in a
# class this spec excludes Europe from. The mechanism is a SHORT gap after the
# US close - Asia opens within hours of it, Europe eleven hours later.
EUROPE_OPEN_UTC = 7
ROUND_THE_CLOCK = ("CRYPTO", "FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC")


def _never_closes(asset):
    return any(asset in config.ASSET_TYPES.get(g, ()) for g in ROUND_THE_CLOCK)


def session_open_hour(con, asset):
    """Median UTC hour of an asset's first bar of a session, or None.

    The median over sessions, not one sample: a single early print would
    otherwise decide the class for a whole exchange.
    """
    rows = con.execute(
        "SELECT substr(ts, 1, 10) AS d, min(substr(ts, 12, 2)) AS h "
        "FROM bars_1h WHERE asset = ? GROUP BY d ORDER BY d DESC LIMIT 60",
        (asset,)).fetchall()
    hours = sorted(int(h) for _d, h in rows if h is not None)
    if len(hours) < 10:
        return None
    return hours[len(hours) // 2]


def classify(con, assets):
    inside, outside, unknown = [], [], []
    for asset in assets:
        if _never_closes(asset):
            outside.append(asset)
            continue
        hour = session_open_hour(con, asset)
        if hour is None:
            unknown.append(asset)
            continue
        # The US session of the previous day is over AND Europe has not started:
        # 20:00 through 06:59 UTC. That short gap is the mechanism being tested.
        (inside if (hour >= US_CLOSE_UTC or hour < EUROPE_OPEN_UTC)
         else outside).append(asset)
    return inside, outside, unknown


def main():
    if not os.path.exists(DB):
        print("no hourly store at %s; the class cannot be assigned by rule" % DB)
        return 1
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    try:
        inside, outside, unknown = classify(con, list(config.FULL_ASSET_MAP))
    finally:
        con.close()
    blob = {"rule": "session opens at or after %02d:00 UTC, or before %02d:00 UTC"
                    % (US_CLOSE_UTC, EUROPE_OPEN_UTC),
            "frozen": "2026-09-12", "in_class": sorted(inside),
            "out_of_class": sorted(outside), "no_hourly_bars": sorted(unknown)}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)
    print("in class     : %d" % len(inside))
    print("out of class : %d" % len(outside))
    print("unassignable : %d (no hourly bars)" % len(unknown))
    print("sample in    : %s" % ", ".join(sorted(inside)[:12]))
    print("written to %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
