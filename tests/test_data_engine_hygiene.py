"""Non-positive prices must never reach the database.

Written after 2026-08-22, when the trade-levels gate returned `mean_d nan` over
316 assets because ONE bar of AZN had high = low = 0: a trade priced off it
returns inf, and inf - inf is nan. The gate now drops non-finite deltas, but
that is the second line. This is the first.
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from data_engine import scrub_ohlc


def _bar(o, h, l, c, v=1):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def test_clean_bars_are_returned_untouched():
    """The control. Every bar already in the database came through here."""
    df = pd.DataFrame([_bar(10.0, 12.0, 9.0, 11.0), _bar(11.0, 12.0, 10.0, 11.5)])
    out, fixed, dropped = scrub_ohlc(df)
    assert (fixed, dropped) == (0, 0)
    assert out.equals(df)


def test_a_close_only_bar_keeps_its_close_and_loses_its_range():
    """Ten real trading days looked like this, all provider glitches."""
    df = pd.DataFrame([_bar(0.0, 0.0, 0.0, 15.86)])
    out, fixed, dropped = scrub_ohlc(df)
    assert (fixed, dropped) == (1, 0)
    row = out.iloc[0]
    assert row["open"] == row["high"] == row["low"] == row["close"] == 15.86


def test_a_zero_leg_is_clamped_inside_the_prices_that_are_real():
    df = pd.DataFrame([_bar(0.16904, 0.19903, 0.0, 0.18737)])
    out, fixed, dropped = scrub_ohlc(df)
    assert (fixed, dropped) == (1, 0)
    row = out.iloc[0]
    assert row["low"] == 0.16904, "the low cannot be above the open"
    assert row["high"] == 0.19903
    assert row["close"] == 0.18737


def test_a_bar_with_no_price_at_all_is_dropped():
    """PEPE's first twelve days: below the provider's precision, so zero."""
    df = pd.DataFrame([_bar(0.0, 0.0, 0.0, 0.0, v=46385210),
                       _bar(10.0, 12.0, 9.0, 11.0)])
    out, fixed, dropped = scrub_ohlc(df)
    assert (fixed, dropped) == (0, 1)
    assert len(out) == 1 and out.iloc[0]["close"] == 11.0


def test_a_missing_price_is_treated_as_missing_not_as_zero():
    df = pd.DataFrame([_bar(None, None, None, 20.0), _bar(1.0, 2.0, 0.5, None)])
    out, fixed, dropped = scrub_ohlc(df)
    assert (fixed, dropped) == (1, 1)
    assert len(out) == 1 and out.iloc[0]["open"] == 20.0


def test_the_repaired_bar_can_no_longer_produce_an_infinite_return():
    """The actual failure, in one line: a zero low prices a trade at inf."""
    df = pd.DataFrame([_bar(0.0, 0.0, 0.0, 11500.0)])
    out, _fixed, _dropped = scrub_ohlc(df)
    row = out.iloc[0]
    assert min(row["open"], row["high"], row["low"], row["close"]) > 0


def test_flat_bar_report_names_the_assets_whose_bars_record_no_range(tmp_path, monkeypatch):
    """111 assets carry close-only bars and nothing said so: the health block
    printed "Daily: clean" while ARKVX had not recorded a range in 60 days."""
    import sqlite3

    import config
    import data_engine as de

    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    for table in ("arkvx", "btc"):
        con.execute('CREATE TABLE %s (Date TEXT, Open REAL, High REAL, '
                    'Low REAL, Close REAL)' % table)
    con.executemany('INSERT INTO arkvx VALUES (?,?,?,?,?)',
                    [("2026-09-%02d" % (i + 1), 10.0, 10.0, 10.0, 10.0) for i in range(30)])
    con.executemany('INSERT INTO btc VALUES (?,?,?,?,?)',
                    [("2026-09-%02d" % (i + 1), 10.0, 10.5, 9.5, 10.0) for i in range(30)])
    con.commit()
    con.close()

    monkeypatch.setattr(config, "FULL_ASSET_MAP", {"ARKVX": "x", "BTC": "y"})
    assert de.flat_bar_report(path) == [("ARKVX", 100, 30)]


def test_a_recent_bar_with_zero_open_high_low_is_held_not_flattened():
    """Yahoo sent 0 for open/high/low on two London sessions (2026-09-30/10-01)
    and filled them in a day later. Filled from the close and stored, the bar
    stayed flat for good, because a stored date is never fetched again. A
    recent one is left out instead, so the next run asks for it again."""
    df = pd.DataFrame({"open": [10.0, 0.0, 0.0], "high": [11.0, 0.0, 0.0],
                       "low": [9.0, 0.0, 0.0], "close": [10.5, 10.2, 10.1]},
                      index=["2026-08-03", "2026-08-04", "2026-10-01"])
    out, fixed, dropped = scrub_ohlc(df, hold_after="2026-09-24")
    assert list(out.index) == ["2026-08-03", "2026-08-04"]
    assert fixed == 1 and dropped == 1
    assert out.loc["2026-08-04", "high"] == 10.2          # an old one is still repaired
