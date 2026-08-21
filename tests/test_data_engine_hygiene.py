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
