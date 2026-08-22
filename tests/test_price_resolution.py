"""A price quoted too coarsely carries no one-bar sign, and must be refused.

Measured 2026-08-22 across 324 assets: PEPE holds 27 distinct close values over
1210 bars, SHIB 104 over 1951, so 72% and 61% of consecutive bars are identical.
Read as accuracy that came out at 5.0% and 18.6%, which looks like a broken
model and is a broken series: a tie has no direction, and the metric counts it
as a miss. Everything else in the book sits below 16%.
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core.backtesting import MAX_TIE_FRACTION, price_resolution_ok


def test_a_normally_quoted_series_passes():
    """The control: 324 assets go through this, and only two must not."""
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 500))
    ok, ties = price_resolution_ok(close)
    assert ok and ties < 0.01


def test_a_series_that_mostly_repeats_itself_is_refused():
    # 30 distinct levels over 500 bars, the shape PEPE has
    rng = np.random.default_rng(1)
    close = np.round(6e-6 * np.cumprod(1.0 + rng.normal(0, 0.05, 500)), 7)
    ok, ties = price_resolution_ok(close)
    assert not ok, "%.2f of bars repeated and it still passed" % ties
    assert ties > MAX_TIE_FRACTION


def test_the_threshold_sits_in_the_gap_the_book_actually_has():
    """PEPE 72% and SHIB 61% on one side, everything else 16% and below."""
    assert 0.20 <= MAX_TIE_FRACTION <= 0.50


def test_a_borderline_series_is_kept():
    """15% ties is STELLANTIS, a real asset nobody wants dropped."""
    close = np.arange(1000, dtype=float)
    close[::7] = close[::7] - 1          # ~14% of steps flat
    ok, ties = price_resolution_ok(close)
    assert ok, "ties %.2f" % ties


def test_too_short_to_judge_is_not_refused():
    ok, ties = price_resolution_ok([1.0])
    assert ok and ties == 0.0
    ok, _ = price_resolution_ok([])
    assert ok


def test_nan_bars_do_not_count_as_repeats():
    close = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0])
    ok, ties = price_resolution_ok(close)
    assert ok and ties == 0.0
