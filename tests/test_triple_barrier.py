"""triple_barrier label: intrabar k*sigma barriers over an H-bar horizon.

Entry is bar index 5 in every case: the base close oscillates +-2% for bars 0..5
(so the causal EWM sigma is finite and positive by index 5, ~0.0166 at vol_window=3),
then is flat. Barriers at k=1 are then upper ~101.67 / lower ~98.33 from close[5]=100,
and post-entry high/low wicks are set explicitly to isolate the intended touch.
Constructions verified against the implementation before this plan was finalized.
"""
import numpy as np
import pandas as pd

from core.features import make_target

# bars 0..5 oscillate, then flat - finite positive sigma at the entry bar (5)
BASE = [100, 102, 100, 102, 100, 100, 100, 100, 100, 100]


def _s(vals):
    return pd.Series([float(v) for v in vals])


def test_upper_barrier_touched_first_is_1():
    close = _s(BASE)
    high = _s([100, 102, 100, 102, 100, 100, 110, 100, 100, 100])  # up wick at j=6
    low = _s(BASE)
    t = make_target(close, "triple_barrier", high=high, low=low,
                    horizon=3, barrier_k=1.0, vol_window=3)
    assert t.iloc[5] == 1.0


def test_lower_barrier_touched_first_is_0():
    close = _s(BASE)
    high = _s(BASE)
    low = _s([100, 102, 100, 102, 100, 100, 90, 100, 100, 100])   # down wick at j=6
    t = make_target(close, "triple_barrier", high=high, low=low,
                    horizon=3, barrier_k=1.0, vol_window=3)
    assert t.iloc[5] == 0.0


def test_vertical_barrier_uses_sign_of_horizon_return():
    # k=5 - barriers unreachable; label = sign(close[5+3] - close[5]).
    close = _s([100, 102, 100, 102, 100, 100, 100.1, 100.2, 100.3, 100.4])
    high = close.copy()
    low = close.copy()
    t = make_target(close, "triple_barrier", high=high, low=low,
                    horizon=3, barrier_k=5.0, vol_window=3)
    assert t.iloc[5] == (1.0 if close.iloc[8] > close.iloc[5] else 0.0)
    assert t.iloc[5] == 1.0  # close[8]=100.3 > close[5]=100


def test_intrabar_touch_is_via_high_low_not_close():
    # close never leaves 100 after entry; only the high wick crosses the upper barrier.
    close = _s(BASE)
    high = _s([100, 102, 100, 102, 100, 100, 110, 100, 100, 100])  # wick at j=6
    low = _s(BASE)
    withwick = make_target(close, "triple_barrier", high=high, low=low,
                           horizon=3, barrier_k=1.0, vol_window=3)
    # same close, no wick (fall back to close): no touch - vertical barrier,
    # close[8]=100 == close[5]=100 - sign false - 0. Proves high/low drove the touch.
    noclose = make_target(close, "triple_barrier", high=None, low=None,
                          horizon=3, barrier_k=1.0, vol_window=3)
    assert withwick.iloc[5] == 1.0
    assert noclose.iloc[5] == 0.0


def test_sigma_has_no_lookahead():
    close = _s([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
    high = close.copy()
    low = close.copy()
    base = make_target(close, "triple_barrier", high=high, low=low,
                       horizon=2, barrier_k=1.0, vol_window=3)
    # mutate the LAST bar; labels whose forward window (t+2) excludes it (t <= 6)
    # must be unchanged, because sigma[t] uses only returns up to t.
    close2 = close.copy()
    close2.iloc[9] = 999.0
    var = make_target(close2, "triple_barrier", high=close2.copy(), low=close2.copy(),
                      horizon=2, barrier_k=1.0, vol_window=3)
    pd.testing.assert_series_equal(base.iloc[:7], var.iloc[:7])


def test_warmup_and_tail_are_nan():
    close = _s([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
    high = close.copy()
    low = close.copy()
    t = make_target(close, "triple_barrier", high=high, low=low,
                    horizon=2, barrier_k=1.0, vol_window=3)
    assert np.isnan(t.iloc[0])   # sigma warm-up (min_periods=vol_window=3)
    assert np.isnan(t.iloc[1])
    assert np.isnan(t.iloc[-1])  # last bar has no next bar


def test_same_bar_tiebreak_uses_close_sign():
    # both barriers pierced on the same bar j=6; close[6]=100 == close[5]=100 - 1
    close = _s(BASE)
    high = _s([100, 102, 100, 102, 100, 100, 115, 100, 100, 100])
    low = _s([100, 102, 100, 102, 100, 100, 85, 100, 100, 100])
    t = make_target(close, "triple_barrier", high=high, low=low,
                    horizon=3, barrier_k=1.0, vol_window=3)
    assert t.iloc[5] == 1.0


def test_falls_back_to_close_when_high_low_missing():
    close = _s([100, 102, 100, 102, 100, 100, 106, 100, 100, 100])  # close[6]=106
    t = make_target(close, "triple_barrier", high=None, low=None,
                    horizon=3, barrier_k=1.0, vol_window=3)
    assert t.iloc[5] == 1.0  # 106 >= upper ~101.67 via close fallback
