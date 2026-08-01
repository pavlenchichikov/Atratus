"""Unit tests for core.levels (pure, no I/O)."""

import pandas as pd

from core.levels import atr_abs, levels, size_for


def _bars(n=20, close=100.0, rng=2.0):
    """n synthetic bars, flat close, constant high-low range."""
    return [{"date": "d%02d" % i, "open": close, "high": close + rng / 2,
             "low": close - rng / 2, "close": close} for i in range(n)]


def test_atr_matches_the_features_formula():
    # The reference expression is copied verbatim from core/features.py:170-176
    # (normalized there by close, so multiplying by close gives the absolute
    # ATR). core/levels.py must not import the training path, hence the
    # duplicate; this test is what keeps the two from drifting apart.
    bars = []
    price = 100.0
    for i in range(40):
        price += (-1) ** i * (i % 5) * 0.3
        bars.append({"date": "d%02d" % i, "open": price, "high": price + 1.5,
                     "low": price - 1.1, "close": price})
    df = pd.DataFrame(bars)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    expected = (tr.rolling(14).mean() / (df['close'] + 1e-9)).iloc[-1] * df['close'].iloc[-1]
    assert abs(atr_abs(bars) - expected) < 1e-6


def test_atr_needs_fourteen_bars():
    assert atr_abs(_bars(13)) is None
    assert atr_abs(_bars(14)) is not None


def test_long_zone_and_stop():
    r = levels(_bars(), "BUY")
    assert r["status"] == "ok" and r["side"] == 1
    assert abs(r["atr"] - 2.0) < 1e-9           # constant 2.0 range
    assert abs(r["entry_low"] - 99.0) < 1e-9    # 100 - 0.5 * 2
    assert abs(r["entry_high"] - 101.0) < 1e-9  # 100 + 0.5 * 2
    assert abs(r["stop"] - 96.0) < 1e-9         # 100 - 2.0 * 2
    assert r["trailing"] is False


def test_short_side_is_mirrored():
    r = levels(_bars(), "SELL")
    assert r["side"] == -1
    assert abs(r["entry_low"] - 99.0) < 1e-9
    assert abs(r["entry_high"] - 101.0) < 1e-9
    assert abs(r["stop"] - 104.0) < 1e-9        # 100 + 2.0 * 2


def test_trailing_stop_takes_the_best_bar_of_the_segment():
    bars = _bars(20)
    for i in (17, 18):                           # jump up, then pull back
        for k in ("open", "high", "low", "close"):
            bars[i][k] += 6.0
    for k in ("open", "high", "low", "close"):
        bars[19][k] += 3.0
    seg = {"side": 1, "start_date": "d17", "end_date": "d19", "open": True,
           "bars": 3}
    r = levels(bars, "BUY", segment=seg)
    assert r["trailing"] is True
    # true ranges: 2.0 everywhere except tr[17]=7.0 (the gap up) and tr[19]=4.0
    # (the pullback below the previous close). So atr at bars 17 and 18 is
    # 33/14 and the stop there is 106 - 2 * 33/14; at bar 19 atr is 35/14 and
    # the stop is only 98.0. The trailing stop keeps the better one.
    assert abs(r["stop"] - (106.0 - 2.0 * (33.0 / 14.0))) < 1e-9
    assert r["stop"] > 98.0


def test_short_whose_trailing_stop_is_already_passed_is_flagged():
    # A short entered at 100 while the price walked up to 106: the trailing stop
    # stays back at the entry bar, below the current price, so it can never
    # trigger. The sheet must say the position is already through its stop.
    bars = _bars(20)
    for i in (18, 19):                           # entry bar d17 stays at 100
        for k in ("open", "high", "low", "close"):
            bars[i][k] += 6.0
    seg = {"side": -1, "start_date": "d17", "end_date": "d19", "open": True,
           "bars": 3}
    r = levels(bars, "SELL", segment=seg)
    assert r["trailing"] is True
    assert abs(r["stop"] - 104.0) < 1e-9         # the entry bar: 100 + 2 * 2.0
    assert r["stop"] < r["close"]                # unreachable for a short
    assert r["status"] == "stop_breached"


def test_a_healthy_position_is_not_flagged():
    seg = {"side": 1, "start_date": "d05", "end_date": "d19", "open": True,
           "bars": 15}
    assert levels(_bars(), "BUY", segment=seg)["status"] == "ok"


def test_wait_signal_reports_no_signal():
    r = levels(_bars(), "WAIT")
    assert r["status"] == "no_signal" and r["side"] == 0
    assert r["entry_low"] is None and r["stop"] is None


def test_empty_and_short_history():
    assert levels([], "BUY")["status"] == "no_bars"
    assert levels(_bars(5), "BUY")["status"] == "short_history"


def test_zero_range_reports_flat_atr():
    flat = [{"date": "d%02d" % i, "open": 50.0, "high": 50.0, "low": 50.0,
             "close": 50.0} for i in range(20)]
    r = levels(flat, "BUY")
    assert r["status"] == "flat_atr"
    assert r["stop"] is None


def test_size_is_risk_over_stop_distance():
    # 10000 capital, 1 percent risk (100), 4.0 to the stop: 25 units of price
    # exposure, i.e. 2500 of position at a close of 100. The cap is raised to
    # 1.0 here so the risk rule is what binds and the arithmetic is visible.
    r = size_for(close=100.0, stop=96.0, equity=10_000.0, risk_per_trade=0.01,
                 max_single_position=1.0)
    assert abs(r["amount"] - 2500.0) < 1e-6
    assert abs(r["pct"] - 0.25) < 1e-9
    assert r["bound_by"] == "risk"


def test_max_single_position_clips_and_is_named():
    r = size_for(close=100.0, stop=99.0, equity=10_000.0, risk_per_trade=0.01,
                 max_single_position=0.10)
    assert abs(r["amount"] - 1000.0) < 1e-6      # 10 percent cap, not 10000
    assert r["bound_by"] == "max_single_position"


def test_risk_binds_when_it_is_the_smallest():
    r = size_for(close=100.0, stop=50.0, equity=10_000.0, risk_per_trade=0.01,
                 max_single_position=0.10)
    assert abs(r["amount"] - 200.0) < 1e-6       # 100 risk / 50 distance * 100
    assert r["bound_by"] == "risk"


def test_kelly_binds_when_it_is_the_smallest():
    r = size_for(close=100.0, stop=50.0, equity=10_000.0, risk_per_trade=0.01,
                 max_single_position=0.10, kelly_pct=0.005)
    assert abs(r["amount"] - 50.0) < 1e-6
    assert r["bound_by"] == "kelly"


def test_unset_equity_reports_percentages_but_still_names_the_limit():
    r = size_for(close=100.0, stop=96.0, equity=0.0, risk_per_trade=0.01,
                 max_single_position=0.10)
    assert r["amount"] is None
    assert abs(r["pct"] - 0.10) < 1e-9           # still capped, still shown
    assert r["bound_by"] == "max_single_position"


def test_stop_equal_to_close_does_not_divide_by_zero():
    r = size_for(close=100.0, stop=100.0, equity=10_000.0, risk_per_trade=0.01,
                 max_single_position=0.10)
    assert r["amount"] is None and r["pct"] == 0.0
    assert r["bound_by"] == "no_stop"


def test_risk_config_carries_the_two_new_keys():
    from risk_manager import _FRACTION_KEYS, RISK_CONFIG
    assert RISK_CONFIG["equity"] == 0.0
    assert RISK_CONFIG["risk_per_trade"] == 0.01
    assert "risk_per_trade" in _FRACTION_KEYS
    assert "equity" not in _FRACTION_KEYS      # money, not a fraction
