"""The hourly series must be the same shape the daily fitters already take:
same keys, same per-bar alignment, and it has to survive split_series."""
import datetime as dt

import numpy as np
import pandas as pd

from core.intraday_series import build_hourly_series, daily_prob_map


def _bars(n_sessions=6, per=8, seed=0):
    rng = np.random.default_rng(seed)
    out, price = [], 100.0
    day0 = pd.Timestamp("2026-01-05", tz="UTC")          # a Monday
    for d in range(n_sessions):
        t0 = day0 + pd.Timedelta(days=d, hours=14)
        for h in range(per):
            o = price
            c = o * float(np.exp(rng.normal(0, 0.004)))
            hi = max(o, c) * 1.001
            lo = min(o, c) * 0.999
            out.append({"ts": (t0 + pd.Timedelta(hours=h)).isoformat(), "open": o,
                        "high": hi, "low": lo, "close": c, "volume": 100.0})
            price = c
    return out


def _sessions(bars):
    return sorted({b["ts"][:10] for b in bars})


def _probs(bars, value=None):
    return {s: (value if value is not None else 0.5 + 0.01 * i)
            for i, s in enumerate(_sessions(bars))}


def test_every_per_bar_array_has_one_length(tmp_path):
    bars = _bars()
    s = build_hourly_series(bars, "UTC", "BTC", _probs(bars), 0.55, 0.45)
    n = len(s["probs"])
    for key in ("next_ret", "atr", "taleb_hi", "dates", "open", "high", "low", "close"):
        assert len(s[key]) == n, key
    assert n > 0


def test_the_daily_call_is_held_through_its_session():
    bars = _bars()
    probs = _probs(bars)
    s = build_hourly_series(bars, "UTC", "BTC", probs, 0.55, 0.45)
    df = pd.DataFrame({"session": s["sessions"], "prob": s["probs"]})
    for session, part in df.groupby("session"):
        assert part["prob"].nunique() == 1
        assert part["prob"].iloc[0] == probs[session]


def test_a_session_without_a_daily_call_is_dropped():
    """An asset that did not trade that day has no call to act on, and carrying
    the previous one forward would invent a signal nobody issued."""
    bars = _bars()
    probs = _probs(bars)
    dropped = max(probs)
    del probs[dropped]
    s = build_hourly_series(bars, "UTC", "BTC", probs, 0.55, 0.45)
    assert dropped not in set(s["sessions"])


def test_the_atr_warm_up_is_cut_so_every_bar_is_finite():
    bars = _bars()
    s = build_hourly_series(bars, "UTC", "BTC", _probs(bars), 0.55, 0.45)
    assert np.isfinite(s["atr"]).all() and (s["atr"] > 0).all()


def test_next_ret_is_the_forward_hourly_return():
    bars = _bars()
    s = build_hourly_series(bars, "UTC", "BTC", _probs(bars), 0.55, 0.45)
    close = s["close"]
    expect = close[1] / close[0] - 1
    assert abs(s["next_ret"][0] - expect) < 1e-12
    assert np.isnan(s["next_ret"][-1])          # nothing follows the last bar


def test_too_little_history_gives_nothing():
    bars = _bars(n_sessions=1, per=4)
    assert build_hourly_series(bars, "UTC", "BTC", _probs(bars), 0.55, 0.45) is None


def test_the_series_survives_the_daily_splitter():
    """split_series cuts the per-bar arrays it knows and copies the scalars. A
    shape it does not recognise would silently reach a fitter half-sliced."""
    import train_timing as tt
    bars = _bars(n_sessions=10)
    s = build_hourly_series(bars, "UTC", "BTC", _probs(bars), 0.55, 0.45,
                            risky=True, is_forex=False)
    train, val, test = tt.split_series(s)
    n = len(s["probs"])
    assert len(train["probs"]) + len(val["probs"]) + len(test["probs"]) == n
    for part in (train, val, test):
        assert part["buy_thr"] == 0.55 and part["sell_thr"] == 0.45
        assert part["risky"] is True
        assert len(part["atr"]) == len(part["probs"]) == len(part["close"])


def test_daily_prob_map_joins_dates_to_sessions():
    daily = {"dates": np.array([np.datetime64("2026-01-05"), np.datetime64("2026-01-06")]),
             "probs": np.array([0.61, 0.42])}
    assert daily_prob_map(daily) == {"2026-01-05": 0.61, "2026-01-06": 0.42}


def test_an_exchange_zone_groups_by_the_local_day():
    """A New York session that runs past midnight UTC is one session, and the
    daily call is keyed by the exchange's date, not the UTC one."""
    base = dt.datetime(2026, 1, 5, 20, 0, tzinfo=dt.UTC)   # 15:00 in NY
    bars = []
    price = 100.0
    for h in range(20):
        ts = base + dt.timedelta(hours=h)
        bars.append({"ts": ts.isoformat(), "open": price, "high": price * 1.001,
                     "low": price * 0.999, "close": price, "volume": 1.0})
    s = build_hourly_series(bars, "America/New_York", "AAPL",
                            {"2026-01-05": 0.6, "2026-01-06": 0.4}, 0.55, 0.45,
                            atr_period=3)
    assert set(s["sessions"]) <= {"2026-01-05", "2026-01-06"}
    assert "2026-01-05" in set(s["sessions"])
