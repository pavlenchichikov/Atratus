"""Label rules on paths where the answer is known in advance."""
import numpy as np
import pandas as pd

from core.intraday import sessionize
from core.intraday_features import build_frame
from core.intraday_labels import (
    BUILDERS,
    first_touch_labels,
    fit_k,
    label_t1,
    label_t2,
    label_t3,
    label_t4,
)


def _bars(n_sessions=40, per=8, seed=0):
    rng = np.random.default_rng(seed)
    out, price = [], 100.0
    day0 = pd.Timestamp("2024-01-01", tz="UTC")
    for d in range(n_sessions):
        t0 = day0 + pd.Timedelta(days=d, hours=14)
        for h in range(per):
            o = price
            c = o * float(np.exp(rng.normal(0, 0.005)))
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.002)))
            out.append({"ts": (t0 + pd.Timedelta(hours=h)).isoformat(), "open": o,
                        "high": hi, "low": lo, "close": c, "volume": 100.0})
            price = c
    return out


def _arr(*xs):
    return np.array(xs, dtype=float)


def test_first_touch_up_first():
    # from bar 0 (close 100, sigma 1%): bar 1 reaches 101.5, bar 2 falls to 98
    y, amb = first_touch_labels(_arr(100, 100, 100), _arr(100, 101.5, 100),
                                _arr(100, 99.5, 98), _arr(100, 100, 99), _arr(.01, .01, .01))
    assert y[0] == 1 and not amb[0]


def test_first_touch_down_first():
    y, _ = first_touch_labels(_arr(100, 100, 100), _arr(100, 100.5, 102),
                              _arr(100, 98.5, 100), _arr(100, 100, 101), _arr(.01, .01, .01))
    assert y[0] == 0


def test_neither_level_falls_back_to_the_sign_at_the_close():
    y, _ = first_touch_labels(_arr(100, 100, 100), _arr(100, 100.4, 100.4),
                              _arr(100, 99.6, 99.6), _arr(100, 100.2, 100.3), _arr(.01, .01, .01))
    assert y[0] == 1


def test_both_levels_in_one_bar_follow_the_ohlc_path():
    # bar 1 spans both levels. Up bar (close >= open): low came first, label 0.
    y, amb = first_touch_labels(_arr(100, 99.8, 100), _arr(100, 101.5, 100),
                                _arr(100, 98.5, 100), _arr(100, 100.9, 100), _arr(.01, .01, .01))
    assert amb[0] and y[0] == 0
    # down bar: high came first, label 1
    y, amb = first_touch_labels(_arr(100, 100.9, 100), _arr(100, 101.5, 100),
                                _arr(100, 98.5, 100), _arr(100, 99.8, 100), _arr(.01, .01, .01))
    assert amb[0] and y[0] == 1


def test_t1_drops_each_session_last_bar_and_labels_against_the_close():
    fr = build_frame(sessionize(_bars(), "UTC"))
    d = label_t1(fr)
    assert len(d) == int((~fr["is_last"] & fr["sigma_h"].notna()).sum())
    keep = fr[~fr["is_last"] & fr["sigma_h"].notna()].reset_index(drop=True)
    assert (d["y"] == (keep["lab_sess_close"] > keep["close"]).astype(int)).all()
    assert d["base"].notna().all()


def test_t2_rows_match_t1_rows_and_carry_the_ambiguity_flag():
    fr = build_frame(sessionize(_bars(), "UTC"))
    d1, d2 = label_t1(fr), label_t2(fr)
    assert len(d1) == len(d2)
    assert set(d2["y"].unique()) <= {0, 1}
    assert d2["_ambiguous"].dtype == bool


def test_fit_k_reads_train_sessions_only():
    # 60 sessions: in the first 40 (train) a move of 1 sigma is reached half the
    # time; in the last 20 it is always reached. k must come from train alone.
    n = 60
    o = np.full(n, 100.0)
    sd = np.full(n, 0.01)
    first_ext = np.full(n, 100.0)
    # 100.27, not 100.3: a level landing exactly on a grid point would make the
    # answer depend on float rounding
    rest_ext = np.where(np.arange(n) % 2 == 0, 101.2, 100.27)
    rest_ext[40:] = 110.0
    train = np.arange(n) < 40
    k = fit_k(o, sd, first_ext, rest_ext, 1, train)
    assert k is not None and 0.25 < k <= 1.2


def test_fit_k_refuses_too_few_sessions():
    n = 10
    assert fit_k(np.full(n, 100.0), np.full(n, .01), np.full(n, 100.0),
                 np.full(n, 101.0), 1, np.ones(n, dtype=bool)) is None


def test_t3_drops_sessions_where_the_first_bar_already_reached_the_level():
    fr = build_frame(sessionize(_bars(n_sessions=120), "UTC"))
    d = label_t3(fr, cutoff="2024-03-15")
    assert len(d) > 0
    assert {"f_side", "f_lvl_dist", "_k", "y", "base"} <= set(d.columns)
    # a kept row's level is still ahead of the first bar's close
    assert (d["f_lvl_dist"] > 0).all()
    assert d["base"].between(0, 1).all()


def test_t4_skips_gap_opens_and_last_bar_touches():
    fr = build_frame(sessionize(_bars(n_sessions=120), "UTC"))
    d = label_t4(fr)
    assert len(d) > 0
    assert set(d["f_side"].unique()) <= {1.0, -1.0}
    # at most one touch per session per side
    assert not d.duplicated(["session", "f_side"]).any()
    assert (d["base"] == d["f_lvl_dist"]).all()
    # never the session's last bar: its label would be known at prediction time
    sizes = fr.groupby("session").size()
    assert (d["_bar_idx"].to_numpy() < sizes[d["session"]].to_numpy() - 1).all()
    # never a session that opened beyond the level: no touch from inside
    first = fr.groupby("session")[["sess_open", "prev_hi", "prev_lo"]].first()
    res, sup = d[d["f_side"] == 1.0], d[d["f_side"] == -1.0]
    assert (first.loc[res["session"], "sess_open"].to_numpy()
            < first.loc[res["session"], "prev_hi"].to_numpy()).all()
    assert (first.loc[sup["session"], "sess_open"].to_numpy()
            > first.loc[sup["session"], "prev_lo"].to_numpy()).all()


def test_t4_label_is_the_session_close_beyond_the_level():
    fr = build_frame(sessionize(_bars(n_sessions=120), "UTC"))
    d = label_t4(fr)
    res = d[d["f_side"] == 1.0]
    closes = fr.groupby("session")["lab_sess_close"].first()
    prev_hi = fr.groupby("session")["prev_hi"].first()
    expect = (closes[res["session"]].to_numpy() > prev_hi[res["session"]].to_numpy()).astype(int)
    assert (res["y"].to_numpy() == expect).all()


def test_every_target_has_a_builder():
    assert set(BUILDERS) == {"T1", "T2", "T3", "T4"}
