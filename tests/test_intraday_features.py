"""The feature frame. The truncation test is the one that matters: a feature
that reads any later bar changes when the history is cut at its own row."""
import numpy as np
import pandas as pd

from core.intraday import sessionize
from core.intraday_features import FEATURES, SIGMA_H_SESSIONS, build_frame


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
                        "high": hi, "low": lo, "close": c, "volume": 100.0 + h})
            price = c
    return out


def test_no_feature_reads_a_later_bar():
    bars = _bars()
    full = build_frame(sessionize(bars, "UTC"))
    # rows past the sigma warm-up, at bar_idx >= 2 so a truncated session still
    # has the three bars sessionize requires
    rows = full.index[(full["bar_idx"] >= 2) & full["sigma_h"].notna()]
    for i in list(rows[::37])[:12]:
        cut = build_frame(sessionize(bars[: i + 1], "UTC"))
        a = full.loc[i, FEATURES].to_numpy(dtype=float)
        b = cut.iloc[-1][FEATURES].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), (i, list(zip(FEATURES, a, b)))


def test_the_future_session_close_is_never_a_feature():
    assert "lab_sess_close" not in FEATURES
    assert not any(f.startswith(("lab_", "_")) for f in FEATURES)


def test_sigma_h_waits_for_five_previous_sessions():
    fr = build_frame(sessionize(_bars(), "UTC"))
    first_sessions = sorted(fr["session"].unique())[:SIGMA_H_SESSIONS]
    assert fr.loc[fr["session"].isin(first_sessions), "sigma_h"].isna().all()
    assert fr["sigma_h"].notna().any()


def test_sigma_h_matches_a_direct_computation():
    fr = build_frame(sessionize(_bars(), "UTC"))
    i = fr.index[fr["sigma_h"].notna()][50]
    sessions = sorted(fr["session"].unique())
    k = sessions.index(fr.loc[i, "session"])
    window = fr[(fr["session"] >= sessions[k - SIGMA_H_SESSIONS]) & (fr.index <= i)]
    expect = window["r"].dropna().std(ddof=0)
    assert abs(fr.loc[i, "sigma_h"] - expect) < 1e-12


def test_sigma_d_is_one_value_per_session():
    fr = build_frame(sessionize(_bars(), "UTC"))
    assert (fr.groupby("session")["sigma_d"].nunique(dropna=False) <= 1).all()


def test_the_frame_carries_every_feature_column():
    fr = build_frame(sessionize(_bars(), "UTC"))
    assert set(FEATURES) <= set(fr.columns)
    assert len(FEATURES) == 19
