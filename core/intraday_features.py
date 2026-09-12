"""The causal feature frame for one asset's hourly bars.

Every feature at row h is known at the close of bar h. `lab_sess_close` is the
one future column in the frame; the label builders read it and FEATURES never
names it. tests/test_intraday_features.py cuts the history at a row and checks
that row's features do not move.
"""
import numpy as np
import pandas as pd

from core.intraday import session_table

SIGMA_H_SESSIONS = 5
SIGMA_D_SESSIONS = 20
MIN_SIGMA_H_OBS = 10
MIN_SIGMA_D_OBS = 10

FEATURES = ["f_pos", "f_dow", "f_r1", "f_r3", "f_r6", "f_r_open", "f_gap",
            "f_sigma_h", "f_rv6", "f_bar_range", "f_path_eff", "f_loc",
            "f_prev_ret", "f_prev_range", "f_prev_loc", "f_d_prev_hi",
            "f_d_prev_lo", "f_vol_rel", "f_trend5"]


def _sigma_h(df):
    """Std of hourly log returns over the previous 5 sessions plus the current
    session up to and including this bar. Cumulative sums, so it is O(n)."""
    r = df["r"].to_numpy(dtype=float)
    ok = ~np.isnan(r)
    r0 = np.where(ok, r, 0.0)
    s1 = np.concatenate([[0.0], np.cumsum(r0)])
    s2 = np.concatenate([[0.0], np.cumsum(r0 * r0)])
    cnt = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    codes, uniq = pd.factorize(df["session"])
    first = np.searchsorted(codes, np.arange(len(uniq)))
    start_code = codes - SIGMA_H_SESSIONS
    valid = start_code >= 0
    start = np.where(valid, first[np.clip(start_code, 0, None)], 0)
    i = np.arange(len(df))
    n = cnt[i + 1] - cnt[start]
    m1 = (s1[i + 1] - s1[start]) / np.maximum(n, 1.0)
    m2 = (s2[i + 1] - s2[start]) / np.maximum(n, 1.0)
    sig = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    return np.where(valid & (n >= MIN_SIGMA_H_OBS) & (sig > 0), sig, np.nan)


def build_frame(df):
    fr = df.copy()
    grp = fr.groupby("session", sort=False)
    fr["r"] = np.log(fr["close"] / grp["close"].shift(1))
    fr["sigma_h"] = _sigma_h(fr)

    st = session_table(fr)
    sess_ret = np.log(st["close"] / st["close"].shift(1))
    st["sigma_d"] = sess_ret.shift(1).rolling(
        SIGMA_D_SESSIONS, min_periods=MIN_SIGMA_D_OBS).std()
    st["prev_close"] = st["close"].shift(1)
    st["prev_hi"] = st["high"].shift(1)
    st["prev_lo"] = st["low"].shift(1)
    st["prev_ret"] = sess_ret.shift(1)
    st["trend5"] = np.log(st["close"].shift(1) / st["close"].shift(6))
    st["len_med"] = st["n"].shift(1).rolling(20, min_periods=5).median()
    for c in ("sigma_d", "prev_close", "prev_hi", "prev_lo", "prev_ret",
              "trend5", "len_med"):
        fr[c] = fr["session"].map(st[c])

    fr["sess_open"] = grp["open"].transform("first")
    fr["lab_sess_close"] = grp["close"].transform("last")   # FUTURE: labels only

    sh, sd = fr["sigma_h"], fr["sigma_d"]
    fr["f_pos"] = fr["bar_idx"] / fr["len_med"]
    fr["f_dow"] = pd.to_datetime(fr["session"]).dt.dayofweek.astype(float)
    fr["f_r1"] = fr["r"] / sh
    fr["f_r3"] = np.log(fr["close"] / grp["close"].shift(3)) / sh
    fr["f_r6"] = np.log(fr["close"] / grp["close"].shift(6)) / sh
    fr["f_r_open"] = np.log(fr["close"] / fr["sess_open"]) / sh
    fr["f_gap"] = np.log(fr["sess_open"] / fr["prev_close"]) / sh
    fr["f_sigma_h"] = sh
    fr["f_rv6"] = fr["r"].rolling(6, min_periods=3).std() / sh
    fr["f_bar_range"] = np.log(fr["high"] / fr["low"]) / sh

    step = fr["r"].where(fr["bar_idx"] > 0, np.log(fr["close"] / fr["open"])).abs()
    path = step.groupby(fr["session"]).cumsum()
    fr["f_path_eff"] = (np.log(fr["close"] / fr["sess_open"]).abs() / path).mask(path == 0, 0.0)
    hi = grp["high"].cummax()
    lo = grp["low"].cummin()
    fr["f_loc"] = ((fr["close"] - lo) / (hi - lo)).mask(hi == lo, 0.5)

    prng = fr["prev_hi"] - fr["prev_lo"]
    fr["f_prev_ret"] = fr["prev_ret"] / sd
    fr["f_prev_range"] = np.log(fr["prev_hi"] / fr["prev_lo"]) / sd
    fr["f_prev_loc"] = ((fr["prev_close"] - fr["prev_lo"]) / prng).mask(prng == 0, 0.5)
    fr["f_d_prev_hi"] = np.log(fr["close"] / fr["prev_hi"]) / sh
    fr["f_d_prev_lo"] = np.log(fr["close"] / fr["prev_lo"]) / sh

    vm = fr.groupby("bar_idx")["volume"].transform(
        lambda v: v.shift(1).rolling(5, min_periods=1).mean())
    fr["f_vol_rel"] = (fr["volume"] / vm).mask(vm == 0, 0.0)
    fr["f_trend5"] = fr["trend5"] / sd
    return fr
