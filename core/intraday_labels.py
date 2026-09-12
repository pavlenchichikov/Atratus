"""The four intraday targets and their baseline scores.

Every builder returns the target's feature columns plus `session`, `y` (0/1)
and `base` (the trivial baseline's score, higher means label 1 more likely).
Columns starting with "_" are diagnostics and are never fitted on.
"""
from itertools import pairwise

import numpy as np
import pandas as pd

from core.intraday_features import FEATURES

EXTRA = ["f_side", "f_lvl_dist"]
TARGET_FEATURES = {"T1": FEATURES, "T2": FEATURES,
                   "T3": FEATURES + EXTRA, "T4": FEATURES + EXTRA}


def _keep(fr):
    """Rows that can carry a label: not a session's last bar, sigma known."""
    return (~fr["is_last"] & fr["sigma_h"].notna()).to_numpy()


def label_t1(fr, cutoff=None):
    """T1: will the session close above this bar's close."""
    d = fr[_keep(fr)]
    out = d[FEATURES + ["session"]].copy()
    out["y"] = (d["lab_sess_close"] > d["close"]).astype(int)
    out["base"] = d["f_r1"].fillna(0.0)
    out["_fwd"] = np.log(d["lab_sess_close"] / d["close"]) / d["sigma_h"]
    return out.reset_index(drop=True)


def first_touch_labels(o, h, l, c, s):
    """For every bar of ONE session: 1 if close*(1+s) is reached before
    close*(1-s) by a later bar of the session, else 0.

    Neither reached: the sign of (session close - this close). Both inside one
    later bar: the standard OHLC path, an up bar is open-low-high-close so the
    lower level came first (0), a down bar the reverse (1); flagged ambiguous.
    """
    n = len(c)
    up = c * (1 + s)
    dn = c * (1 - s)
    later = np.triu(np.ones((n, n), dtype=bool), 1)
    hit_u = (h[None, :] >= up[:, None]) & later
    hit_d = (l[None, :] <= dn[:, None]) & later
    ju = np.where(hit_u.any(1), hit_u.argmax(1), n)
    jd = np.where(hit_d.any(1), hit_d.argmax(1), n)
    y = np.where(ju < jd, 1.0, 0.0)
    neither = (ju == n) & (jd == n)
    y[neither] = (c[-1] > c[neither]).astype(float)
    both = (ju == jd) & (ju < n)
    jb = ju[both]
    y[both] = np.where(c[jb] >= o[jb], 0.0, 1.0)
    return y, both


def label_t2(fr, cutoff=None):
    """T2: which of close +/- 1 sigma_h is touched first within the session."""
    o, h, l, c, s = (fr[k].to_numpy(dtype=float)
                     for k in ("open", "high", "low", "close", "sigma_h"))
    y = np.zeros(len(fr))
    amb = np.zeros(len(fr), dtype=bool)
    starts = np.flatnonzero(fr["bar_idx"].to_numpy() == 0).tolist() + [len(fr)]
    for a, b in pairwise(starts):
        y[a:b], amb[a:b] = first_touch_labels(o[a:b], h[a:b], l[a:b], c[a:b], s[a:b])
    keep = _keep(fr)
    out = fr.loc[keep, FEATURES + ["session"]].copy()
    out["y"] = y[keep].astype(int)
    out["base"] = fr.loc[keep, "f_r1"].fillna(0.0).to_numpy()
    out["_ambiguous"] = amb[keep]
    return out.reset_index(drop=True)


K_GRID = np.round(np.arange(0.10, 3.0001, 0.05), 2)
MIN_K_SESSIONS = 30
_T3_OUT = TARGET_FEATURES["T3"] + ["session", "y", "base", "_k"]


def fit_k(o, sd, first_ext, rest_ext, side, train):
    """The k whose level open*(1 + side*k*sigma_d) is reached in closest to half
    of the TRAIN sessions where the first bar had not already reached it.
    None when fewer than MIN_K_SESSIONS train sessions qualify at every k."""
    best, best_gap = None, None
    for k in K_GRID:
        level = o * (1 + side * k * sd)
        ahead = (first_ext < level) if side == 1 else (first_ext > level)
        valid = train & ahead
        if valid.sum() < MIN_K_SESSIONS:
            continue
        reached = (rest_ext >= level) if side == 1 else (rest_ext <= level)
        gap = abs(reached[valid].mean() - 0.5)
        if best_gap is None or gap < best_gap:
            best, best_gap = float(k), gap
    return best


def label_t3(fr, cutoff):
    """T3: after the first bar, will the session reach open +/- k*sigma_d."""
    first = fr[fr["bar_idx"] == 0].set_index("session")
    rest = (fr[fr["bar_idx"] > 0].groupby("session")
              .agg(rest_hi=("high", "max"), rest_lo=("low", "min")))
    ses = first.join(rest, how="inner")
    ses = ses[ses["sigma_d"].notna() & ses["sigma_h"].notna()]
    o = ses["sess_open"].to_numpy(dtype=float)
    sd = ses["sigma_d"].to_numpy(dtype=float)
    train = np.asarray(ses.index < cutoff)
    parts = []
    for side in (1, -1):
        first_ext = (ses["high"] if side == 1 else ses["low"]).to_numpy(dtype=float)
        rest_ext = (ses["rest_hi"] if side == 1 else ses["rest_lo"]).to_numpy(dtype=float)
        k = fit_k(o, sd, first_ext, rest_ext, side, train)
        if k is None:
            continue
        level = o * (1 + side * k * sd)
        ahead = (first_ext < level) if side == 1 else (first_ext > level)
        reached = (rest_ext >= level) if side == 1 else (rest_ext <= level)
        past = pd.Series(np.where(ahead, reached.astype(float), np.nan), index=ses.index)
        base = past.shift(1).rolling(20, min_periods=5).mean().to_numpy()
        d = ses[ahead].copy()
        d["f_side"] = float(side)
        d["f_lvl_dist"] = side * np.log(level[ahead] / d["close"].to_numpy()) / d["sigma_h"]
        d["y"] = reached[ahead].astype(int)
        d["base"] = base[ahead]
        d["_k"] = k
        parts.append(d.reset_index()[_T3_OUT])
    if not parts:
        return pd.DataFrame(columns=_T3_OUT)
    return pd.concat(parts).dropna(subset=["base"]).reset_index(drop=True)


def label_t4(fr, cutoff=None):
    """T4: at the first touch of yesterday's high (or low), does the session
    close beyond it (break, 1) or back inside (bounce, 0)."""
    parts = []
    for side, col in ((1, "prev_hi"), (-1, "prev_lo")):
        lvl = fr[col]
        if side == 1:
            inside_open, touch = fr["sess_open"] < lvl, fr["high"] >= lvl
        else:
            inside_open, touch = fr["sess_open"] > lvl, fr["low"] <= lvl
        t = fr[inside_open & touch].groupby("session", sort=False).head(1)
        t = t[~t["is_last"] & t["sigma_h"].notna()]
        level = t[col]
        d = t[FEATURES + ["session"]].copy()
        d["f_side"] = float(side)
        d["f_lvl_dist"] = side * np.log(t["close"] / level) / t["sigma_h"]
        beyond = (t["lab_sess_close"] > level) if side == 1 else (t["lab_sess_close"] < level)
        d["y"] = beyond.astype(int)
        d["base"] = d["f_lvl_dist"]
        d["_bar_idx"] = t["bar_idx"]
        parts.append(d)
    return pd.concat(parts).reset_index(drop=True)


BUILDERS = {"T1": label_t1, "T2": label_t2, "T3": label_t3, "T4": label_t4}
