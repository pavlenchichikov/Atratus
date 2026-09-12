"""The series train_timing's fitters take, built from HOURLY bars.

`train_timing.build_asset_series` reconstructs one asset's daily history for the
timing policy and the Q. Everything downstream of it - policy_step, the rules,
series_features, rollout, fit_q, eval_policy, split_series, gate_policy - counts
in BARS and never in days, so the same machinery runs intraday as soon as
something hands it an hourly series in the same shape. This is that something.

The probability stream is the DAILY champion's, held constant through the
session it belongs to. Direction inside the session is not predictable here
(measured 2026-09-12: pooled hourly direction AUC 0.527 against a 0.515
baseline), so the intraday layer is not given a direction call of its own. It
decides whether to act on the day's call and how long to stay, which is the
question a 0.158% hourly bar against a 0.50% round trip actually poses.

Deliberately takes the daily probabilities as an argument instead of loading a
champion: train_timing pins market.db and models/ to its own checkout root, and
a builder that reaches for them cannot be tested without both stores.
"""
import numpy as np
import pandas as pd

from core.features import compute_taleb_risk
from core.intraday import session_zone, sessionize
from core.levels import ATR_PERIOD, atr_series

TALEB_HI = 0.7

# The per-bar arrays train_timing._SLICED cuts, plus the scalars it copies.
SERIES_KEYS = ("probs", "next_ret", "atr", "taleb_hi", "dates",
               "high", "low", "close", "open")


def build_hourly_series(bars, tz_name, asset, daily_probs, buy_thr, sell_thr,
                        risky=False, is_forex=False, atr_period=ATR_PERIOD):
    """One asset's hourly series, or None when too little of it is usable.

    `bars`     oldest-first hourly bars as intraday_fetch.load returns them.
    `tz_name`  the exchange zone stored beside them; crypto and forex are
               grouped by the UTC day (see core.intraday.session_zone).
    `daily_probs`  {session date 'YYYY-MM-DD': probability}. A session with no
               daily call is dropped rather than carried forward: an asset that
               did not trade that day has no call to act on, and filling the gap
               would invent one.
    `buy_thr`, `sell_thr`  the asset's tuned thresholds, unchanged from serving.

    `next_ret` is the plain forward hourly return, so a position held into the
    close earns the overnight gap. That is what holding actually pays; charging
    the policy for a night it chose to sit through is the point of letting it
    choose.
    """
    df = sessionize(bars, session_zone(asset, tz_name))
    if not len(df):
        return None
    df = df[df["session"].isin(daily_probs)].reset_index(drop=True)
    # ATR is None until `atr_period` bars exist, and every array here has to be
    # finite for the state row; drop the warm-up rather than fill it.
    if len(df) <= atr_period:
        return None
    ohlc = df[["open", "high", "low", "close"]].to_dict("records")
    atr = atr_series(ohlc, period=atr_period)
    keep = slice(atr_period - 1, len(df))
    df = df.iloc[keep].reset_index(drop=True)
    atr = np.asarray(atr[keep], dtype=float)

    close = df["close"].to_numpy(dtype=float)
    taleb = compute_taleb_risk(df["close"])
    series = {
        "probs": df["session"].map(daily_probs).to_numpy(dtype=float),
        "next_ret": pd.Series(close).pct_change().shift(-1).to_numpy(dtype=float),
        "atr": atr,
        "taleb_hi": (taleb > TALEB_HI).fillna(False).to_numpy(),
        "dates": df["ts"].to_numpy(),
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": close,
        "buy_thr": float(buy_thr), "sell_thr": float(sell_thr),
        "risky": bool(risky), "is_forex": bool(is_forex),
        "sessions": df["session"].to_numpy(),
    }
    return series


def daily_prob_map(daily_series):
    """{date: probability} from what train_timing.build_asset_series returned.

    Its `dates` are numpy datetimes and the sessions here are 'YYYY-MM-DD'
    strings, so the join has to happen somewhere; here, once, rather than at
    every call site.
    """
    dates = pd.to_datetime(pd.Series(daily_series["dates"])).dt.strftime("%Y-%m-%d")
    return dict(zip(dates, (float(p) for p in daily_series["probs"])))
