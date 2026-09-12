"""Hourly bars grouped into exchange sessions. Pure: no sockets, no databases.

Everything intraday is built from these sessions, including "yesterday": the
daily rows in market.db are never mixed in, because for forex, futures and
crypto the vendor's daily and hourly series do not reconcile (measured
2026-09-08 on the intraday spike).
"""
import pandas as pd

import config

# A one- or two-bar session has no path to speak of: most features and every
# label below need at least one bar after the one being predicted from.
MIN_SESSION_BARS = 3

_ROUND_THE_CLOCK = ("CRYPTO", "FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC")
_COLS = ["ts", "open", "high", "low", "close", "volume"]


def session_zone(asset, tz_name):
    """Crypto and forex trade round the clock; their session is the UTC day."""
    for group in _ROUND_THE_CLOCK:
        if asset in config.ASSET_TYPES.get(group, ()):
            return "UTC"
    return tz_name


def sessionize(bars, tz_name):
    """Bars as a frame with session, bar_idx and is_last; short sessions dropped."""
    df = pd.DataFrame(list(bars), columns=_COLS)
    if df.empty:
        return df.assign(session=pd.Series(dtype=str), bar_idx=pd.Series(dtype=int),
                         is_last=pd.Series(dtype=bool))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in _COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    df = (df.dropna(subset=["open", "high", "low", "close"])
            .sort_values("ts").drop_duplicates("ts"))
    df["session"] = df["ts"].dt.tz_convert(tz_name).dt.strftime("%Y-%m-%d")
    size = df.groupby("session")["close"].transform("size")
    df = df[size >= MIN_SESSION_BARS].reset_index(drop=True)
    df["bar_idx"] = df.groupby("session").cumcount()
    df["is_last"] = df["bar_idx"] == df.groupby("session")["close"].transform("size") - 1
    return df


def session_table(df):
    """One row per session: open, high, low, close, bar count. Sorted by date."""
    g = df.groupby("session", sort=True)
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last(),
                         "n": g["close"].size()})
