"""Unit tests for the shared feature chain (pure; a fake engine, no database)."""


import pandas as pd


def frame(n=60):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": 1.0, "high": 1.2, "low": 0.9,
                         "close": 1.0, "volume": 100.0}, index=idx)


