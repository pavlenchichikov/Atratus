import numpy as np
import pandas as pd

from core import features as F


def _frame(n=100):
    d = pd.date_range("2026-01-01", periods=n, freq="B")
    c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame({"Date": d, "close": c, "high": c + 1, "low": c - 1})


def test_har_range_is_a_positive_next_day_range_forecast_without_nans():
    out = F.add_har_features(_frame())
    assert not out["har_range"].isna().any()
    assert (out["har_range"].iloc[30:] > 0).all()

