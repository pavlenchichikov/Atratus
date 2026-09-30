import os

import pandas as pd

import train_global as tg
from core.panel import EMBARGO_DAYS, date_folds


def test_the_gap_covers_the_label_horizon():
    """A 20-bar label on the last training date reads 20 closes ahead: a 5-day
    gap would let it resolve inside the test window."""
    assert tg.embargo_days(1) == EMBARGO_DAYS
    assert tg.embargo_days(20) >= 20 + 1
    dates = pd.bdate_range("2020-01-01", periods=600)
    for train, test in date_folds(dates, n_folds=3, embargo=tg.embargo_days(20)):
        gap = len(pd.bdate_range(train[-1], test[0])) - 2
        assert gap >= 20


def test_a_horizon_sets_the_label_and_its_own_report(monkeypatch):
    monkeypatch.setenv("GTRADE_LABEL_MODE", "x")
    monkeypatch.setenv("GTRADE_LABEL_HORIZON", "x")
    tg.use_horizon(20)
    assert os.environ["GTRADE_LABEL_MODE"] == "direction_h"
    assert os.environ["GTRADE_LABEL_HORIZON"] == "20"
    assert tg.report_path(20).endswith("global_report_h20.json")
    tg.use_horizon(1)
    assert os.environ["GTRADE_LABEL_MODE"] == "direction"
    assert "GTRADE_LABEL_HORIZON" not in os.environ
    assert tg.report_path(1) == tg.REPORT


def test_horizons_parse_and_reject_nonsense():
    assert tg.parse_horizons("5, 10,20") == [5, 10, 20]
    assert tg.parse_horizons("") == [1]
    for bad in ("0", "x", "-5"):
        try:
            tg.parse_horizons(bad)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % bad)
