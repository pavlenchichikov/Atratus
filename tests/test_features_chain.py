"""Unit tests for the shared feature chain (pure; a fake engine, no database)."""


import pandas as pd


def frame(n=60):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": 1.0, "high": 1.2, "low": 0.9,
                         "close": 1.0, "volume": 100.0}, index=idx)


def _weekly_frame(n=80):
    """Mondays, each carrying the close of that same week's Friday, which is how
    Yahoo stamps a 1wk bar. n is generous: w_rsi needs a 14-week warm-up and the
    chain drops those rows, which is what made the first version of this test
    assert on rows that did not exist."""
    idx = pd.date_range("2025-01-06", periods=n, freq="7D")
    return pd.DataFrame({"close": [100.0 + (i % 7) * 3.0 + i for i in range(n)]},
                        index=idx)


def _stub_read_sql(monkeypatch, weekly):
    import core.features as F
    monkeypatch.setattr(F.pd, "read_sql",
                        lambda sql, engine, **kw: weekly.copy())


def _joined(monkeypatch, weekly):
    from core.features import add_weekly_features
    idx = pd.date_range("2025-01-06", periods=7 * len(weekly), freq="D")
    daily = pd.DataFrame({"close": 1.0}, index=idx)
    out = add_weekly_features(daily, "any", engine=None)
    return out.set_index(pd.to_datetime(out["Date"]))


def test_a_daily_row_never_sees_its_own_week(monkeypatch):
    """The leak found 2026-09-08. Yahoo stamps the weekly bar on Monday and fills
    it with Friday's close, so an unshifted join hands the Monday row four days
    of future while its target is Tuesday. Measured cost while it was there:
    pooled out-of-sample AUC 0.7075, and 0.5245 once the three columns went."""
    weekly = _weekly_frame()
    _stub_read_sql(monkeypatch, weekly)
    out = _joined(monkeypatch, weekly)
    w_ret = weekly["close"].pct_change()

    this_monday = pd.Timestamp("2026-01-05")
    prev_monday = this_monday - pd.Timedelta(days=7)
    for day in (this_monday, this_monday + pd.Timedelta(days=3)):
        got = float(out.loc[day, "w_ret"])
        assert abs(got - float(w_ret.loc[prev_monday])) < 1e-9, "not the completed week"
        assert abs(got - float(w_ret.loc[this_monday])) > 1e-12, "reads its own week"


def test_the_week_long_shift_is_what_prevents_it(monkeypatch):
    """Positive control. With the shift neutralised the same row reads its own
    week again, so the test above is checking the fix and not the fixture."""
    import core.features as F
    weekly = _weekly_frame()
    _stub_read_sql(monkeypatch, weekly)
    real = pd.Timedelta
    monkeypatch.setattr(F.pd, "Timedelta", lambda **kw: real(days=0))
    out = _joined(monkeypatch, weekly)
    w_ret = weekly["close"].pct_change()
    this_monday = pd.Timestamp("2026-01-05")
    assert abs(float(out.loc[this_monday, "w_ret"])
               - float(w_ret.loc[this_monday])) < 1e-9
