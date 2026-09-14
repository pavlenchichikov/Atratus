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


def _cot_frame():
    """Weekly reports, published three days after the date they describe, which
    is what the real table carries: a lag of exactly 3 days on all 40982 rows."""
    report = pd.date_range("2026-01-06", periods=12, freq="7D")
    return pd.DataFrame(
        {"open_interest": [1000.0] * 12,
         "noncomm_long": [500.0 + 40 * i for i in range(12)],
         "noncomm_short": [300.0] * 12},
        index=pd.Index(report + pd.Timedelta(days=3), name="available_date"))


def _daily(periods=40, start="2026-01-05"):
    """A frame shaped the way the real chain hands it over: engineer_features
    ends with reset_index(), so Date is a COLUMN, not the index. The first
    version of these tests passed a DatetimeIndex instead, which sent every
    function down its "no date column" branch and returned zeros."""
    idx = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"Date": idx, "close": 1.0})


def test_cot_is_joined_on_the_date_it_was_published_not_the_date_it_describes(monkeypatch):
    """The COT report describes a Tuesday and reaches the public the Friday
    after. Joining on report_date would hand every row three days it could not
    have known, which is the exact shape of the weekly leak."""
    import core.features as F
    cot = _cot_frame()
    monkeypatch.setattr(F.pd, "read_sql", lambda sql, engine, **kw: cot.copy())
    daily = _daily()
    out = F.add_cot_features(daily, "GOLD", engine=None).set_index("Date")

    published = cot.index[0]                      # 2026-01-09
    first_value = (500.0 - 300.0) / 1000.0
    # the day before publication cannot carry it
    assert float(out.loc[published - pd.Timedelta(days=1), "cot_net_pct"]) != first_value
    # the day of publication, and the days after, can
    assert abs(float(out.loc[published, "cot_net_pct"]) - first_value) < 1e-9
    assert abs(float(out.loc[published + pd.Timedelta(days=2), "cot_net_pct"])
               - first_value) < 1e-9


def test_a_dead_cot_series_decays_instead_of_persisting(monkeypatch):
    """NZDUSD stops in 2022-02 and USDRUB in 2022-03. Carrying their last
    reading for ever would state a positioning nobody reported."""
    import core.features as F
    cot = _cot_frame().iloc[:1]                   # one report, then silence
    monkeypatch.setattr(F.pd, "read_sql", lambda sql, engine, **kw: cot.copy())
    daily = _daily(periods=60)
    out = F.add_cot_features(daily, "NZDUSD", engine=None).set_index("Date")
    tail = float(out["cot_net_pct"].iloc[-1])
    assert tail == 0.0, "a stopped series was carried to the end of history"


def test_an_asset_without_cot_keeps_the_column_shape(monkeypatch):
    """820 of 847 assets have no COT at all. The columns must still exist, or
    feature_version and the model input width change per asset."""
    import core.features as F
    monkeypatch.setattr(F.pd, "read_sql",
                        lambda sql, engine, **kw: pd.DataFrame())
    daily = _daily(periods=10)
    out = F.add_cot_features(daily, "AAPL", engine=None)
    for c in F._COT_FEATURES:
        assert c in out.columns and float(out[c].iloc[0]) == 0.0


def test_a_frame_without_dates_still_keeps_the_column_shape(monkeypatch):
    """Both functions carry a "no date column" branch, and nothing covered it.
    The first version of these very tests fell into it silently and asserted on
    the zeros it returns, which is how a test passes while measuring nothing."""
    import core.features as F
    monkeypatch.setattr(F.pd, "read_sql",
                        lambda sql, engine, **kw: pd.DataFrame())
    undated = pd.DataFrame({"close": [1.0, 1.0, 1.0]})
    b = F.add_breadth_features(undated.copy(), engine=None)
    c = F.add_cot_features(undated.copy(), "GOLD", engine=None)
    for col in F._BREADTH_FEATURES:
        assert col in b.columns and float(b[col].iloc[0]) == 0.0
    for col in F._COT_FEATURES:
        assert col in c.columns and float(c[col].iloc[0]) == 0.0


def test_breadth_never_reads_a_row_dated_after_the_bar(monkeypatch):
    """The as-of join. A row dated D may use the breadth of D and no later."""
    import core.features as F
    idx = pd.date_range("2026-01-05", periods=20, freq="D")
    b = pd.DataFrame({"above_sma50_pct": [i / 100.0 for i in range(20)],
                      "positive_20d_pct": [0.5] * 20}, index=idx)
    b.index.name = "Date"
    monkeypatch.setattr(F.pd, "read_sql", lambda sql, engine, **kw: b.copy())
    daily = pd.DataFrame({"Date": idx, "close": 1.0})
    out = F.add_breadth_features(daily, engine=None).set_index("Date")
    for day in idx:
        assert abs(float(out.loc[day, "breadth_above_sma50"])
                   - float(b.loc[day, "above_sma50_pct"])) < 1e-9


def test_history_older_than_the_breadth_series_is_not_called_a_crash(monkeypatch):
    """market_breadth starts 2001-10-02 and 94 assets trade before it. Filling
    those rows with 0.0 would state that none of the book was above its SMA50,
    which is a crash regime rather than a missing reading."""
    import core.features as F
    b_idx = pd.date_range("2026-01-15", periods=10, freq="D")
    b = pd.DataFrame({"above_sma50_pct": [0.62] * 10,
                      "positive_20d_pct": [0.55] * 10}, index=b_idx)
    b.index.name = "Date"
    monkeypatch.setattr(F.pd, "read_sql", lambda sql, engine, **kw: b.copy())
    daily = _daily(periods=20)
    out = F.add_breadth_features(daily, engine=None).set_index("Date")
    head = float(out["breadth_above_sma50"].iloc[0])
    assert head == 0.62, "the head was zero-filled into a fake crash"


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
