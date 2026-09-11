"""A bar is stored only once its session has closed.

Measured 2026-09-11: the daily fetchers stored the bar of a session still in
progress, and the next run started from the following day, so the snapshot was
never replaced. Against a fresh fetch, stored closes differed by more than 0.1%
on 38 of 60 recent days for SBER, 43 of 59 for BTC, 40 of 52 for GOLD (worst
8.4% GAZP, 15.3% ETH), in every month of daily runs since March and in none of
the bulk-fetched months before it.
"""
import datetime as dt

import pandas as pd
from sqlalchemy import create_engine

import data_engine as de
import net


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p

    status_code = 200


def _daily_payload(stamps, closes, period=None):
    n = len(stamps)
    res = {"timestamp": [int(s.timestamp()) for s in stamps],
           "indicators": {"quote": [{"open": closes, "close": closes, "high": closes,
                                     "low": closes, "volume": [1.0] * n}]}}
    if period is not None:
        res["meta"] = {"currentTradingPeriod": {"regular": {
            "start": int(period[0].timestamp()), "end": int(period[1].timestamp())}}}
    return {"chart": {"result": [res]}}


def _recent_stamps():
    now = dt.datetime.now().replace(microsecond=0)
    return now, [now - dt.timedelta(days=2), now - dt.timedelta(days=1),
                 now - dt.timedelta(hours=1)]


def test_a_session_still_in_progress_is_not_stored(monkeypatch):
    now, stamps = _recent_stamps()
    period = (now - dt.timedelta(hours=2), now + dt.timedelta(hours=3))
    payload = _daily_payload(stamps, [1.0, 2.0, 3.0], period)
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(payload))
    got = de.fetch_yahoo_smart("AAPL", None)
    assert len(got) == 2 and got["Close"].iloc[-1] == 2.0


def test_a_session_that_has_closed_is_stored(monkeypatch):
    """Positive control: the rule reads the session, not "drop the last row"."""
    now, stamps = _recent_stamps()
    period = (now - dt.timedelta(hours=8), now - dt.timedelta(minutes=30))
    payload = _daily_payload(stamps, [1.0, 2.0, 3.0], period)
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(payload))
    assert len(de.fetch_yahoo_smart("AAPL", None)) == 3


def test_without_session_metadata_nothing_is_dropped(monkeypatch):
    _now, stamps = _recent_stamps()
    payload = _daily_payload(stamps, [1.0, 2.0, 3.0])
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(payload))
    assert len(de.fetch_yahoo_smart("AAPL", None)) == 3


def _moex_payload(dates):
    cols = ["open", "close", "high", "low", "value", "volume", "begin", "end"]
    data = [[1.0, 1.0, 1.0, 1.0, 0.0, 5.0, d + " 00:00:00", d + " 23:59:59"] for d in dates]
    return {"candles": {"columns": cols, "data": data}}


def test_the_moex_candle_for_today_is_not_stored(monkeypatch):
    today = de._moex_today()
    days = [(today - dt.timedelta(days=d)).isoformat() for d in (2, 1, 0)]
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(_moex_payload(days)))
    got = de.fetch_moex_smart("SBER", None)
    assert [str(d.date()) for d in got.index] == days[:2]


def _weekly_payload(mondays, closes):
    n = len(mondays)
    return {"chart": {"result": [{
        "timestamp": [int(m.timestamp()) for m in mondays],
        "indicators": {"quote": [{"open": closes, "close": closes, "high": closes,
                                  "low": closes, "volume": [1.0] * n}]}}]}}


def test_empty_vendor_weeks_are_built_from_our_own_daily_bars(tmp_path, monkeypatch):
    """TON: Yahoo returns the weeks after 2026-06-15 with every field null while
    the daily series is intact, so the weekly table stood still for 3 months."""
    eng = create_engine("sqlite:///%s" % (tmp_path / "m.db"))
    monday = pd.Timestamp(dt.date.today() - dt.timedelta(days=dt.date.today().weekday() + 21))
    week = [monday + pd.Timedelta(days=i) for i in range(7)]
    daily = pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d in week],
                          "open": [10, 11, 12, 13, 14, 15, 16], "high": [11, 19, 13, 14, 15, 16, 17],
                          "low": [9, 10, 5, 12, 13, 14, 15], "close": [10.5, 12, 12.5, 13.5, 14.5, 15.5, 16.5],
                          "volume": [1, 1, 1, 1, 1, 1, 1]})
    daily.to_sql("ton", eng, index=False)
    monkeypatch.setattr(de, "engine", eng)
    stamps = [monday - pd.Timedelta(days=7) + pd.Timedelta(hours=3), monday + pd.Timedelta(hours=3)]
    monkeypatch.setattr(net, "http_get",
                        lambda url, **kw: _Resp(_weekly_payload(stamps, [9.0, None])))
    got = de.fetch_yahoo_weekly("TON", None)
    assert len(got) == 2
    row = got.iloc[-1]
    assert (row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]) == (10, 19, 5, 16.5, 7)


def test_a_stored_week_is_not_fetched_again_for_its_time_of_day(monkeypatch):
    """Yahoo stamps a UTC week at 03:00 Moscow time; the stored row is the bare
    date. Compared raw, 03:00 is later than 00:00, so the week already stored
    came back as "+1 weekly bars" on every run."""
    monday = pd.Timestamp(dt.date.today() - dt.timedelta(days=dt.date.today().weekday() + 14))
    stamps = [monday + pd.Timedelta(hours=3)]
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(_weekly_payload(stamps, [1.0])))
    assert de.fetch_yahoo_weekly("TON", monday) is None
