"""Fetch and storage tests. No network: every payload is a literal."""

from intraday_fetch import MOEX_TZ, fetch_asset, load, parse_moex, parse_yahoo, store


def _payload(ts, closes, tz="America/New_York"):
    return {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": tz},
        "timestamp": ts,
        "indicators": {"quote": [{
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [100.0] * len(closes)}]}}]}}


def test_parse_yahoo_normalises_epoch_seconds_to_utc():
    # 1788874200 is 13:30 UTC, which is 09:30 in New York. The value below was
    # read from the interpreter, not reasoned about: the plan's own draft said
    # 09:30 and was wrong, which is the whole reason the check exists.
    bars, tz = parse_yahoo(_payload([1788874200], [10.0]))
    assert tz == "America/New_York"
    assert bars[0]["ts"] == "2026-09-08T13:30:00+00:00"
    assert bars[0]["close"] == 10.0


def test_parse_yahoo_drops_a_bar_with_a_null_close():
    """Yahoo emits the in-progress hour with nulls; it is not a bar yet."""
    bars, _tz = parse_yahoo(_payload([1788874200, 1788877800], [10.0, None]))
    assert len(bars) == 1


def test_parse_yahoo_drops_the_hour_still_in_progress():
    """The daily fetchers stored unfinished sessions and never replaced them
    (38 of 60 recent SBER closes off by more than 0.1%, 2026-09-11). The hourly
    store must not repeat it: an hour is a bar only once it has ended."""
    import time
    now = int(time.time())
    bars, _tz = parse_yahoo(_payload([now - 7200, now - 1800], [10.0, 11.0]), now=now)
    assert len(bars) == 1 and bars[0]["close"] == 10.0


def test_parse_moex_drops_the_hour_still_in_progress():
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    msk = datetime.now(ZoneInfo(MOEX_TZ)).replace(tzinfo=None, microsecond=0)
    fmt = "%Y-%m-%d %H:%M:%S"
    done = msk - timedelta(hours=2)
    live = msk - timedelta(minutes=20)
    rows = [[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, done.strftime(fmt),
             (done + timedelta(minutes=59, seconds=59)).strftime(fmt)],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, live.strftime(fmt),
             (live + timedelta(minutes=59, seconds=59)).strftime(fmt)]]
    assert len(parse_moex(_moex(rows))) == 1


def test_a_refetched_bar_replaces_the_stored_one(tmp_path):
    """So a refetch can heal a bar stored before this rule existed."""
    db = str(tmp_path / "intraday.db")
    bar = {"ts": "2026-09-04T14:30:00+00:00", "open": 1.0, "high": 2.0,
           "low": 0.5, "close": 1.5, "volume": 10.0}
    store("SBER", [bar], db_path=db)
    store("SBER", [dict(bar, close=1.9)], db_path=db)
    got = load("SBER", db_path=db)
    assert len(got) == 1 and got[0]["close"] == 1.9


def test_parse_yahoo_returns_nothing_for_an_empty_result():
    assert parse_yahoo({"chart": {"result": []}}) == ([], None)


def _moex(rows):
    return {"candles": {
        "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
        "data": rows}}


def test_parse_moex_reads_naive_moscow_time_as_moscow():
    """The three-hour trap. ISS returns a naive local string; reading it as UTC
    would shift every Russian bar by three hours and nothing would complain."""
    rows = [[10.0, 11.0, 12.0, 9.0, 0.0, 500.0,
             "2026-09-04 10:00:00", "2026-09-04 10:59:59"]]
    bars = parse_moex(_moex(rows))
    assert bars[0]["ts"] == "2026-09-04T07:00:00+00:00"     # 10:00 MSK is 07:00 UTC
    assert bars[0]["close"] == 11.0
    assert bars[0]["volume"] == 500.0


def test_parse_moex_handles_an_empty_page():
    assert parse_moex(_moex([])) == []


def test_the_moscow_zone_is_named_once():
    assert MOEX_TZ == "Europe/Moscow"


def test_store_and_load_round_trip(tmp_path):
    db = str(tmp_path / "intraday.db")
    bars = [{"ts": "2026-09-04T14:30:00+00:00", "open": 1.0, "high": 2.0,
             "low": 0.5, "close": 1.5, "volume": 10.0}]
    assert store("SBER", bars, db_path=db) == 1
    got = load("SBER", db_path=db)
    assert len(got) == 1 and got[0]["close"] == 1.5


def test_storing_the_same_bar_twice_does_not_duplicate_it(tmp_path):
    """Re-running the fetch must be safe: the primary key is (asset, ts)."""
    db = str(tmp_path / "intraday.db")
    bars = [{"ts": "2026-09-04T14:30:00+00:00", "open": 1.0, "high": 2.0,
             "low": 0.5, "close": 1.5, "volume": 10.0}]
    store("SBER", bars, db_path=db)
    store("SBER", bars, db_path=db)
    assert len(load("SBER", db_path=db)) == 1


def test_assets_do_not_collide(tmp_path):
    db = str(tmp_path / "intraday.db")
    bar = {"ts": "2026-09-04T14:30:00+00:00", "open": 1.0, "high": 1.0,
           "low": 1.0, "close": 1.0, "volume": 1.0}
    store("SBER", [bar], db_path=db)
    store("GAZP", [bar], db_path=db)
    assert len(load("SBER", db_path=db)) == 1
    assert len(load("GAZP", db_path=db)) == 1


def test_loading_an_absent_database_returns_nothing(tmp_path):
    """A read must never create the store. market.db grew a stub exactly this
    way, and this file must not repeat it."""
    import os
    missing = str(tmp_path / "absent.db")
    assert load("SBER", db_path=missing) == []
    assert not os.path.exists(missing)


class _FakeHttp:
    """Stands in for net.http_get so no test touches the network."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append(url)
        payload = self.payload

        class R:
            def json(_self):
                return payload
        return R()


def test_fetch_asset_stores_what_it_parsed(tmp_path, monkeypatch):
    import intraday_fetch as f
    monkeypatch.setattr(f, "DB_PATH", str(tmp_path / "intraday.db"))
    payload = {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [1788874200],
        "indicators": {"quote": [{"open": [1.0], "high": [1.0], "low": [1.0],
                                  "close": [1.0], "volume": [5.0]}]}}]}}
    http = _FakeHttp(payload)
    written, tz = fetch_asset("AAPL", http=http)
    assert written == 1 and tz == "America/New_York"
    assert "AAPL" in http.calls[0]


def test_fetch_asset_uses_the_vendor_symbol_not_the_internal_name(tmp_path, monkeypatch):
    """SP500 is ^GSPC to Yahoo. config.FULL_ASSET_MAP already knows; the
    universe file must not carry a second copy of that mapping."""
    import intraday_fetch as f
    monkeypatch.setattr(f, "DB_PATH", str(tmp_path / "intraday.db"))
    http = _FakeHttp({"chart": {"result": []}})
    fetch_asset("SP500", http=http)
    assert "%5EGSPC" in http.calls[0] or "^GSPC" in http.calls[0]


def test_an_index_uses_the_moex_index_endpoint(tmp_path, monkeypatch):
    """IMOEX is an index, not a share: ISS serves it from a different board,
    and data_engine already special-cases it for the daily fetch."""
    import intraday_fetch as f
    monkeypatch.setattr(f, "DB_PATH", str(tmp_path / "intraday.db"))
    http = _FakeHttp({"candles": {"columns": [], "data": []}})
    fetch_asset("IMOEX", http=http)
    assert "markets/index" in http.calls[0]


def test_source_follows_the_moex_list_not_the_symbol_shape():
    """config.MOEX_ASSETS is the single source of which exchange serves an
    asset. A second copy is how POL once stayed stale for a thousand days."""
    import config
    import intraday_fetch as f
    assert f.source_for("SBER") == "moex"
    assert f.source_for("IMOEX") == "moex"
    assert f.source_for("AAPL") == "yahoo"
    assert f.source_for("BTC") == "yahoo"
    assert all(f.source_for(a) == "moex" for a in config.MOEX_ASSETS)


def test_a_moex_share_uses_the_share_board(tmp_path, monkeypatch):
    import intraday_fetch as f
    monkeypatch.setattr(f, "DB_PATH", str(tmp_path / "intraday.db"))
    http = _FakeHttp({"candles": {"columns": [], "data": []}})
    f.fetch_asset("GAZP", http=http)
    assert "markets/shares/boards/TQBR" in http.calls[0]


def test_fetch_all_skips_assets_already_stored(tmp_path, monkeypatch):
    import intraday_fetch as f
    db = str(tmp_path / "intraday.db")
    monkeypatch.setattr(f, "DB_PATH", db)
    f.store("AAPL", [{"ts": "2026-09-04T14:30:00+00:00", "open": 1.0, "high": 1.0,
                      "low": 1.0, "close": 1.0, "volume": 1.0}], db_path=db)
    http = _FakeHttp({"chart": {"result": []}})
    out = f.fetch_all(["AAPL", "MSFT"], http=http, log=lambda *a: None)
    assert out["AAPL"] == "skip"
    assert len(http.calls) == 1 and "MSFT" in http.calls[0]


def test_fetch_all_records_an_error_and_carries_on(tmp_path, monkeypatch):
    """One dead ticker must not stop 846 others."""
    import intraday_fetch as f
    monkeypatch.setattr(f, "DB_PATH", str(tmp_path / "intraday.db"))

    def boom(url, **kw):
        raise ConnectionError("reset")
    out = f.fetch_all(["AAPL", "MSFT"], http=boom, log=lambda *a: None)
    assert out["AAPL"].startswith("error") and out["MSFT"].startswith("error")


def test_the_timezone_is_stored_with_the_bars(tmp_path, monkeypatch):
    """Bars without their zone cannot be grouped into sessions, and guessing
    UTC silently breaks every asset that trades across midnight. Found on
    2026-09-08 when the round-trip scored forex at 0.65 and gold at 0.11."""
    import intraday_fetch as f
    db = str(tmp_path / "intraday.db")
    monkeypatch.setattr(f, "DB_PATH", db)
    f.store("AAPL", [{"ts": "2026-09-04T14:30:00+00:00", "open": 1.0, "high": 1.0,
                      "low": 1.0, "close": 1.0, "volume": 1.0}], db_path=db)
    f.store_tz("AAPL", "America/New_York", db_path=db)
    assert f.load_tz("AAPL", db_path=db) == "America/New_York"


def test_an_unknown_asset_has_no_timezone_rather_than_a_wrong_one(tmp_path):
    """None, never a default: a wrong zone is worse than a missing one because
    it produces plausible sessions that are silently off."""
    import intraday_fetch as f
    assert f.load_tz("NOPE", db_path=str(tmp_path / "absent.db")) is None


# --- keeping the store fresh -------------------------------------------------

def _stored(f, db, asset="AAPL", ts="2026-09-04T14:30:00+00:00"):
    f.store(asset, [{"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                     "close": 1.0, "volume": 1.0}], db_path=db)


def test_last_ts_reads_one_value_and_never_creates_the_store(tmp_path):
    """fetch_all used to answer "has this asset any bars" by loading every bar
    it has. Same rule as load(): a reader must not conjure the database."""
    import os

    import intraday_fetch as f
    missing = str(tmp_path / "absent.db")
    assert f.last_ts("AAPL", db_path=missing) is None
    assert not os.path.exists(missing)

    db = str(tmp_path / "intraday.db")
    _stored(f, db, ts="2026-09-04T14:30:00+00:00")
    _stored(f, db, ts="2026-09-05T14:30:00+00:00")
    assert f.last_ts("AAPL", db_path=db) == "2026-09-05T14:30:00+00:00"
    assert f.last_ts("NOPE", db_path=db) is None


def test_a_top_up_asks_for_a_short_window_and_a_first_fetch_for_the_long_one(tmp_path, monkeypatch):
    """The positive control for the whole change. A top-up that still requested
    730d would work and would keep costing 844 full downloads to collect a few
    days, which is exactly why the store was never refreshed."""
    import intraday_fetch as f
    db = str(tmp_path / "intraday.db")
    monkeypatch.setattr(f, "DB_PATH", db)
    _stored(f, db)

    http = _FakeHttp({"chart": {"result": []}})
    f.fetch_all(["AAPL"], http=http, db_path=db, mode="top_up", log=lambda *a: None)
    assert "range=1mo" in http.calls[0], http.calls[0]

    http2 = _FakeHttp({"chart": {"result": []}})
    f.fetch_all(["MSFT"], http=http2, db_path=db, mode="top_up", log=lambda *a: None)
    assert "range=730d" in http2.calls[0], (
        "an asset with no history must still get the full window")


def test_a_moex_top_up_starts_from_the_last_stored_bar_not_2015(tmp_path, monkeypatch):
    """The expensive half: ISS pages 500 candles at a time from MOEX_START, so
    a top-up that ignored the stored history re-walked eleven years per asset."""
    import intraday_fetch as f
    db = str(tmp_path / "intraday.db")
    monkeypatch.setattr(f, "DB_PATH", db)
    _stored(f, db, asset="GAZP", ts="2026-09-05T07:00:00+00:00")

    http = _FakeHttp({"candles": {"columns": [], "data": []}})
    f.fetch_all(["GAZP"], http=http, db_path=db, mode="top_up", log=lambda *a: None)
    assert "from=2026-09-05" in http.calls[0], http.calls[0]
    assert f.MOEX_START not in http.calls[0]


def test_the_default_mode_still_skips_a_stored_asset(tmp_path, monkeypatch):
    """Backwards compatibility is the point: `new` is what every existing
    caller means, and a mode that quietly started refetching would turn a
    cheap call into 844 downloads."""
    import intraday_fetch as f
    db = str(tmp_path / "intraday.db")
    monkeypatch.setattr(f, "DB_PATH", db)
    _stored(f, db)
    http = _FakeHttp({"chart": {"result": []}})
    out = f.fetch_all(["AAPL", "MSFT"], http=http, db_path=db, log=lambda *a: None)
    assert out["AAPL"] == "skip"
    assert len(http.calls) == 1 and "MSFT" in http.calls[0]


def test_refetch_keyword_and_mode_agree(tmp_path, monkeypatch):
    """`refetch=True` predates `mode`; it must keep meaning exactly what it did."""
    import intraday_fetch as f
    db = str(tmp_path / "intraday.db")
    monkeypatch.setattr(f, "DB_PATH", db)
    _stored(f, db)
    http = _FakeHttp({"chart": {"result": []}})
    f.fetch_all(["AAPL"], http=http, refetch=True, db_path=db, log=lambda *a: None)
    assert "range=730d" in http.calls[0], "a refetch takes everything, not a window"


def test_an_unknown_mode_is_refused_rather_than_guessed(tmp_path, monkeypatch):
    import pytest

    import intraday_fetch as f
    monkeypatch.setattr(f, "DB_PATH", str(tmp_path / "intraday.db"))
    with pytest.raises(ValueError):
        f.fetch_all(["AAPL"], mode="fresh", log=lambda *a: None)
