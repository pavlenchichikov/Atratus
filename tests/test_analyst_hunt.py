import datetime
import json

from core.analyst import agent, hunt, tools

TODAY = datetime.date(2026, 9, 29)


def _csv(last="2026-09-28"):
    return "date,close\n2026-09-26,10.5\n2026-09-27,10.7\n%s,10.9\n" % last


def test_parse_rows_reads_csv_json_and_moex_iss_but_not_html():
    assert len(hunt.parse_rows(_csv())) == 3
    assert hunt.parse_rows(json.dumps([{"date": "2026-09-28", "v": 1}]))[0]["v"] == 1
    iss = {"history": {"columns": ["TRADEDATE", "CLOSE"],
                       "data": [["2026-09-28", 300.1]]}}
    assert hunt.parse_rows(json.dumps(iss)) == [{"TRADEDATE": "2026-09-28", "CLOSE": 300.1}]
    assert hunt.parse_rows("<html><body>Not found, sorry</body></html>") == []


def _cand(url="https://iss.moex.com/x/{secid}.csv", freq="daily"):
    return {"name": "moex_hist", "url_template": url, "format": "csv",
            "fields": "date,close", "frequency": freq, "symbol_kind": "secid"}


def test_verify_accepts_fresh_rows_and_rejects_the_rest():
    ok, why, sample = hunt.verify(_cand(), lambda url: _csv(), asset="SBER", today=TODAY)
    assert ok, why
    assert sample
    cases = [
        (lambda url: "date,close\n2026-07-29,1\n2026-07-30,1\n2026-08-01,2\n", _cand(), "stale"),
        (lambda url: "a,b\nx,1\nz,2\nq,3\n", _cand(), "date"),
        (lambda url: "<html>error page</html>", _cand(), "rows"),
        (lambda url: _csv(), _cand("https://www.tipranks.com/{ticker}"), "opinion"),
        (lambda url: 1 / 0, _cand(), "fetch"),
    ]
    for fetch, cand, word in cases:
        ok, why, _ = hunt.verify(cand, fetch, asset="SBER", today=TODAY)
        assert not ok and word in why, (word, why)
    ok, _, _ = hunt.verify(_cand(freq="monthly"),
                           lambda url: "date,close\n2026-07-01,1\n2026-08-01,1\n2026-09-01,2\n",
                           asset="SBER", today=TODAY)
    assert ok


def test_symbols_follow_the_template_kind():
    assert hunt.symbol("AAPL", "ticker") == "AAPL"
    assert hunt.symbol("BTC", "pair") == "BTCUSDT"
    assert hunt.symbol("SBER", "secid") == "SBER"
    assert hunt.market_of("EURUSD") == "fx"
    assert hunt.market_of("AAPL") == "us"


def test_registry_round_trip(tmp_path):
    db = str(tmp_path / "m.db")
    hunt.save({**_cand(), "market": "ru"}, "claude-code:opus", "sample", db_path=db)
    rows = hunt.listing(db_path=db)
    assert rows[0]["name"] == "moex_hist" and rows[0]["status"] == "verified"
    hunt.set_status("moex_hist", "off", db_path=db)
    assert hunt.listing(market="ru", status="verified", db_path=db) == []


def test_raw_source_lists_then_reads_with_the_asset_symbol(tmp_path, monkeypatch):
    db = str(tmp_path / "m.db")
    monkeypatch.setattr(hunt, "DB_PATH", db)
    hunt.save({**_cand(), "market": "ru"}, "ollama", "s", db_path=db)
    seen = []
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: seen.append(url) or _csv())
    listed = tools.call({"tool": "raw_source", "args": {}}, asset="SBER")["result"]
    assert [s["name"] for s in listed] == ["moex_hist"]
    rows = tools.call({"tool": "raw_source", "args": {"name": "moex_hist"}},
                      asset="SBER")["result"]
    assert seen == ["https://iss.moex.com/x/SBER.csv"] and rows[-1]["close"] == "10.9"


def test_ask_json_runs_a_tool_round_then_validates(monkeypatch):
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: _csv())
    replies = [json.dumps({"tools": [{"tool": "web_fetch",
                                      "args": {"url": "https://x.org/a.csv"}}]}),
               "not json", json.dumps({"sources": [1]})]
    prompts = []

    def call(p):
        prompts.append(p)
        return replies.pop(0)

    log = []
    out = agent.ask_json("find", call, lambda o: o.get("sources") if isinstance(o, dict)
                         else None, tool_calls=log)
    assert out == [1] and log[0]["tool"] == "web_fetch"
    assert "2026-09-28,10.9" in prompts[1]


def test_hunt_saves_only_the_verified(tmp_path):
    db = str(tmp_path / "m.db")
    reply = json.dumps({"sources": [_cand(), {**_cand(), "name": "bad",
                                              "url_template": "https://zacks.com/{ticker}"}]})
    report = hunt.run("ru", lambda p: reply, fetch=lambda url: _csv(), brain="test",
                      db_path=db, today=TODAY, log=lambda s: None)
    assert [r["name"] for r in hunt.listing(db_path=db)] == ["moex_hist"]
    assert any("bad" in line for line in report)


def test_world_bank_shape_reads_the_rows_not_the_metadata():
    wb = [{"page": 1, "lastupdated": "2026-07-13"},
          [{"date": "2025", "value": 2.1}, {"date": "2024", "value": 2.8}]]
    assert hunt.parse_rows(json.dumps(wb)) == wb[1]


def test_a_single_row_reply_is_too_few_to_trust():
    ok, why, _ = hunt.verify(_cand(), lambda url: "date,close\n2026-09-28,1\n",
                             asset="SBER", today=TODAY)
    assert not ok and "few" in why


def test_raw_source_returns_the_newest_rows_even_when_the_feed_is_newest_first(
        tmp_path, monkeypatch):
    db = str(tmp_path / "m.db")
    monkeypatch.setattr(hunt, "DB_PATH", db)
    hunt.save({**_cand(), "market": "ru"}, "x", "s", db_path=db)
    desc = "date,close\n" + "\n".join("2026-09-%02d,%d" % (d, d) for d in range(28, 0, -1))
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: desc)
    rows = tools.call({"tool": "raw_source", "args": {"name": "moex_hist"}},
                      asset="SBER")["result"]
    assert rows[-1]["date"] == "2026-09-28" and rows[0]["date"] == "2026-09-01"
