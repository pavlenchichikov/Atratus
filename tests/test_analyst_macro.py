import datetime
import json

from core.analyst import macro

RAW = {"fred": {"us_10y_yield": {"last": ["2026-09-26", 4.1]}},
       "prices": {"BTC": {"ret_20": 3.2}}, "key_rates": {"ru": {"rate": 17.0}}}


def _reply(evidence):
    return {"regime": "tightening, risk-off",
            "drivers": [{"what": "US long rates rising", "evidence": evidence}],
            "by_market": {"us": "rates weigh on growth", "ru": "key rate high"}}


def test_a_driver_backed_only_by_opinion_is_rejected():
    bad = _reply([{"source": "https://www.zacks.com/a", "kind": "data", "value": "1"}])
    assert macro.validate(bad, RAW) is None
    good = _reply(["fred.us_10y_yield",
                   {"source": "https://www.zacks.com/a", "kind": "data", "value": "1"}])
    out = macro.validate(good, RAW)
    assert out["drivers"][0]["evidence"] == ["fred.us_10y_yield"]


def test_a_driver_citing_an_unknown_raw_key_is_dropped():
    assert macro.validate(_reply(["fred.made_up"]), RAW) is None


def test_stored_view_is_read_back_for_the_market(tmp_path):
    db = str(tmp_path / "m.db")
    macro.save("2026-09-29", "ollama:x", macro.validate(_reply(["prices.BTC"]), RAW),
               db_path=db)
    view = macro.for_date("2026-09-30", "ru", db_path=db)
    assert view["regime"] == "tightening, risk-off"
    assert view["this_market"] == "key rate high"
    assert view["drivers"] == ["US long rates rising"]
    assert macro.for_date("2026-10-09", "ru", db_path=db) is None   # too old


def test_run_asks_with_the_raw_block_and_saves(tmp_path, monkeypatch):
    db = str(tmp_path / "m.db")
    monkeypatch.setattr(macro, "raw_block", lambda today=None: RAW)
    prompts = []

    def call(p):
        prompts.append(p)
        return json.dumps(_reply(["key_rates.ru"]))

    assert macro.run(call, today="2026-09-29", brain="b", db_path=db)
    assert '"us_10y_yield"' in prompts[0]
    assert macro.for_date("2026-09-29", "us", db_path=db)["this_market"]


def test_dossier_carries_the_view_only_when_one_exists(monkeypatch):
    from core.analyst import dossier

    monkeypatch.setattr(macro, "for_date", lambda day, market, db_path=None: None)
    assert dossier._macro_view("AAPL", None) is None
    monkeypatch.setattr(macro, "for_date",
                        lambda day, market, db_path=None: {"regime": "r", "market": market})
    assert dossier._macro_view("AAPL", None)["market"] == "us"


def test_auto_builds_the_macro_view_before_the_scout_picks(tmp_path, monkeypatch):
    import analyst
    from core.analyst import scout, store

    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "m.db"))
    monkeypatch.setattr(store, "backfill_outcomes", lambda *a, **k: 0)
    monkeypatch.setattr("core.analyst.lessons.learn", lambda *a, **k: 0)
    monkeypatch.setattr(scout, "summary", lambda **k: {})
    order = []
    monkeypatch.setattr(analyst, "_daily_macro", lambda: order.append("macro"))
    monkeypatch.setattr(analyst, "_weekly_recheck", lambda: order.append("recheck"))
    monkeypatch.setattr(scout, "pick", lambda *a, **k: order.append("scout") or [])
    monkeypatch.setattr(analyst, "_load_table", lambda: {"asset": {}, "class": {}})
    monkeypatch.delenv("GTRADE_ANALYST", raising=False)
    monkeypatch.setattr("core.analyst.brains.call_for", lambda role: (lambda p: ""))
    args = type("A", (), {"max_assets": 3, "mode": "solo", "depth": None,
                          "horizons": "1", "report_dir": str(tmp_path)})()
    analyst.cmd_auto(args)
    assert order == ["macro", "recheck", "scout"]


def test_weekly_recheck_runs_only_when_a_source_is_a_week_old(monkeypatch):
    import analyst
    from core.analyst import hunt

    ran = []
    monkeypatch.setattr(hunt, "recheck", lambda: ran.append(1) or (1, 0))
    monkeypatch.setattr(hunt, "listing", lambda: [
        {"status": "verified", "verified_at": datetime.date.today().isoformat()}])
    analyst._weekly_recheck()
    assert ran == []
    monkeypatch.setattr(hunt, "listing", lambda: [
        {"status": "verified", "verified_at": "2026-01-01"}])
    analyst._weekly_recheck()
    assert ran == [1]
