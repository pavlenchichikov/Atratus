import sqlite3


def test_invalid_menu_values_are_refused_and_env_unchanged(tmp_path, monkeypatch):
    import analyst
    from core.analyst import brains
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(brains, "ENV_PATH", str(env))
    for bad in ("GTRADE_ANALYST_AUTO=yes", "GTRADE_ANALYST_MODE=duo",
                "GTRADE_ANALYST_AUTO_MAX=many"):
        assert analyst.main(["brains", "--set", bad]) == 1
    assert env.read_text(encoding="utf-8") == ""
    for k in ("GTRADE_ANALYST_AUTO", "GTRADE_ANALYST_MODE", "GTRADE_ANALYST_AUTO_MAX"):
        monkeypatch.delenv(k, raising=False)
    for good in ("GTRADE_ANALYST_AUTO=1", "GTRADE_ANALYST_MODE=team",
                 "GTRADE_ANALYST_AUTO_MAX=4"):
        assert analyst.main(["brains", "--set", good]) == 0


def test_auto_picks_runs_labels_and_reports(tmp_path, monkeypatch):
    import analyst
    from core.analyst import scout, store
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setattr(store, "backfill_outcomes", lambda *a, **k: 0)
    monkeypatch.setattr("core.analyst.lessons.learn", lambda *a, **k: 0)
    monkeypatch.setattr(scout, "summary", lambda **k: {"movers": [], "misses": [], "stale": []})
    monkeypatch.setattr(scout, "pick", lambda call, s, max_assets=5: [
        {"asset": "SP500", "reason": "moved", "by": "scout"}])
    seen = {}

    def fake_run(args):
        seen.update(assets=args.assets, mode=args.mode, label=args.mode_label,
                    skip=args.skip_judged)
        store.write_judgment({"date": "2026-09-25", "asset": "SP500", "horizon": 1,
                              "direction": "up", "conviction": 3, "thesis": "t",
                              "mode": args.mode_label}, db_path=path)
        return 0

    monkeypatch.setattr(analyst, "cmd_run", fake_run)
    monkeypatch.setattr(analyst, "_daily_macro", lambda: None)
    monkeypatch.setattr(analyst, "_weekly_recheck", lambda: None)
    monkeypatch.setattr(analyst, "_load_table", lambda: {"asset": {}, "class": {}})
    monkeypatch.delenv("GTRADE_ANALYST", raising=False)
    monkeypatch.setattr("core.analyst.brains.call_for", lambda role: (lambda p: ""))
    args = type("A", (), {"max_assets": 3, "mode": "team", "depth": None,
                          "horizons": "1", "report_dir": str(tmp_path)})()
    assert analyst.cmd_auto(args) == 0
    assert seen == {"assets": "SP500", "mode": "team", "label": "auto-team", "skip": True}
    import datetime
    name = "analyst_%s.md" % datetime.date.today().isoformat()   # named by run day
    text = (tmp_path / name).read_text(encoding="utf-8")
    assert "SP500" in text and "moved" in text and "up, conviction 3" in text
    con = sqlite3.connect(path)
    assert con.execute("SELECT asset, picked_by FROM analyst_auto_log").fetchall() == [
        ("SP500", "scout")]


def test_the_run_label_reaches_the_log(tmp_path, monkeypatch):
    import json

    import analyst
    from core.analyst import store
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    monkeypatch.setattr(analyst.calibrate, "forecast",
                        lambda *a, **k: {"pct": None, "lo": None, "hi": None})
    monkeypatch.setattr(analyst, "_print_judgment", lambda *a, **k: None)
    j = json.dumps({"direction": "up", "conviction": 2, "vol_regime": "normal",
                    "key_risk": "r", "thesis": "t", "evidence": ["rsi_14"]})
    d = {"asset": "SP500", "date": "2026-09-25", "close": 1.0, "atr": 0.1, "rsi_14": 50.0}
    analyst._judge_one(d, "SP500", "h", 1, lambda p: j, "full", {}, {}, 0, 0,
                       mode="solo", mode_label="auto-solo")
    assert sqlite3.connect(path).execute("SELECT mode FROM analyst_log").fetchone() == (
        "auto-solo",)


def test_the_loop_runs_auto_only_when_switched_on(monkeypatch):
    import loop_cycle
    monkeypatch.delenv("GTRADE_ANALYST_AUTO", raising=False)
    assert loop_cycle.analyst_auto_step() is None
    monkeypatch.setenv("GTRADE_ANALYST_AUTO", "1")
    ran = []
    monkeypatch.setattr(loop_cycle.subprocess, "run", lambda cmd, **k: ran.append(cmd))
    step = loop_cycle.analyst_auto_step()
    assert step["step"] == "analyst_auto" and step["status"] == "ok"
    assert ran and ran[0][1:3] == ["analyst.py", "auto"]


def test_auto_never_overwrites_a_judgment_already_made_for_that_bar(tmp_path, monkeypatch):
    from core.analyst import store
    path = str(tmp_path / "m.db")
    store.write_judgment({"date": "2026-09-25", "asset": "SP500", "horizon": 1,
                          "direction": "up", "mode": "solo"}, db_path=path)
    assert store.has_row("2026-09-25", "SP500", 1, db_path=path)
    assert not store.has_row("2026-09-25", "SP500", 20, db_path=path)


def test_auto_respects_the_kill_switch_before_spending_anything(monkeypatch):
    import analyst
    monkeypatch.setenv("GTRADE_ANALYST", "0")
    monkeypatch.setattr("core.analyst.brains.call_for",
                        lambda role: (_ for _ in ()).throw(AssertionError("called")))
    args = type("A", (), {"max_assets": 1, "mode": "solo", "depth": None,
                          "horizons": "1", "report_dir": "x"})()
    assert analyst.cmd_auto(args) == 1


def test_the_loop_step_is_brief_and_bounded(monkeypatch):
    import loop_cycle
    monkeypatch.setenv("GTRADE_ANALYST_AUTO", "1")
    seen = {}
    monkeypatch.setattr(loop_cycle.subprocess, "run",
                        lambda cmd, **k: seen.update(cmd=cmd, **k))
    loop_cycle.analyst_auto_step()
    assert seen["cmd"][-2:] == ["--depth", "brief"] and seen["timeout"] > 0


def test_the_report_shows_each_pick_at_its_own_bar_date(tmp_path):
    from core.analyst import report, store
    path = str(tmp_path / "m.db")
    store.write_judgment({"date": "2026-09-27", "asset": "BTC", "horizon": 1,
                          "direction": "up", "conviction": 2, "mode": "auto-solo"}, db_path=path)
    store.write_judgment({"date": "2026-09-25", "asset": "NVDA", "horizon": 1,
                          "direction": "down", "conviction": 4, "mode": "auto-solo"},
                         db_path=path)
    picks = [{"asset": "BTC", "reason": "a|b\nc", "by": "scout"},
             {"asset": "NVDA", "reason": "r", "by": "scout"}]
    out = report.write_daily("2026-09-28", picks, mode_label="auto-solo",
                             path_dir=str(tmp_path), db_path=path)
    text = open(out, encoding="utf-8").read()
    assert out.endswith("analyst_2026-09-28.md")
    assert "down, conviction 4" in text and "up, conviction 2" in text
    assert "a/b c" in text
