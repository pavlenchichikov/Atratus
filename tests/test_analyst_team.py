import json

import pytest

from core.analyst import dossier as dz
from core.analyst import team, tools

D = {"asset": "BTC", "date": "2026-09-20", "close": 1.0, "atr": 0.1, "ret_1": 0.0,
     "ret_5": 1.0, "ret_20": 2.0, "rsi_14": 60.0, "vix_level": 18.0, "pe": None,
     "pb": None, "headlines": [], "news_publishers": 0, "policy_rate": 5.0}


def test_a_slice_keeps_common_plus_its_own_fields_and_no_model_signal():
    s = team.slice_dossier(dict(D, signal="BUY", probability=0.7), "technical")
    assert {"asset", "close", "rsi_14"} <= set(s) and "vix_level" not in s
    assert not set(s) & dz.FORBIDDEN_KEYS
    assert "vix_level" in team.slice_dossier(D, "macro")


def test_fundamental_has_nothing_to_read_on_crypto():
    assert not team.has_material(D, "fundamental")
    assert team.has_material(D, "technical")


def test_the_menu_can_be_limited_to_a_role():
    names = {t.name for t in tools.available(None, "SP500", only=("price_history",))}
    assert names == {"price_history"}
    assert "news_search" not in tools.spec_lines(None, "SP500", only=("price_history",))


def test_parse_report_validates_and_drops_unknown_evidence():
    ok = team.parse_report(json.dumps({"lean": "up", "strength": 4,
                                       "findings": ["rsi high", "x" * 999],
                                       "evidence": ["rsi_14", "made_up"]}), {"rsi_14"})
    assert ok["lean"] == "up" and ok["strength"] == 4 and ok["evidence"] == ["rsi_14"]
    assert len(ok["findings"][1]) == 300
    assert team.parse_report('{"lean": "sideways", "strength": 2}', {"rsi_14"}) is None
    assert team.parse_report('{"lean": "up", "strength": 9, "evidence": ["rsi_14"]}',
                             {"rsi_14"}) is None
    assert team.parse_report('{"lean": "up", "strength": 2, "evidence": ["made_up"]}',
                             {"rsi_14"}) is None



def _rep(lean, ev="rsi_14"):
    return json.dumps({"lean": lean, "strength": 3, "findings": ["f"], "evidence": [ev]})


def test_consult_runs_only_its_own_tools_then_reports(monkeypatch):
    ran = []
    monkeypatch.setattr(tools, "call", lambda req, asset, today=None: ran.append(req["tool"])
                        or {"tool": req["tool"], "result": {"ok": 1}})
    replies = iter([json.dumps({"tools": [{"tool": "price_history", "args": {}},
                                          {"tool": "news_search", "args": {}}]}),
                    _rep("up")])
    out = team.consult("technical", dict(D, asset="SP500"), lambda p: next(replies),
                       tool_calls=[])
    assert out["lean"] == "up" and ran == ["price_history"]


def test_a_specialist_that_fails_twice_is_skipped_and_the_lead_is_told(monkeypatch):
    prompts = {}

    def call_for(role):
        def f(p):
            prompts.setdefault(role, []).append(p)
            if role == "macro":
                return "garbage"
            return _rep("down", "rsi_14") if role == "technical" else _rep("up", "headlines")
        return f

    lead_seen = []

    def lead(p):
        lead_seen.append(p)
        return json.dumps({"direction": "down", "conviction": 2, "vol_regime": "normal",
                           "key_risk": "r", "thesis": "t", "evidence": ["rsi_14"]})

    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    j, reports = team.run_team(D, lead, call_for)
    roles = {r["role"]: r for r in reports}
    assert j["direction"] == "down"
    assert roles["macro"]["skipped"] and roles["fundamental"]["skipped"] == "nothing to read"
    assert "fundamental" not in prompts                      # skipped without a call
    assert len(prompts["macro"]) == team.MAX_TRIES
    assert roles["technical"]["lean"] == "down"
    assert "macro: skipped" in lead_seen[0] and '"lean": "down"' in lead_seen[0]


def test_a_timeout_in_a_specialist_stops_the_team():
    from core.llm_proposer import CallTimedOut

    def call_for(role):
        def f(p):
            raise CallTimedOut("slow")
        return f

    with pytest.raises(CallTimedOut):
        team.run_team(D, lambda p: "", call_for)


def test_team_rows_round_trip_and_score_only_resolved_outcomes(tmp_path):
    import sqlite3

    from core.analyst import score, store
    path = str(tmp_path / "m.db")
    store.write_judgment({"date": "2026-09-20", "asset": "SP500", "horizon": 1,
                          "direction": "down", "mode": "team"}, db_path=path)
    store.write_judgment({"date": "2026-09-21", "asset": "SP500", "horizon": 1,
                          "direction": "up", "mode": "team"}, db_path=path)
    reps = [{"role": "technical", "brain": "b", "lean": "down", "strength": 3,
             "findings": [], "evidence": ["rsi_14"]},
            {"role": "macro", "brain": "b", "skipped": "no usable report"}]
    store.write_team_reports("2026-09-20", "SP500", 1, reps, db_path=path)
    store.write_team_reports("2026-09-21", "SP500", 1, reps, db_path=path)
    con = sqlite3.connect(path)
    con.execute("UPDATE analyst_log SET realized_ret=-0.01 WHERE date='2026-09-20'")
    con.commit()
    con.close()
    rows = store.team_scored_rows(db_path=path)
    assert {(r["date"], r["role"]) for r in rows} == {("2026-09-20", "technical"),
                                                     ("2026-09-20", "macro")}
    by = score.by_role(rows)
    assert by["technical"] == {"n": 1, "hit": 1.0} and by["macro"]["n"] == 0


def _lead(d):
    return lambda p: json.dumps({"direction": d, "conviction": 3, "vol_regime": "normal",
                                 "key_risk": "r", "thesis": "t", "evidence": ["rsi_14"]})


def test_team_mode_writes_one_judgment_and_one_row_per_role_in_both_directions(
        tmp_path, monkeypatch):
    import sqlite3

    import analyst
    from core.analyst import store
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    monkeypatch.setattr(analyst.calibrate, "forecast",
                        lambda *a, **k: {"pct": None, "lo": None, "hi": None})
    monkeypatch.setattr(analyst, "_print_judgment", lambda *a, **k: None)
    for asset, d in (("SP500", "up"), ("GOLD", "down")):
        monkeypatch.setattr("core.analyst.brains.call_for",
                            lambda role, d=d: (lambda p: _rep(d)))
        w, r = analyst._judge_one(dict(D, asset=asset), asset, "h" + asset, 1, _lead(d),
                                  "full", {}, {}, 0, 0, mode="team")
        assert (w, r) == (1, 0)
    con = sqlite3.connect(path)
    assert sorted(con.execute("SELECT asset, direction, mode FROM analyst_log")) == [
        ("GOLD", "down", "team"), ("SP500", "up", "team")]
    assert con.execute("SELECT COUNT(*) FROM analyst_team_log").fetchone()[0] == 8


def test_a_refused_lead_writes_nothing(tmp_path, monkeypatch):
    import sqlite3

    import analyst
    from core.analyst import store
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    monkeypatch.setattr("core.analyst.brains.call_for", lambda role: (lambda p: _rep("up")))
    w, r = analyst._judge_one(dict(D, asset="SP500"), "SP500", "h", 1, lambda p: "no",
                              "full", {}, {}, 0, 0, mode="team")
    assert (w, r) == (0, 1)
    con = sqlite3.connect(path)
    for t in ("analyst_log", "analyst_team_log"):
        try:
            assert con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0
        except sqlite3.OperationalError:
            pass                                   # table never created: nothing written


def test_llm_flag_in_team_mode_reaches_every_role(monkeypatch):
    import analyst
    from core.analyst import brains
    for k in ("GTRADE_ANALYST_BRAIN", "GTRADE_ANALYST_BRAIN_SOLO", "GTRADE_ANALYST_BRAIN_LEAD"):
        monkeypatch.delenv(k, raising=False)
    analyst._apply_llm_flag("ollama-cloud", "gpt-oss:120b", team=True)
    for role in ("lead",) + team.SPECIALISTS:
        assert brains.label(role) == "ollama-cloud:gpt-oss:120b"
    monkeypatch.delenv("GTRADE_ANALYST_BRAIN", raising=False)


def test_specialist_retries_are_reported():
    seen = []
    team.consult("technical", dict(D, asset="SP500"), lambda p: "junk",
                 on_reject=seen.append)
    assert len(seen) == team.MAX_TRIES and all(s.startswith("technical:") for s in seen)


def test_solo_and_team_do_not_share_the_dossier_skip(tmp_path):
    from core.analyst import store
    path = str(tmp_path / "m.db")
    store.write_judgment({"date": "2026-09-20", "asset": "SP500", "horizon": 1,
                          "direction": "up", "dossier_hash": "h", "mode": "solo"},
                         db_path=path)
    assert store.judged_with_hash("SP500", "h", db_path=path, horizon=1, mode="solo")
    assert not store.judged_with_hash("SP500", "h", db_path=path, horizon=1, mode="team")
