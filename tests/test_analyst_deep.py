import json
import sqlite3

from core.analyst import agent, store

DOSSIER = {"asset": "SP500", "date": "2026-08-20", "close": 100.0, "atr": 1.0,
           "rsi_14": 55.0, "ret_20d": 2.0}


def _judgment(direction):
    return json.dumps({"direction": direction, "conviction": 3, "vol_regime": "normal",
                       "key_risk": "r", "thesis": "t", "evidence": ["rsi_14"]})


def test_a_plan_sent_with_the_tool_request_is_kept(monkeypatch):
    monkeypatch.setenv("GTRADE_ANALYST_REQUIRE_TOOL", "1")
    replies = iter([json.dumps({"plan": "check momentum first",
                                "tools": [{"tool": "price_history", "args": {"days": 5}}]}),
                    _judgment("up")])
    notes = {}
    j = agent.judge(DOSSIER, call=lambda p: next(replies), tool_calls=[], notes=notes,
                    today="2026-08-20")
    assert j["direction"] == "up" and notes["plan"] == "check momentum first"


def test_the_menu_tells_the_model_it_may_send_a_plan():
    from core.analyst import tools
    assert '"plan"' in tools.spec_lines(None, "SP500")


def test_the_critic_can_overturn_and_a_dead_critic_keeps_the_verdict():
    first = agent.parse_judgment(_judgment("up"))
    seen = []

    def critic(p):
        seen.append(p)
        return _judgment("down")

    assert agent.critique(DOSSIER, first, critic)["direction"] == "down"
    assert "strongest objection" in seen[0] and '"direction": "up"' in seen[0]
    assert agent.critique(DOSSIER, first, lambda p: "no json here") is None


def test_the_new_columns_round_trip_and_migrate_an_old_table(tmp_path):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    con.execute(  # the table as it stood before mode/brain/plan/pre_critic_json
        "CREATE TABLE analyst_log (date TEXT, asset TEXT, horizon INTEGER, "
        "direction TEXT, conviction INTEGER, vol_regime TEXT, key_risk TEXT, "
        "thesis TEXT, evidence_json TEXT, dossier_hash TEXT, llm_model TEXT, "
        "forecast_pct REAL, lo_pct REAL, hi_pct REAL, atr_at_signal REAL, "
        "close_at_signal REAL, realized_ret REAL, realized_atr_units REAL, "
        "inside_interval INTEGER, abs_err_atr REAL, tool_calls_json TEXT, "
        "PRIMARY KEY (date, asset, horizon))")
    con.commit()
    con.close()
    store.write_judgment({"date": "2026-08-20", "asset": "SP500", "horizon": 1,
                          "direction": "down", "mode": "solo", "brain": "ollama:gemma4:12b",
                          "plan": "p", "pre_critic_json": _judgment("up")}, db_path=path)
    row = sqlite3.connect(path).execute(
        "SELECT mode, brain, plan, pre_critic_json FROM analyst_log").fetchone()
    assert row[0] == "solo" and row[1] == "ollama:gemma4:12b" and "up" in row[3]


def test_judge_one_writes_mode_brain_and_the_pre_critic_verdict(tmp_path, monkeypatch):
    import analyst
    from core.analyst import brains

    path = str(tmp_path / "m.db")
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN", "ollama:gemma4:12b")
    monkeypatch.setattr(brains, "call_for", lambda role: (lambda p: _judgment("down")))
    monkeypatch.setattr(analyst.calibrate, "forecast",
                        lambda *a, **k: {"pct": None, "lo": None, "hi": None})
    monkeypatch.setattr(analyst, "_print_judgment", lambda *a, **k: None)
    for first in ("up", "down"):
        written, refused = analyst._judge_one(
            dict(DOSSIER, asset="SP500"), "SP500" if first == "up" else "GOLD",
            "h-" + first, 1, lambda p, f=first: _judgment(f), "deep", {}, {}, 0, 0)
        assert (written, refused) == (1, 0)
    rows = sqlite3.connect(path).execute(
        "SELECT direction, mode, brain, pre_critic_json FROM analyst_log").fetchall()
    assert len(rows) == 2
    assert all(r[0] == "down" and r[1] == "solo" and r[2] == "ollama:gemma4:12b" for r in rows)
    assert any(r[3] and '"up"' in r[3] for r in rows)


def test_a_critic_that_times_out_stops_the_run_instead_of_every_asset_paying():
    import pytest

    from core.llm_proposer import CallTimedOut, ProviderUnavailable
    first = agent.parse_judgment(_judgment("up"))
    for exc in (CallTimedOut("slow"), ProviderUnavailable("no key")):
        def dead(p, e=exc):
            raise e
        with pytest.raises(type(exc)):
            agent.critique(DOSSIER, first, dead)


def test_llm_flag_keeps_the_configured_model_and_base(monkeypatch):
    import analyst
    for k in ("GTRADE_ANALYST_BRAIN", "GTRADE_ANALYST_BRAIN_SOLO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "gemma4:26b")
    monkeypatch.setenv("GTRADE_AR_LLM_BASE_URL", "http://192.168.1.5:11434/v1")
    monkeypatch.setenv("GTRADE_AR_LLM", "ollama")      # cmd_run sets this from --llm first
    analyst._apply_llm_flag("ollama", None)
    from core.analyst import brains
    assert brains.env_for("solo") == {}                      # legacy path, .env intact
    assert brains.label("solo") == "ollama:gemma4:26b"
    analyst._apply_llm_flag("ollama-cloud", None)
    assert brains.env_for("solo")["GTRADE_AR_LLM_MODEL"] == "gemma4:26b"


def test_a_judgment_with_a_fallback_call_says_so_in_its_brain(tmp_path, monkeypatch):
    """The brain column must show when any call of the judgment (a specialist,
    the lead, the critic) was answered by the fallback, or two brains mix
    unseen in one sample."""
    import analyst
    from core.analyst import brains

    path = str(tmp_path / "m.db")
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN", "claude-code:opus")

    def critic_on_fallback(role):
        def call(p):
            brains._FALLBACKS += 1              # what call_for does on a fallback
            return _judgment("down")
        return call

    monkeypatch.setattr(brains, "call_for", critic_on_fallback)
    monkeypatch.setattr(analyst.calibrate, "forecast",
                        lambda *a, **k: {"pct": None, "lo": None, "hi": None})
    monkeypatch.setattr(analyst, "_print_judgment", lambda *a, **k: None)
    analyst._judge_one(dict(DOSSIER, asset="SP500"), "SP500", "h", 1,
                       lambda p: _judgment("up"), "deep", {}, {}, 0, 0)
    brain = sqlite3.connect(path).execute("SELECT brain FROM analyst_log").fetchone()[0]
    assert brain == "claude-code:opus +1 fallback"
