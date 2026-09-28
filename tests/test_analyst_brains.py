import os

import pytest

from core.analyst import brains


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith("GTRADE_ANALYST_BRAIN") or k in (
                "GTRADE_AR_LLM", "GTRADE_AR_LLM_MODEL", "GTRADE_AR_LLM_BASE_URL"):
            monkeypatch.delenv(k)
    monkeypatch.setattr(brains, "SPEED_PATH", str(tmp_path / "speed.json"))


def test_parse_keeps_colons_in_the_model_name():
    assert brains.parse("ollama:gemma4:12b") == ("ollama", "gemma4:12b")
    assert brains.parse("ollama-cloud:gpt-oss:120b") == ("ollama-cloud", "gpt-oss:120b")
    assert brains.parse("anthropic") == ("anthropic", None)
    with pytest.raises(ValueError):
        brains.parse("vibes:x")


def test_a_role_setting_beats_the_default_and_unset_means_legacy(monkeypatch):
    assert brains.spec_for("lead") is None and brains.env_for("lead") == {}
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN", "ollama:gemma4:12b")
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN_LEAD", "ollama-cloud:gpt-oss:120b")
    assert brains.env_for("macro")["GTRADE_AR_LLM_MODEL"] == "gemma4:12b"
    lead = brains.env_for("lead")
    assert lead == {"GTRADE_AR_LLM": "ollama", "GTRADE_AR_LLM_MODEL": "gpt-oss:120b",
                    "GTRADE_AR_LLM_BASE_URL": "https://ollama.com"}
    assert brains.label("lead") == "ollama-cloud:gpt-oss:120b"


def test_the_call_sees_its_env_and_restores_it_even_on_error(monkeypatch):
    seen = {}

    def fake_backend(what):
        def f(prompt):
            seen.update(llm=os.getenv("GTRADE_AR_LLM"), url=os.getenv("GTRADE_AR_LLM_BASE_URL"))
            if prompt == "boom":
                raise RuntimeError("down")
            return "ok"
        return f

    monkeypatch.setattr("core.llm_proposer._backend", fake_backend)
    monkeypatch.setenv("GTRADE_AR_LLM", "anthropic")
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN_CRITIC", "ollama-cloud:gpt-oss:120b")
    call = brains.call_for("critic")
    assert call("hi") == "ok" and seen == {"llm": "ollama", "url": "https://ollama.com"}
    with pytest.raises(RuntimeError):
        call("boom")
    assert os.getenv("GTRADE_AR_LLM") == "anthropic"
    assert os.getenv("GTRADE_AR_LLM_BASE_URL") is None


def test_speed_is_recorded_per_brain_and_estimates_hours(monkeypatch):
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN", "ollama:gemma4:12b")
    brains._record_speed(brains.label("solo"), 600.0)
    assert brains.seconds_per_call("solo") == 600.0
    assert brains.estimate_hours({"solo": 12}) == 2.0
    assert brains.estimate_hours({"lead": 3, "scout": 1}) == pytest.approx(4 * 600 / 3600)
    assert brains.estimate_hours({"solo": 1, "critic": 1}) == pytest.approx(1200 / 3600)
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN_CRITIC", "anthropic:claude-sonnet-5")
    assert brains.estimate_hours({"critic": 1}) is None     # never timed


def test_the_brains_command_lists_every_role_and_pings_each_brain_once(monkeypatch, capsys):
    import analyst
    calls = []
    monkeypatch.setattr("core.llm_proposer._backend", lambda what: (lambda p: calls.append(p) or "OK"))
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN", "ollama:gemma4:12b")
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN_LEAD", "ollama-cloud:gpt-oss:120b")
    assert analyst.main(["brains", "--ping"]) == 0
    out = capsys.readouterr().out
    assert all(r in out for r in brains.ROLES) and len(calls) == 2


def test_brains_set_and_unset_persist_to_env_through_a_whitelist(monkeypatch, tmp_path):
    import analyst
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n", encoding="utf-8")
    monkeypatch.setattr(brains, "ENV_PATH", str(env))
    assert analyst.main(["brains", "--set", "critic=ollama-cloud:gpt-oss:120b"]) == 0
    assert analyst.main(["brains", "--set", "GTRADE_OLLAMA_MIN_FREE_MB=3000"]) == 0
    text = env.read_text(encoding="utf-8")
    assert "GTRADE_ANALYST_BRAIN_CRITIC='ollama-cloud:gpt-oss:120b'" in text
    assert "GTRADE_OLLAMA_MIN_FREE_MB='3000'" in text and "OTHER=1" in text
    assert os.getenv("GTRADE_ANALYST_BRAIN_CRITIC") == "ollama-cloud:gpt-oss:120b"
    assert analyst.main(["brains", "--set", "critic=vibes:x"]) == 1          # bad provider
    assert analyst.main(["brains", "--set", "PATH=C:/evil"]) == 1            # not whitelisted
    assert analyst.main(["brains", "--unset", "critic"]) == 0
    assert "BRAIN_CRITIC" not in env.read_text(encoding="utf-8")
    assert os.getenv("GTRADE_ANALYST_BRAIN_CRITIC") is None


def test_the_cloud_key_is_read_hidden_and_saved(monkeypatch, tmp_path):
    import analyst
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(brains, "ENV_PATH", str(env))
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)   # restored after the test
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "  sk-abc  ")
    assert analyst.main(["brains", "--cloud-key"]) == 0
    assert "OLLAMA_API_KEY='sk-abc'" in env.read_text(encoding="utf-8")


def test_numeric_menu_settings_are_validated(monkeypatch, tmp_path):
    import analyst
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(brains, "ENV_PATH", str(env))
    for bad in ("GTRADE_ANALYST_MAX_HOURS=8h", "GTRADE_OLLAMA_MIN_FREE_MB=-5",
                "GTRADE_ANALYST_TOOL_ROUNDS=two"):
        assert analyst.main(["brains", "--set", bad]) == 1
    assert env.read_text(encoding="utf-8") == ""


def test_the_combiner_setting_is_validated(monkeypatch, tmp_path):
    import analyst
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(brains, "ENV_PATH", str(env))
    monkeypatch.delenv("GTRADE_COMBINER", raising=False)
    assert analyst.main(["brains", "--set", "GTRADE_COMBINER=blend"]) == 1
    assert analyst.main(["brains", "--set", "GTRADE_COMBINER=stack"]) == 0
    assert "GTRADE_COMBINER='stack'" in env.read_text(encoding="utf-8")
