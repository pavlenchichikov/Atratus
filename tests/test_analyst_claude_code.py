import json
import os
import subprocess

import pytest

from core.analyst import brains
from core.analyst import claude_code as cc


class _Runner:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        out = self.replies.pop(0)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")


def test_the_command_is_isolated_and_bills_the_subscription(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    run = _Runner([{"result": "OK", "is_error": False}])
    assert cc.run("hi", "sonnet", 3, 60, runner=run) == "OK"
    cmd, kw = run.calls[0]
    assert "--bare" not in cmd
    assert cmd[cmd.index("--allowedTools") + 1] == "WebSearch,WebFetch"
    assert "Bash" in cmd[cmd.index("--disallowedTools") + 1]
    assert "ANTHROPIC_API_KEY" not in kw["env"] and "ANTHROPIC_AUTH_TOKEN" not in kw["env"]
    assert kw["input"] == "hi"


def test_an_error_reply_or_a_missing_cli_is_unavailable():
    with pytest.raises(cc.ClaudeUnavailable):
        cc.run("hi", "sonnet", 3, 60, runner=_Runner([{"result": "limit reached",
                                                       "is_error": True}]))

    def missing(cmd, **kw):
        raise FileNotFoundError("claude")

    with pytest.raises(cc.ClaudeUnavailable):
        cc.run("hi", "sonnet", 3, 60, runner=missing)


def test_the_daily_cap_counts_per_date(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CAP_PATH", str(tmp_path / "calls.json"))
    monkeypatch.setenv("GTRADE_ANALYST_CLAUDE_MAX_CALLS", "2")
    assert cc.spend("2026-09-29") and cc.spend("2026-09-29")
    assert not cc.spend("2026-09-29")
    assert cc.spend("2026-09-30")


def test_over_the_cap_the_fallback_answers_and_the_label_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CAP_PATH", str(tmp_path / "calls.json"))
    monkeypatch.setattr(brains, "SPEED_PATH", str(tmp_path / "speed.json"))
    monkeypatch.setenv("GTRADE_ANALYST_CLAUDE_MAX_CALLS", "1")
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN", "claude-code:opus")
    monkeypatch.setenv("GTRADE_ANALYST_FALLBACK", "ollama:gemma")
    monkeypatch.setattr(cc, "run", lambda prompt, model, turns, timeout: "from claude " + model)
    monkeypatch.setattr(brains, "_plain_call", lambda role, spec: (lambda p: "from ollama"))
    call = brains.call_for("solo")
    assert call("x") == "from claude opus" and call.last_label == "claude-code:opus"
    assert call("x") == "from ollama"
    assert call.last_label == "ollama:gemma (fallback)"


def test_claude_code_parses_as_a_provider():
    assert brains.parse("claude-code:opus") == ("claude-code", "opus")


def test_llm_flag_claude_code_sets_the_solo_brain_only(monkeypatch):
    import analyst

    for k in ("GTRADE_ANALYST_BRAIN_SOLO", "GTRADE_ANALYST_BRAIN", "GTRADE_ANALYST_BRAIN_CRITIC"):
        monkeypatch.setenv(k, "x")      # recorded, so the flag's writes are undone
        monkeypatch.delenv(k)
    monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "gemma4:26b")
    analyst._apply_llm_flag("claude-code", "opus")
    assert brains.spec_for("solo") == "claude-code:opus"
    analyst._apply_llm_flag("claude-code", None)
    assert brains.label("solo") == "claude-code:" + cc.model()
    analyst._apply_llm_flag("claude-code", None, team=True)
    assert brains.spec_for("critic").startswith("claude-code")


def test_the_run_parser_accepts_claude_code():
    import analyst

    ns = analyst.build_parser().parse_args(["run", "--llm", "claude-code", "--model", "opus"])
    assert ns.llm == "claude-code"


def test_a_rewound_run_gets_no_web_tools(monkeypatch):
    """A judgment as of a past date must not read today's web: that is look-ahead."""
    monkeypatch.setenv("GTRADE_ANALYST_REWIND", "1")
    run = _Runner([{"result": "OK", "is_error": False}])
    cc.run("hi", "opus", 3, 60, runner=run)
    cmd = run.calls[0][0]
    assert "--allowedTools" not in cmd
    denied = cmd[cmd.index("--disallowedTools") + 1]
    assert "WebSearch" in denied and "WebFetch" in denied


def test_cmd_run_marks_a_rewound_run_and_clears_the_mark(monkeypatch):
    import analyst

    monkeypatch.setenv("GTRADE_ANALYST_REWIND", "x")      # recorded, so undone after
    analyst._mark_rewind("2026-07-01")
    assert os.environ["GTRADE_ANALYST_REWIND"] == "1"
    analyst._mark_rewind(None)
    assert "GTRADE_ANALYST_REWIND" not in os.environ


def _claude_down(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CAP_PATH", str(tmp_path / "calls.json"))
    monkeypatch.setattr(brains, "SPEED_PATH", str(tmp_path / "speed.json"))
    monkeypatch.setenv("GTRADE_ANALYST_BRAIN", "claude-code:opus")

    def down(*a, **k):
        raise cc.ClaudeUnavailable("usage limit reached")

    monkeypatch.setattr(cc, "run", down)


def test_with_no_fallback_set_a_dead_claude_stops_the_run(monkeypatch, tmp_path):
    """A local model started beside a training crashed it (2026-09-30), and
    a silent swap mixes two brains in one sample. Unset means: stop."""
    from core.llm_proposer import ProviderUnavailable

    _claude_down(monkeypatch, tmp_path)
    monkeypatch.delenv("GTRADE_ANALYST_FALLBACK", raising=False)
    with pytest.raises(ProviderUnavailable, match="usage limit"):
        brains.call_for("solo")("x")
    monkeypatch.setenv("GTRADE_ANALYST_FALLBACK", "none")
    with pytest.raises(ProviderUnavailable):
        brains.call_for("solo")("x")


def test_every_fallback_is_counted(monkeypatch, tmp_path):
    _claude_down(monkeypatch, tmp_path)
    monkeypatch.setenv("GTRADE_ANALYST_FALLBACK", "ollama:gemma")
    monkeypatch.setattr(brains, "_plain_call", lambda role, spec: (lambda p: "ok"))
    before = brains.fallback_count()
    brains.call_for("critic")("x")
    brains.call_for("lead")("x")
    assert brains.fallback_count() - before == 2


def test_none_is_a_valid_fallback_setting():
    assert brains._VALID["GTRADE_ANALYST_FALLBACK"]("none")
    assert not brains._VALID["GTRADE_ANALYST_FALLBACK"]("claude-code:opus")


def test_the_claude_model_setting_takes_an_alias_or_a_full_id():
    ok = brains._VALID["GTRADE_ANALYST_CLAUDE_MODEL"]
    assert ok("opus") and ok("claude-opus-5-5")
    assert not ok("gpt-4o") and not ok("")
