import json
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
