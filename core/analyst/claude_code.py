"""Claude Code as an analyst brain, on a Claude subscription.

One call is one headless `claude -p` run: the prompt on stdin, the answer in
the JSON reply's `result`. It may search and read the web with its own tools,
and nothing else: no shell, no file edits, an empty working folder, no MCP
servers, no saved session.

It must never bill the API. With ANTHROPIC_API_KEY in the environment Claude
Code uses the key instead of the subscription login, so both API variables are
removed from the child's environment, and `--bare` is not used because it
forces key auth. Calls are capped per day (GTRADE_ANALYST_CLAUDE_MAX_CALLS),
since subscription limits are finite and shared with interactive use.
"""

import datetime
import json
import os
import subprocess
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAP_PATH = os.path.join(BASE, "_analyst_claude_calls.json")
_API_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class ClaudeUnavailable(Exception):
    """No answer from Claude Code: not installed, a limit, a timeout, an error."""


def _int(name, default):
    try:
        return max(1, int(os.getenv(name) or default))
    except ValueError:
        return default


def model():
    return (os.getenv("GTRADE_ANALYST_CLAUDE_MODEL") or "sonnet").strip()


def turns():
    return _int("GTRADE_ANALYST_CLAUDE_TURNS", 12)


def timeout():
    return _int("GTRADE_ANALYST_CLAUDE_TIMEOUT", 600)


def cap():
    return _int("GTRADE_ANALYST_CLAUDE_MAX_CALLS", 20)


def spend(today=None):
    """Count one call against today's cap; False when the cap is reached."""
    today = today or datetime.date.today().isoformat()
    try:
        with open(CAP_PATH, encoding="utf-8") as fh:
            used = int(json.load(fh).get(today, 0))
    except (OSError, ValueError, AttributeError):
        used = 0
    if used >= cap():
        return False
    try:
        with open(CAP_PATH, "w", encoding="utf-8") as fh:
            json.dump({today: used + 1}, fh)
    except OSError:
        pass
    return True


def command(model_name, max_turns):
    return ["claude", "-p", "--output-format", "json", "--model", model_name,
            "--max-turns", str(max_turns),
            "--allowedTools", "WebSearch,WebFetch",
            "--disallowedTools", "Bash,Edit,Write,NotebookEdit,Task",
            "--strict-mcp-config", "--no-session-persistence",
            "--setting-sources", "project"]


def run(prompt, model_name, max_turns, seconds, runner=subprocess.run):
    """The answer text, or ClaudeUnavailable."""
    env = {k: v for k, v in os.environ.items() if k not in _API_VARS}
    with tempfile.TemporaryDirectory(prefix="analyst_cc_") as work:
        try:
            r = runner(command(model_name, max_turns), input=prompt, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=seconds,
                       cwd=work, env=env)
        except FileNotFoundError as exc:
            raise ClaudeUnavailable("the claude CLI is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeUnavailable("no answer in %d s" % seconds) from exc
    try:
        data = json.loads(r.stdout)
    except (TypeError, ValueError) as exc:
        raise ClaudeUnavailable(((r.stderr or "") + (r.stdout or ""))[:300]
                                or "empty reply") from exc
    if data.get("is_error") or not isinstance(data.get("result"), str):
        raise ClaudeUnavailable(str(data.get("result") or data.get("subtype"))[:300])
    return data["result"]
