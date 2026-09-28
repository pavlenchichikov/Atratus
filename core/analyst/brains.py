"""Which LLM thinks for which analyst role.

One setting per role, `provider:model`:

    GTRADE_ANALYST_BRAIN=ollama:gemma4:12b              default for every role
    GTRADE_ANALYST_BRAIN_LEAD=ollama-cloud:gpt-oss:120b
    GTRADE_ANALYST_BRAIN_CRITIC=anthropic:claude-sonnet-5

Unset everywhere means the old behaviour exactly: the call goes through
llm_proposer._backend("analyst") on GTRADE_AR_LLM / GTRADE_AR_LLM_MODEL.
A brain is applied by setting those same variables for the duration of the
call and restoring them afterwards, so every retry, trace and timeout rule in
llm_proposer applies unchanged. Calls are sequential; nothing here is
thread-safe on purpose.
"""

import json
import os
import time
from contextlib import contextmanager

ROLES = ("solo", "lead", "macro", "fundamental", "technical", "news",
         "critic", "scout", "memory")
PROVIDERS = ("ollama", "ollama-cloud", "anthropic", "openai")
CLOUD_BASE = "https://ollama.com"
LOCAL_BASE = "http://127.0.0.1:11434/v1"
_KEYS = ("GTRADE_AR_LLM", "GTRADE_AR_LLM_MODEL", "GTRADE_AR_LLM_BASE_URL")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEED_PATH = os.path.join(BASE, "_analyst_brain_speed.json")
ENV_PATH = os.path.join(BASE, ".env")

# What the menu may write into .env, and nothing else: the .bat passes free
# text, so an unlisted key is refused rather than written.
SETTABLE = ("GTRADE_ANALYST_BRAIN", "GTRADE_OLLAMA_MIN_FREE_MB", "GTRADE_OLLAMA_NUM_GPU",
            "GTRADE_ANALYST_MAX_HOURS", "GTRADE_ANALYST_TOOL_ROUNDS",
            "GTRADE_ANALYST_TOOL_CALLS", "GTRADE_ANALYST_OLLAMA_URL", "OLLAMA_API_KEY",
            "GTRADE_ANALYST_AUTO", "GTRADE_ANALYST_AUTO_MAX", "GTRADE_ANALYST_MODE")


def parse(spec):
    """("provider", "model" or None). The model keeps its own colons."""
    provider, _, model = str(spec).strip().partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError("unknown provider %r in %r (use %s)"
                         % (provider, spec, ", ".join(PROVIDERS)))
    return provider, (model.strip() or None)


def spec_for(role):
    return (os.getenv("GTRADE_ANALYST_BRAIN_" + role.upper())
            or os.getenv("GTRADE_ANALYST_BRAIN") or None)


def env_for(role):
    """The variables this role's calls run under; {} means legacy."""
    spec = spec_for(role)
    if not spec:
        return {}
    provider, model = parse(spec)
    env = {"GTRADE_AR_LLM": "ollama" if provider.startswith("ollama") else provider,
           "GTRADE_AR_LLM_MODEL": model or "auto"}
    if provider == "ollama-cloud":
        env["GTRADE_AR_LLM_BASE_URL"] = CLOUD_BASE
    elif provider == "ollama":
        env["GTRADE_AR_LLM_BASE_URL"] = os.getenv("GTRADE_ANALYST_OLLAMA_URL") or LOCAL_BASE
    return env


def label(role):
    spec = spec_for(role)
    if spec:
        provider, model = parse(spec)
        return "%s:%s" % (provider, model or "auto")
    return ":".join(v for v in (os.getenv("GTRADE_AR_LLM", "anthropic"),
                                os.getenv("GTRADE_AR_LLM_MODEL")) if v)


@contextmanager
def _env(overrides):
    saved = {k: os.environ.get(k) for k in _KEYS}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load_speed():
    try:
        with open(SPEED_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _record_speed(brain_label, seconds):
    """Mean seconds per call, per brain: an exponential average, weight 0.3."""
    data = _load_speed()
    old = data.get(brain_label)
    data[brain_label] = seconds if old is None else 0.7 * old + 0.3 * seconds
    try:
        with open(SPEED_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
    except OSError:
        pass


def seconds_per_call(role):
    return _load_speed().get(label(role))


def estimate_hours(calls):
    """Hours for {role: n_calls}, or None when any brain was never timed."""
    total = 0.0
    for role, n in calls.items():
        s = seconds_per_call(role)
        if s is None:
            return None
        total += s * n
    return total / 3600.0


def call_for(role):
    """f(prompt) -> str running under this role's brain."""
    from core import llm_proposer

    def call(prompt):
        with _env(env_for(role)):
            fn = llm_proposer._backend("analyst")
            t0 = time.monotonic()
            out = fn(prompt)
            _record_speed(label(role), time.monotonic() - t0)
            return out

    return call


def _positive_int(v):
    return v.isdigit() and int(v) > 0


# A bad number in .env would stop every later run at startup, so the menu
# refuses it instead of saving it.
_VALID = {
    "GTRADE_OLLAMA_MIN_FREE_MB": _positive_int,
    "GTRADE_OLLAMA_NUM_GPU": str.isdigit,
    "GTRADE_ANALYST_MAX_HOURS": _positive_int,
    "GTRADE_ANALYST_TOOL_ROUNDS": str.isdigit,
    "GTRADE_ANALYST_TOOL_CALLS": str.isdigit,
    "GTRADE_ANALYST_AUTO": lambda v: v in ("0", "1"),
    "GTRADE_ANALYST_AUTO_MAX": _positive_int,
    "GTRADE_ANALYST_MODE": lambda v: v in ("solo", "team"),
}


def env_key(name):
    """A role name ("critic", "default") or a SETTABLE key -> the .env key, or None."""
    name = str(name).strip()
    if name.lower() == "default":
        return "GTRADE_ANALYST_BRAIN"
    if name.lower() in ROLES:
        return "GTRADE_ANALYST_BRAIN_" + name.upper()
    return name if name in SETTABLE else None


def persist(key, value):
    """Write key=value into .env and this process. Raises ValueError on a bad pair."""
    from dotenv import set_key

    if key.startswith("GTRADE_ANALYST_BRAIN"):
        parse(value)
    value = str(value).strip()
    if not value:
        raise ValueError("empty value for %s" % key)
    if key in _VALID and not _VALID[key](value):
        raise ValueError("invalid value %r for %s" % (value, key))
    set_key(ENV_PATH, key, value)
    os.environ[key] = value


def forget(key):
    from dotenv import unset_key

    if os.path.exists(ENV_PATH):
        unset_key(ENV_PATH, key)
    os.environ.pop(key, None)
