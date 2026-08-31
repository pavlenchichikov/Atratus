"""The LLM layer of the auto-research agent: prompts, providers, parsing,
retries. auto_research.py only calls the public functions here; this module
never imports auto_research (no cycle) and returns plain dicts.

Providers (GTRADE_AR_LLM): anthropic (default), openai (or any OpenAI-
compatible endpoint via GTRADE_AR_LLM_BASE_URL), ollama (local; added in the
ollama task). SDK imports happen inside the call functions so the module
imports cleanly without them."""

import json
import os

DSL_MENU = (
    "ops: zscore(window 2-200), ratio(a,b), lag(k 1-20), diff(k 1-20), "
    "rolling(window,agg in mean|std|sum), interaction(a,b), lead_lag(leader in "
    "sp500|vix|btc|gold|dxy|tnx, horizon 1-20). Each spec: "
    '{"name": lower_snake, "op": ..., "inputs": [...], "params": {...}}.'
)


def llm_selected():
    """Whether the user picked the LLM proposer (GTRADE_AR_PROPOSER=llm)."""
    return (os.getenv("GTRADE_AR_PROPOSER") or "evolutionary").strip().lower() == "llm"


def _proposer_prompt(log, base_features):
    """The shared features-axis prompt for any LLM provider."""
    history = json.dumps(log[-8:], ensure_ascii=True)
    return (
        "You are proposing engineered features for a trading model to revive weak "
        "neural members. Use ONLY this DSL.\n" + DSL_MENU +
        "\nBase columns you can reference: " + ",".join(base_features) +
        "\nPast experiments (spec + held-back selection Score deltas):\n" + history +
        "\nReturn STRICT JSON: a list of 1-2 new spec dicts, no prose."
    )


AVOID_BUDGET = 3000   # characters of history the prompt may spend
# 3000, not the 1500 this started at: compacting the entries roughly halved
# them (460 characters to 244 on the real registry), and the whole prompt is
# now about 9k against the 26k that used to time the local model out, so the
# budget can afford the breadth. Twelve remembered candidates instead of three.


def _spec_line(sig):
    """One spec signature `[op, inputs, params]` as `+ratio(bb_pos,rsi)`; "" if unreadable."""
    try:
        op, inputs, params = sig
    except Exception:
        return ""
    args = list(inputs) + [f"{k}={v}" for k, v in (params or [])]
    return "+%s(%s)" % (op, ",".join(str(a) for a in args))


def _compact_sig(entry):
    """One already-tried genome as a short line.

    The features axis registers SPEC signatures, not genomes, so an entry is
    either the genome dict or a bare `[op, inputs, params]` list; both arrive
    here from tried_recent.

    The registry stores a canonical JSON signature whose `extra` field is itself
    a list of JSON strings, so the quoting costs more than the content. The model
    does not need to reconstruct a genome from this list, only to recognise what
    is taken, and `drop rsi,vol_z +ratio(bb_pos,rsi) rel_median/30` says that in
    a little under half the characters (460 to 244 measured on the real
    registry). Unparseable entries fall back to a truncated raw string rather
    than being dropped: a hint the model cannot read is still better than
    silently pretending the candidate was never tried.
    """
    try:
        d = json.loads(entry)
    except Exception:
        return str(entry)[:120]
    if not isinstance(d, dict):
        return _spec_line(d) or str(entry)[:120]
    parts = []
    drops = d.get("drops") or []
    if drops:
        parts.append("drop " + ",".join(drops))
    for spec in d.get("extra") or []:
        try:
            line = _spec_line(json.loads(spec))
        except Exception:
            continue
        if line:
            parts.append(line)
    label = d.get("label")
    if label:
        parts.append("%s/%s" % (label[0], label[1]))
    for gene in ("hyper", "nets", "tuning"):
        if d.get(gene):
            parts.append("%s=%s" % (gene, ",".join(str(x) for x in d[gene])))
    return " ".join(parts) or str(entry)[:120]


def _avoid_clause(avoid):
    """A prompt line listing already-tried candidates so the model proposes something
    novel. Empty string when there is nothing to avoid, so the prompt is unchanged.

    Budgeted, and joined instead of json.dumps'd: the entries are already JSON
    strings, so dumping the list re-escapes every quote. Unbudgeted this one
    clause reached 17.8k of a 25.9k prompt - two thirds of it spent restating
    history in double-escaped form. On a local CPU model that is minutes of
    prompt processing per call, and it was the difference between an answer and
    a timeout. Newest first: recent history is what the proposer must avoid.

    Each entry is compacted (see _compact_sig), which is what makes the budget
    worth having: the point of this clause is breadth of history rather than
    detail, so more shorter entries beat fewer verbatim ones.
    """
    if not avoid:
        return ""
    kept, used = [], 0
    for item in reversed(list(avoid)):
        s = _compact_sig(item)
        if used + len(s) > AVOID_BUDGET:
            break
        kept.append(s)
        used += len(s)
    if not kept:
        return ""
    return ("\nAlready tried (do NOT repeat these - propose something genuinely "
            "different):\n" + "\n".join(kept))


def _parse_specs(text):
    """Extract the JSON list of specs from a model reply, tolerant of stray prose."""
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        specs = json.loads(text[start:end + 1])
    except Exception:
        return []
    return specs if isinstance(specs, list) else []


class ProviderUnavailable(RuntimeError):
    """The provider cannot be reached at all: its SDK is not installed.

    Distinct from a call that failed, because the two deserve opposite
    handling. A failed call is worth retrying and is a fair "refused". A
    missing package will fail identically every time, and counting it as a
    refusal reports a model that would not answer when the truth is that
    nothing was ever asked. On 2026-08-31 an analyst run reported refused=1
    for exactly this, after two identical attempts.
    """


def _require(module, provider, pip_name=None):
    """Import a provider SDK or say, once and precisely, how to get it."""
    try:
        return __import__(module)
    except ImportError as exc:
        raise ProviderUnavailable(
            "the %s provider needs the %s package, which is not installed "
            "here: pip install %s" % (provider, module, pip_name or module)
        ) from exc


def _call_anthropic(prompt):
    """Anthropic SDK. Model via GTRADE_AR_LLM_MODEL (default claude-opus-4-8)."""
    anthropic = _require("anthropic", "anthropic")
    client = anthropic.Anthropic()
    model = model_override() or "claude-opus-4-8"
    last_err = None
    for _attempt in range(3):
        try:
            msg = client.messages.create(
                model=model, max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text.strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"anthropic proposer failed after 3 attempts: {last_err}")


def _call_openai(prompt):
    """OpenAI-compatible chat API. Works with OpenAI and any compatible endpoint
    (Mistral, LM Studio, etc.) via GTRADE_AR_LLM_BASE_URL. Model via
    GTRADE_AR_LLM_MODEL (default gpt-4o)."""
    openai = _require("openai", "openai")
    client = openai.OpenAI(base_url=os.getenv("GTRADE_AR_LLM_BASE_URL") or None,
                           timeout=_llm_timeout())
    model = model_override() or "gpt-4o"
    last_err = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"openai proposer failed after 3 attempts: {last_err}")


def _llm_timeout():
    """Client-side timeout (seconds) for a single LLM HTTP call. The OpenAI SDK
    default is 600s (10 min); a slow local reasoning model on CPU can blow far
    past that and get stuck in a retry storm (one wall-clock timeout per attempt,
    compounded by the SDK's own retries). GTRADE_AR_LLM_TIMEOUT overrides in
    seconds; 0/none/unlimited disables the timeout entirely."""
    raw = (os.getenv("GTRADE_AR_LLM_TIMEOUT") or "600").strip().lower()
    if raw in ("0", "none", "unlimited"):
        return None
    try:
        return float(raw)
    except ValueError:
        return 600.0


def _ollama_base_url():
    # 127.0.0.1, not localhost: a Windows system proxy (VPN clients set one) bypasses
    # "<local>" for urllib but NOT for the httpx client inside the openai SDK, which
    # then routes localhost through the proxy and every call dies before reaching
    # Ollama - silently, because the proposer swallows the error and falls back.
    return os.getenv("GTRADE_AR_LLM_BASE_URL") or "http://127.0.0.1:11434/v1"


def list_ollama_models():
    """Every model installed in the local Ollama, newest-first as Ollama returns
    them, via its native tags endpoint (the OpenAI-compatible /v1 API has no model
    listing). Raises RuntimeError if Ollama is unreachable."""
    import urllib.request
    base = _ollama_base_url()
    host = base.removesuffix("/v1")
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach Ollama at {url} (is Ollama running?): {exc}")
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def model_override():
    """GTRADE_AR_LLM_MODEL, or None when the caller wants the provider default.

    "auto" counts as not-set. The launcher must emit a real value rather than a
    blank, because cmd's  set "VAR="  DELETES the variable and load_dotenv then
    refills it from .env - which is how a 17 GB model stayed pinned on a 15.7 GB
    machine through every menu choice (2026-08-14).
    """
    v = (os.getenv("GTRADE_AR_LLM_MODEL") or "").strip()
    return None if v.lower() in ("", "auto") else v


def _detect_ollama_model():
    """The installed model to use when GTRADE_AR_LLM_MODEL is not set: the first
    gemma* model, else the first installed model (any local model works)."""
    names = list_ollama_models()
    if not names:
        raise RuntimeError("no Ollama models installed; run: ollama pull gemma3")
    gemma = [n for n in names if n.lower().startswith("gemma")]
    return gemma[0] if gemma else names[0]


def _print_ollama_models():
    """Print installed Ollama models as a numbered list for the launcher menu.
    Never raises: an unreachable Ollama prints a friendly note instead."""
    try:
        names = list_ollama_models()
    except RuntimeError as exc:
        print(f"  (could not list local models: {exc})")
        return
    if not names:
        print("  (no Ollama models installed; run: ollama pull gemma3)")
        return
    for i, name in enumerate(names, 1):
        print("  [%d] %s" % (i, name))


def _call_ollama(prompt):
    """Local Ollama via its OpenAI-compatible API. Base URL via
    GTRADE_AR_LLM_BASE_URL (default localhost:11434/v1); model via
    GTRADE_AR_LLM_MODEL or auto-detected (gemma preferred)."""
    openai = _require("openai", "ollama")
    base = _ollama_base_url()
    model = model_override() or _detect_ollama_model()
    # Reasoning models (e.g. gemma) spend tokens on an internal reasoning trace before
    # the answer; a small cap gets fully consumed by reasoning and returns EMPTY content
    # (the silent cause of a wiki/proposer that "runs" but produces nothing). Budget
    # generously. GTRADE_AR_LLM_MAX_TOKENS overrides; set it to 0 for NO cap (local model
    # is free, but the only cost is wall-clock time - an uncapped reasoning trace can run
    # long, fine for the one-shot wiki, risky for the many-call proposer path).
    raw = (os.getenv("GTRADE_AR_LLM_MAX_TOKENS") or "8000").strip().lower()
    if raw in ("0", "none", "unlimited"):
        max_toks = None
    else:
        try:
            max_toks = int(raw)
        except ValueError:
            max_toks = 8000
    # max_retries=0: this function already loops 3x below, and the SDK's own
    # retries each wait a full timeout -> without this a slow local model turns
    # one stuck call into a multi-hour retry storm (the 10-min-apart retries).
    # trust_env=False: httpx picks up the Windows registry proxy (a VPN client sets
    # one) and ignores its bypass list, so every local call is routed through the
    # proxy - it works while the VPN is up and dies with a bare "Connection error."
    # the moment it is not, silently demoting the run to the evolutionary proposer.
    # Ollama is on loopback; it never needs a proxy.
    import httpx
    client = openai.OpenAI(base_url=base, api_key="ollama",
                           timeout=_llm_timeout(), max_retries=0,
                           http_client=httpx.Client(trust_env=False,
                                                    timeout=_llm_timeout()))
    last_err = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_toks,
                messages=[{"role": "user", "content": prompt}])
            return (resp.choices[0].message.content or "").strip()
        except openai.APITimeoutError as exc:
            # A timeout is not transient: the same prompt, model and machine will
            # be just as slow next time, so a retry only multiplies the wait.
            # Three attempts at the 600s default cost half an hour to learn what
            # the first one already said.
            raise RuntimeError(
                f"ollama call timed out after {_llm_timeout()}s (model too slow for this prompt; "
                "try a smaller model or raise GTRADE_AR_LLM_TIMEOUT)") from exc
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        f"ollama proposer failed after 3 attempts (is Ollama running at {base}?): {last_err}")


def _traced(fn, what, provider):
    """fn with one console line before the call and one after. A local model can
    sit on a single call for tens of minutes and an empty reply is indistinguishable
    from a refusal, so a silent LLM arm is unreadable from the run output."""
    def call(prompt):
        import time
        model = model_override() or "auto"
        print("[llm] %s: asking %s/%s, %d char prompt" % (what, provider, model,
                                                          len(prompt)), flush=True)
        t0 = time.time()
        try:
            out = fn(prompt)
        except Exception as exc:
            print(f"[llm] {what}: FAILED after {time.time() - t0:.0f}s: {exc}",
                  flush=True)
            raise
        print("[llm] %s: %d char reply in %.0fs" % (what, len(out or ""),
                                                    time.time() - t0), flush=True)
        return out
    return call


def _backend(what="llm"):
    """The provider call function for GTRADE_AR_LLM, resolved at call time so
    tests can monkeypatch the _call_* functions. `what` labels the console trace."""
    provider = (os.getenv("GTRADE_AR_LLM") or "anthropic").strip().lower()
    backends = {"anthropic": _call_anthropic, "openai": _call_openai,
                "ollama": _call_ollama}
    fn = backends.get(provider)
    if fn is None:
        raise RuntimeError(
            f"unknown GTRADE_AR_LLM {provider!r} (use anthropic, openai or ollama)")
    return _traced(fn, what, provider)


def reflect_on():
    """GTRADE_AR_REFLECT: run a 'reflect then propose' step on the LLM path (default OFF)."""
    return (os.getenv("GTRADE_AR_REFLECT") or "").strip() in ("1", "true", "True")


def _wiki_preamble():
    """The compounding research wiki as a prompt preamble when GTRADE_AR_WIKI is on;
    '' otherwise (so the prompt is unchanged). Any error yields ''."""
    try:
        from core import ar_wiki
        if not ar_wiki.wiki_on():
            return ""
        text = ar_wiki.wiki_summary()
        if not text:
            return ""
        return "Accumulated research wiki (distilled prior findings):\n" + text + "\n"
    except Exception:
        return ""


def _reflect_hypothesis():
    """One-line hypothesis of why recent experiments did not clear the gate, from the
    findings journal. Empty string when reflection is off, the journal is empty, or any
    error - so the caller's prompt is unchanged in those cases."""
    if not reflect_on():
        return ""
    try:
        from core import ar_memory
        recent = ar_memory.findings_recent(5)
        if not recent:
            return ""
        prompt = (
            "Here are recent auto-research experiments and whether they cleared the "
            "held-out gate:\n" + json.dumps(recent, ensure_ascii=True)[:4000] +
            "\nIn ONE sentence, hypothesize why they did not improve the model. No prose.")
        return (_backend("reflect")(prompt) or "").strip()
    except Exception:
        return ""


GENOME_MENU = (
    'A genome is JSON: {"drops": [features to drop], "extra": [spec dicts], '
    '"label_mode": "direction" or "rel_median", "label_window": 20 or 30 or 60}. '
    "extra specs use this DSL: " + DSL_MENU
)


def _parse_obj(text):
    """Extract ONE JSON object from a model reply, tolerant of stray prose."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def propose_genome(parent, elites, active, base_features, avoid=None):
    """Ask the LLM for ONE modified experiment genome (the QD agent's
    LLM-guided mutation). Returns a plain dict or None on any parse/shape
    problem; the caller validates and falls back to evolutionary operators.
    Retries once (the QD loop has a cheap fallback, unlike the features axis).
    avoid: already-tried genome signatures the model must not repeat (default None,
    so the prompt is unchanged)."""
    prompt = (
        "You are evolving experiment genomes for a trading-model search "
        "(MAP-Elites). Propose ONE child genome likely to beat the elites.\n"
        + GENOME_MENU +
        "\nDroppable features: " + ",".join(active) +
        "\nBase columns for specs: " + ",".join(base_features) +
        "\nParent genome: " + json.dumps(parent, ensure_ascii=True) +
        "\nCurrent elites (genome + fitness): " + json.dumps(elites, ensure_ascii=True) +
        _avoid_clause(avoid) +
        "\nReturn STRICT JSON: one genome object, no prose."
    )
    prompt = _wiki_preamble() + prompt
    hyp = _reflect_hypothesis()
    if hyp:
        prompt = "Reflection: " + hyp + "\n" + prompt
    backend = _backend("genome")
    for _attempt in range(2):
        obj = _parse_obj(backend(prompt))
        if obj is not None:
            print(f"[llm] genome: {json.dumps(obj, ensure_ascii=True)[:300]}",
                  flush=True)
            return obj
        print("[llm] genome: no JSON object in the reply, retrying", flush=True)
    return None


def propose_specs(log, base_features, avoid=None):
    """Ask the selected LLM for the next 1-2 feature specs. The backend retries
    a few times then raises cleanly; a non-JSON reply yields no specs (that
    iteration is skipped). avoid: already-tried spec signatures the model must not
    repeat (default None, so the prompt is unchanged)."""
    prompt = _proposer_prompt(log, base_features) + _avoid_clause(avoid)
    prompt = _wiki_preamble() + prompt
    hyp = _reflect_hypothesis()
    if hyp:
        prompt = "Reflection: " + hyp + "\n" + prompt
    return _parse_specs(_backend("specs")(prompt))


if __name__ == "__main__":
    import sys
    if "--list-ollama" in sys.argv:
        _print_ollama_models()
