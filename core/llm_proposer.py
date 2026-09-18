"""The LLM layer of the auto-research agent: prompts, providers, parsing,
retries. auto_research.py only calls the public functions here; this module
never imports auto_research (no cycle) and returns plain dicts.

Providers (GTRADE_AR_LLM): anthropic (default), openai (or any OpenAI-
compatible endpoint via GTRADE_AR_LLM_BASE_URL), ollama (local; added in the
ollama task). SDK imports happen inside the call functions so the module
imports cleanly without them."""

import json
import os
import re

# "no argument given", so a caller that says nothing keeps the old behaviour and
# None can still mean "no cap at all".
_UNSET = object()

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


# Which parameter a bare number in "lag(vol_z, 3)" belongs to, per op. The DSL
# itself is the authority on the names; this is only the reverse mapping for a
# call the model wrote in function form.
_OP_NUMERIC_PARAM = {"lag": "k", "diff": "k", "zscore": "window",
                     "rolling": "window", "lead_lag": "horizon"}


def _normalise_spec(spec):
    """Rewrite `{"op": "ratio(bb_pos, rsi)"}` as `{"op": "ratio", "inputs": [...]}`.

    Local models write the transform the way a person would, as a call. The DSL
    validator takes an op NAME plus `inputs`, so every such spec was illegal,
    and an illegal spec makes the whole genome illegal - which the QD loop then
    replaces with an evolutionary child. On the 2026-09-18 campaign that threw
    away most of the LLM proposals after 10 to 30 minutes of generation each,
    silently: the run looked like it was using the model and was not.

    Anything not in function form, and any op not in the DSL, is returned
    untouched: this normalises a shape, it never invents an op.
    """
    if not isinstance(spec, dict) or not isinstance(spec.get("op"), str):
        return spec
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", spec["op"])
    if not m:
        return spec
    op, args = m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]
    if op not in _OP_NUMERIC_PARAM and op not in ("ratio", "interaction"):
        return spec
    out = dict(spec)
    out["op"] = op
    names = [a for a in args if not a.lstrip("+-").isdigit()]
    numbers = [int(a) for a in args if a.lstrip("+-").isdigit()]
    if names and not out.get("inputs"):
        out["inputs"] = names
    key = _OP_NUMERIC_PARAM.get(op)
    if key and numbers:
        params = dict(out.get("params") or {})
        params.setdefault(key, numbers[0])
        out["params"] = params
    return out


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
    if not isinstance(specs, list):
        return []
    return [_normalise_spec(sp) for sp in specs]


class TerminalCallError(RuntimeError):
    """A call that will fail the same way if it is simply repeated."""


class AnswerLostToTrace(TerminalCallError):
    """The model spent its whole token budget on reasoning and never answered.

    Structural, like a timeout: the same prompt, model and cap spend the same
    budget the same way. Named separately because the cure is different - raise
    the cap, or ask without the trace.
    """


class CallTimedOut(TerminalCallError):
    """The call ran past GTRADE_AR_LLM_TIMEOUT.

    Its own layer already refuses to retry a timeout - the same prompt, model
    and machine will be just as slow next time - but that only covered the
    retries inside one call. The analyst's judge() loop caught it as an
    ordinary failure and asked again, so 2026-09-18 spent three and a half
    hours on four identical hour-long timeouts of gemma4:26b. A caller must be
    able to tell this apart from a model that answered badly.
    """


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


def _require_key(var, provider):
    """The provider's credential, or the same clear stop a missing SDK gives.

    Checked here rather than left to the SDK for two reasons. The SDK raises at
    CLIENT CONSTRUCTION, inside the retry loop, so one missing key produced
    three identical attempts and then two more from the caller, six failures
    for one absent string. And its message ("Could not resolve authentication
    method. Expected one of api_key, auth_token, or credentials...") names
    neither the variable nor the file, which is what a person actually needs.

    config.py loads .env at import, so a key written there is already in the
    environment by the time any of this runs.
    """
    if (os.getenv(var) or "").strip():
        return
    raise ProviderUnavailable(
        "the %s provider needs %s, which is not set. Put it in .env next to "
        "the other secrets (see .env.example), or use provider ollama, which "
        "needs no key at all." % (provider, var))


# 4xx that will read exactly the same on the next attempt: a bad request, a
# rejected key, a model this account cannot use, an empty wallet. 429 and 408
# are deliberately absent - those DO deserve a retry.
_TERMINAL_STATUS = (400, 401, 403, 404, 405, 413, 422)


def _api_detail(exc):
    """The provider's own words for a failed call, or None.

    Worth extracting rather than str(exc) because the SDKs stringify a status
    error as "Connection error." while the body says "Your credit balance is
    too low to access the Anthropic API". On 2026-08-31 that cost eighteen
    calls and an hour, and the answer was in the first response.
    """
    for attr in ("response", "body"):
        obj = getattr(exc, attr, None)
        if obj is None:
            continue
        try:
            data = obj.json() if hasattr(obj, "json") else obj
            if isinstance(data, dict):
                err = data.get("error") or data
                msg = err.get("message") if isinstance(err, dict) else None
                if msg:
                    return str(msg)
        except Exception:
            text = getattr(obj, "text", None)
            if text:
                return str(text)[:300]
    return None


def _raise_if_terminal(exc, provider):
    """Stop on an error that repeating cannot fix, with the API's own message.

    Raised as ProviderUnavailable for the same reason a missing package and a
    missing key are: none of the three is the model declining to answer, and
    counting them as refusals is what made an empty balance look like a
    judgment the analyst chose not to make.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status not in _TERMINAL_STATUS:
        return
    raise ProviderUnavailable(
        "%s refused the request with HTTP %s: %s"
        % (provider, status, _api_detail(exc) or exc)) from exc


def _call_anthropic(prompt, temperature=None, max_tokens=_UNSET):
    """Anthropic SDK. Model via GTRADE_AR_LLM_MODEL (default claude-opus-4-8)."""
    _require_key("ANTHROPIC_API_KEY", "anthropic")
    anthropic = _require("anthropic", "anthropic")
    client = anthropic.Anthropic()
    model = model_override() or "claude-opus-4-8"
    last_err = None
    for _attempt in range(3):
        try:
            msg = client.messages.create(
                model=model, max_tokens=(600 if max_tokens is _UNSET else max_tokens),
                **_temp_kw(temperature),
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text.strip()
        except Exception as exc:
            _raise_if_terminal(exc, "anthropic")
            last_err = exc
    raise RuntimeError("anthropic proposer failed after 3 attempts: %s"
                       % (_api_detail(last_err) or last_err))


def _call_openai(prompt, temperature=None, max_tokens=_UNSET):
    """OpenAI-compatible chat API. Works with OpenAI and any compatible endpoint
    (Mistral, LM Studio, etc.) via GTRADE_AR_LLM_BASE_URL. Model via
    GTRADE_AR_LLM_MODEL (default gpt-4o)."""
    # Only when talking to OpenAI itself. GTRADE_AR_LLM_BASE_URL points this
    # same backend at LM Studio and friends, which the docstring above promises
    # and which need no key, so the check must not fire there.
    if not (os.getenv("GTRADE_AR_LLM_BASE_URL") or "").strip():
        _require_key("OPENAI_API_KEY", "openai")
    openai = _require("openai", "openai")
    client = openai.OpenAI(base_url=os.getenv("GTRADE_AR_LLM_BASE_URL") or None,
                           timeout=_llm_timeout())
    model = model_override() or "gpt-4o"
    last_err = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=(600 if max_tokens is _UNSET else max_tokens),
                **_temp_kw(temperature),
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            _raise_if_terminal(exc, "openai")
            last_err = exc
    raise RuntimeError("openai proposer failed after 3 attempts: %s"
                       % (_api_detail(last_err) or last_err))


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


def _ollama_unload(base, model, post=None):
    """Ask Ollama to drop the model from VRAM now instead of in five minutes.

    Ollama keeps a model resident for OLLAMA_KEEP_ALIVE (5 minutes by default)
    after a reply, and auto_research starts training seconds later. On a 4 GB
    card gemma4:12b leaves about 1.5 GiB free, gpu_fit_jobs correctly refuses
    the second training process, and the run halves its own throughput while
    the card looks busy - measured on the 2026-09-18 campaign, where every step
    that followed an LLM call printed "fits 1 training process(es), not 2" and
    every step without one ran two.

    Best effort: a failure here costs the old behaviour, never the run. The
    reload on the next call is seconds against calls that take minutes.
    """
    root = base.removesuffix("/v1")
    try:
        if post is None:
            import httpx

            post = httpx.Client(trust_env=False, timeout=10).post
        post(root.rstrip("/") + "/api/generate",
             json={"model": model, "keep_alive": 0})
    except Exception:
        pass


def _ollama_native_chat(base, model, prompt, temperature, max_tokens, think):
    """One call over Ollama's OWN /api/chat, which is the only place `think`
    is honoured.

    The OpenAI-compatible layer accepts the field and ignores it: measured
    2026-09-18 on gemma4:26b, `think: false` through /v1 still produced 7992
    tokens of reasoning and returned an empty message - the whole 8000 cap
    spent before a word of the answer, after 3577 seconds. The same model
    answers over /api/chat with the trace actually off.

    Only the analyst takes this path. The search never passes `think`, so its
    calls keep going through the SDK exactly as before.
    """
    import httpx

    options = {}
    if temperature is not None:
        options["temperature"] = float(temperature)
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": prompt}]}
    if think is not None:
        payload["think"] = bool(think)
    if options:
        payload["options"] = options
    url = base.removesuffix("/v1").rstrip("/") + "/api/chat"
    client = httpx.Client(trust_env=False, timeout=_llm_timeout())
    try:
        resp = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise CallTimedOut(
            f"ollama call timed out after {_llm_timeout()}s (model too slow for "
            "this prompt; try a smaller model or raise GTRADE_AR_LLM_TIMEOUT)") from exc
    resp.raise_for_status()
    msg = (resp.json().get("message") or {})
    # Separate fields, which is the point of this endpoint: an empty answer
    # beside a long trace is a budget problem, an empty answer beside no trace
    # is a model that said nothing.
    return (msg.get("content") or "").strip(), (msg.get("thinking") or "").strip()


def _call_ollama(prompt, temperature=None, max_tokens=_UNSET, think=None):
    """Local Ollama over its OWN /api/chat. Base URL via GTRADE_AR_LLM_BASE_URL
    (default localhost:11434, /v1 suffix tolerated); model via
    GTRADE_AR_LLM_MODEL or auto-detected (gemma preferred).

    Every path goes here now, not only the analyst. The OpenAI-compatible layer
    was never wrong, it was just narrower: it accepts `think` and ignores it
    (measured 2026-09-18 on gemma4:26b - 7992 tokens of reasoning, an empty
    message, 3577 seconds), and it reports the reasoning and the answer in one
    field, so a trace that runs long is indistinguishable from a model with
    nothing to say. The native endpoint returns them separately, which is what
    lets this function say WHICH of the two happened.
    """
    base = _ollama_base_url()
    model = model_override() or _detect_ollama_model()
    # Reasoning models spend tokens on the trace BEFORE the answer, so a cap the
    # trace uses up leaves nothing for the answer. GTRADE_AR_LLM_MAX_TOKENS
    # overrides; 0 means no cap, which is fine for a one-shot call and risky for
    # the many-call search path.
    if max_tokens is not _UNSET:
        max_toks = max_tokens
    else:
        raw = (os.getenv("GTRADE_AR_LLM_MAX_TOKENS") or "8000").strip().lower()
        if raw in ("0", "none", "unlimited"):
            max_toks = None
        else:
            try:
                max_toks = int(raw)
            except ValueError:
                max_toks = 8000
    last_err = None
    for _attempt in range(3):
        try:
            out, trace = _ollama_native_chat(base, model, prompt, temperature,
                                             max_toks, think)
        except TerminalCallError:
            # A timeout, or an answer the trace ate: the same prompt, model and
            # cap produce the same outcome next time, so asking again only
            # multiplies the wait. 2026-09-18 spent three and a half hours
            # learning that four times over.
            raise
        except Exception as exc:
            last_err = exc
            continue
        _ollama_unload(base, model)
        if not out and trace:
            raise AnswerLostToTrace(
                "the model spent its whole budget (%s tokens) on reasoning and "
                "returned no answer. Raise the cap, or ask without the trace "
                "(GTRADE_ANALYST_THINK=0 on the analyst path)."
                % ("no" if max_toks is None else max_toks))
        return out
    raise RuntimeError(
        f"ollama proposer failed after 3 attempts (is Ollama running at {base}?): {last_err}")


# A judgment that cannot be reproduced cannot be audited. Measured 2026-09-18
# against the local model: at the provider's own default the same question came
# back 412, 402, 402, 412, and at temperature 0 it came back 412 three times.
# The analyst therefore asks at 0, so a changed verdict means changed EVIDENCE.
# The genome proposer keeps the provider default on purpose: there the spread is
# the point, and a search that proposes one genome forever is worse than a noisy
# one (the 2026-09-18 campaign already lost steps to "no unseen child").
ANALYST_TEMPERATURE = 0.0


def analyst_temperature():
    """GTRADE_ANALYST_TEMPERATURE: 0 by default, or empty/none to send nothing
    and let the provider decide.

    The knob exists because greedy decoding is not free of risk on a local
    model: it can fall into a repetition loop and never emit a stop token,
    which on gemma4:26b looks exactly like the runaway this file already
    describes. Leave it at 0 for an auditable judgment; clear it if a
    particular model starts writing forever.
    """
    raw = (os.getenv("GTRADE_ANALYST_TEMPERATURE") or str(ANALYST_TEMPERATURE)).strip().lower()
    if raw in ("", "none", "default"):
        return None
    try:
        return float(raw)
    except ValueError:
        return ANALYST_TEMPERATURE

# Ollama only: ask the model to skip its reasoning trace. Raising the token cap
# was not enough on its own - measured 2026-09-18 on the same 7.9k-char analyst
# prompt and the same machine:
#
#   cap 8000, trace on    0 chars after 2031s   (the trace spent the whole cap)
#   no cap,   trace on    timed out at 3600s
#   no cap,   trace off   1944 chars after 983s, a valid judgment
#
# The judgment itself is a small JSON object, so the wait buys the trace, not
# the answer. The genome proposer is left alone: it writes a new hypothesis
# rather than filling in a form, and that is where a trace may earn its minutes.
def analyst_think():
    """GTRADE_ANALYST_THINK: 1 by default - the analyst reasons before it judges.

    It was briefly off, because through the OpenAI-compatible layer the trace
    and the answer shared one field and one budget, so a long trace returned an
    empty judgment after the full wait. On the native endpoint they are separate
    and a trace that ran long is reported as such, so the reason for turning it
    off is gone. Set 0 if a model turns out to spend its whole budget thinking.
    """
    raw = (os.getenv("GTRADE_ANALYST_THINK") or "1").strip().lower()
    return raw not in ("0", "false", "off", "no")

# A reasoning model spends tokens on its trace BEFORE it answers, so a cap that
# the trace uses up returns empty content - not an error, not a refusal, just
# nothing. Measured 2026-09-18: the 7871-char analyst prompt came back 0 chars
# after 2031 seconds at the shared 8000-token cap, and each retry costs another
# half hour. The analyst therefore asks with no cap by default: it is ONE call
# per asset, unlike the proposer, where an uncapped trace would be paid for on
# every step of a hundred-step search.
def analyst_max_tokens():
    """GTRADE_ANALYST_MAX_TOKENS: the cap on ONE judgment, or 0/none/unlimited
    for no cap.

    Default 8000. It was briefly no-cap, which was the wrong lesson from the
    empty replies: the trace was what spent the budget, and with the trace off
    a judgment is a small JSON object that fits several times over. Uncapped,
    a model that ignores the no-trace flag - gemma4:26b does - generates until
    the hour-long timeout instead of until the cap, which is how a run that
    used to answer stopped answering.
    """
    raw = (os.getenv("GTRADE_ANALYST_MAX_TOKENS") or "8000").strip().lower()
    if raw in ("0", "none", "unlimited"):
        return None
    try:
        return int(raw)
    except ValueError:
        return 8000


def _temp_kw(temperature):
    """{"temperature": t} or {}, so a caller that says nothing keeps the
    provider's default and the request stays byte-identical to before."""
    return {} if temperature is None else {"temperature": float(temperature)}


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
    if what == "analyst":
        base_fn = fn
        kw = {"temperature": analyst_temperature(),
              "max_tokens": analyst_max_tokens()}
        if provider == "ollama":
            kw["think"] = analyst_think()

        def fn(prompt):
            return base_fn(prompt, **kw)
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
        if isinstance(obj, dict) and isinstance(obj.get("extra"), list):
            obj["extra"] = [_normalise_spec(sp) for sp in obj["extra"]]
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
