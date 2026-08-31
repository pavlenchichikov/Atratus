import json
import sys
import types

import pytest

from core import llm_proposer as lp


def test_parse_specs_tolerates_prose():
    text = ('Sure! [{"name": "f", "op": "lag", "inputs": ["ret_1"],'
            ' "params": {"k": 1}}] hope this helps')
    specs = lp._parse_specs(text)
    assert specs and specs[0]["name"] == "f"
    assert lp._parse_specs("no json here") == []
    assert lp._parse_specs("") == []
    assert lp._parse_specs('{"a": 1}') == []      # dict, not a list


def test_backend_selection_unknown_raises(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "bogus")
    with pytest.raises(RuntimeError):
        lp.propose_specs([], ["ret_1"])


def test_propose_specs_uses_selected_backend(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: (
        '[{"name": "f", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]'))
    specs = lp.propose_specs([], ["ret_1"])
    assert specs == [{"name": "f", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]


def test_propose_specs_prompt_carries_history(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setattr(lp, "_call_openai",
                        lambda prompt: seen.setdefault("prompt", prompt) and "[]" or "[]")
    lp.propose_specs([{"iter": 0, "score": 0.5}], ["ret_1", "rsi"])
    assert "ret_1,rsi" in seen["prompt"]
    assert '"score": 0.5' in seen["prompt"]


def test_llm_selected(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_PROPOSER", raising=False)
    assert lp.llm_selected() is False
    monkeypatch.setenv("GTRADE_AR_PROPOSER", "llm")
    assert lp.llm_selected() is True


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_detect_ollama_prefers_gemma(monkeypatch):
    import urllib.request
    monkeypatch.delenv("GTRADE_AR_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=5: _FakeResp(
        {"models": [{"name": "llama3:8b"}, {"name": "gemma4:26b"}]}))
    assert lp._detect_ollama_model() == "gemma4:26b"


def test_detect_ollama_first_model_fallback(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=5: _FakeResp(
        {"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]}))
    assert lp._detect_ollama_model() == "llama3:8b"


def test_detect_ollama_no_models_raises(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=5: _FakeResp({"models": []}))
    with pytest.raises(RuntimeError, match="ollama pull"):
        lp._detect_ollama_model()


def test_detect_ollama_unreachable_raises(monkeypatch):
    import urllib.request

    def dead(url, timeout=5):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", dead)
    with pytest.raises(RuntimeError, match="running"):
        lp._detect_ollama_model()


def test_call_ollama_defaults(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, model, max_tokens, messages):
            captured["model"] = model
            msg = types.SimpleNamespace(content=" [] ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, base_url=None, api_key=None, **kwargs):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured.update(kwargs)
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    monkeypatch.delenv("GTRADE_AR_LLM_MODEL", raising=False)
    monkeypatch.delenv("GTRADE_AR_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(lp, "_detect_ollama_model", lambda: "gemma4:26b")
    assert lp._call_ollama("hi") == "[]"
    # 127.0.0.1, never "localhost": a Windows system proxy swallows localhost
    # inside the openai SDK's httpx client (see _ollama_base_url).
    assert captured["base_url"] == "http://127.0.0.1:11434/v1"
    assert captured["api_key"] == "ollama"
    assert captured["model"] == "gemma4:26b"
    # ...and the SDK must not inherit the system proxy: httpx honours the Windows
    # registry proxy but not its loopback bypass list, which kills every call the
    # moment the VPN proxy is down.
    assert captured["http_client"].trust_env is False


def test_call_ollama_model_env_override(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, model, max_tokens, messages):
            captured["model"] = model
            msg = types.SimpleNamespace(content="[]")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, base_url=None, api_key=None, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "gemma3:latest")
    lp._call_ollama("hi")
    assert captured["model"] == "gemma3:latest"


def test_backend_traces_every_call(monkeypatch, capsys):
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "gemma4:26b")
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: "reply")
    assert lp._backend("genome")("a prompt") == "reply"
    out = capsys.readouterr().out
    assert "[llm] genome: asking openai/gemma4:26b, 8 char prompt" in out
    assert "5 char reply in" in out

    def boom(prompt):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(lp, "_call_openai", boom)
    with pytest.raises(RuntimeError):
        lp._backend("wiki")("p")
    assert "[llm] wiki: FAILED after" in capsys.readouterr().out


def test_backend_knows_ollama(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "ollama")
    monkeypatch.setattr(lp, "_call_ollama", lambda prompt: "[]")
    assert lp.propose_specs([], ["ret_1"]) == []


def test_propose_genome_good_json(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    reply = ('Here you go: {"drops": ["rsi"], "extra": [], '
             '"label_mode": "rel_median", "label_window": 20} enjoy')
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: reply)
    obj = lp.propose_genome({"drops": [], "extra": [], "label_mode": "direction",
                             "label_window": 30}, [], ["rsi", "atr"], ["ret_1"])
    assert obj == {"drops": ["rsi"], "extra": [],
                   "label_mode": "rel_median", "label_window": 20}


def test_propose_genome_garbage_returns_none(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: "I cannot help with that.")
    assert lp.propose_genome({}, [], ["rsi"], ["ret_1"]) is None


def test_propose_genome_prompt_mentions_parent_and_elites(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")

    def capture(prompt):
        seen["prompt"] = prompt
        return "{}"

    monkeypatch.setattr(lp, "_call_openai", capture)
    lp.propose_genome({"drops": ["atr"]}, [{"genome": {"drops": []}, "fitness": 1.5}],
                      ["rsi", "atr"], ["ret_1"])
    assert '"drops": ["atr"]' in seen["prompt"]
    assert '"fitness": 1.5' in seen["prompt"]
    assert "rsi,atr" in seen["prompt"]


def test_reflect_on_env(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_REFLECT", raising=False)
    assert lp.reflect_on() is False
    monkeypatch.setenv("GTRADE_AR_REFLECT", "1")
    assert lp.reflect_on() is True


def test_reflect_hypothesis_reads_journal(monkeypatch):
    from core import ar_memory
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setenv("GTRADE_AR_REFLECT", "1")
    ar_memory.findings_append({"ts": "t1", "winners": [{"axis": "features", "adoptable": False}]})
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: "the features were too noisy")
    assert lp._reflect_hypothesis() == "the features were too noisy"


def test_reflect_hypothesis_empty_when_off_or_error(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_REFLECT", raising=False)
    assert lp._reflect_hypothesis() == ""              # off
    monkeypatch.setenv("GTRADE_AR_REFLECT", "1")
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    def boom(prompt):
        raise RuntimeError("down")
    monkeypatch.setattr(lp, "_call_openai", boom)
    assert lp._reflect_hypothesis() == ""              # error - ""


def test_propose_specs_includes_reflection(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setenv("GTRADE_AR_REFLECT", "1")
    monkeypatch.setattr(lp, "_reflect_hypothesis", lambda: "too much overfitting")
    def capture(prompt):
        seen["prompt"] = prompt
        return "[]"
    monkeypatch.setattr(lp, "_call_openai", capture)
    lp.propose_specs([], ["ret_1"])
    assert "Reflection: too much overfitting" in seen["prompt"]


def test_propose_specs_no_reflection_when_off(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.delenv("GTRADE_AR_REFLECT", raising=False)
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: seen.setdefault("prompt", prompt) or "[]")
    lp.propose_specs([], ["ret_1"])
    assert "Reflection:" not in seen["prompt"]         # byte-identical prompt when off


def test_propose_genome_includes_reflection(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setenv("GTRADE_AR_REFLECT", "1")
    monkeypatch.setattr(lp, "_reflect_hypothesis", lambda: "labels too noisy")

    def capture(prompt):
        seen["prompt"] = prompt
        return "{}"

    monkeypatch.setattr(lp, "_call_openai", capture)
    lp.propose_genome({"drops": []}, [], ["rsi", "atr"], ["ret_1"])
    assert "Reflection: labels too noisy" in seen["prompt"]


def test_list_ollama_models_returns_all(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=5: _FakeResp(
        {"models": [{"name": "gemma3:27b"}, {"name": "gemma2:2b"}, {"name": ""}]}))
    assert lp.list_ollama_models() == ["gemma3:27b", "gemma2:2b"]   # blanks dropped


def test_avoid_clause_off_by_default():
    assert lp._avoid_clause(None) == ""
    assert lp._avoid_clause([]) == ""


def test_propose_specs_avoid_appended_only_when_given(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.delenv("GTRADE_AR_REFLECT", raising=False)

    def capture(prompt):
        seen["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(lp, "_call_openai", capture)
    lp.propose_specs([], ["ret_1"])
    without = seen["prompt"]
    assert "Already tried" not in without                 # None - unchanged prompt

    lp.propose_specs([], ["ret_1"], avoid=["sig-A", "sig-B"])
    assert "Already tried" in seen["prompt"] and "sig-A" in seen["prompt"]
    assert seen["prompt"].startswith(without.split("\nAlready tried")[0][:20])


def test_propose_genome_avoid_appended(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.delenv("GTRADE_AR_REFLECT", raising=False)
    monkeypatch.setattr(lp, "_call_openai", lambda p: seen.setdefault("prompt", p) or "{}")
    lp.propose_genome({"drops": []}, [], ["rsi"], ["ret_1"], avoid=["gsig-1"])
    assert "Already tried" in seen["prompt"] and "gsig-1" in seen["prompt"]


def test_wiki_prepended_when_on(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setenv("GTRADE_AR_WIKI", "1")
    monkeypatch.delenv("GTRADE_AR_REFLECT", raising=False)
    import core.ar_wiki as w
    monkeypatch.setattr(w, "wiki_summary", lambda max_chars=6000: "## features\nmacro drops hurt")
    monkeypatch.setattr(lp, "_call_openai", lambda p: seen.setdefault("prompt", p) or "[]")
    lp.propose_specs([], ["ret_1"])
    assert "research wiki" in seen["prompt"].lower() and "macro drops hurt" in seen["prompt"]


def test_wiki_off_prompt_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.delenv("GTRADE_AR_WIKI", raising=False)
    monkeypatch.delenv("GTRADE_AR_REFLECT", raising=False)
    monkeypatch.setattr(lp, "_call_openai", lambda p: seen.setdefault("prompt", p) or "[]")
    lp.propose_specs([], ["ret_1"])
    assert "research wiki" not in seen["prompt"].lower()          # off - unchanged


def _sig(i):
    """A tried-registry entry in the real stored shape: a canonical JSON
    signature whose `extra` is itself a list of JSON strings."""
    return json.dumps({
        # genome_sig stores drops sorted, so the fixture does too
        "drops": sorted(["corr_sp500", "vol_z", "trend_strength", f"f{i}"]),
        "extra": [json.dumps(["interaction", ["ret_20", "ret_5"], []]),
                  json.dumps(["lag", ["vol_z"], [["k", 3]]]),
                  json.dumps(["lead_lag", ["vix"], [["horizon", 5]]]),
                  json.dumps(["zscore", ["vol_z"], [["window", 20]]])],
        "label": ["rel_median", 30],
        "tuning": [0.05, 0.0, "taleb_only"],
    }, sort_keys=True)


def test_compact_sig_keeps_the_identity_and_drops_the_syntax():
    out = lp._compact_sig(_sig(7))
    assert "drop corr_sp500,f7,trend_strength,vol_z" in out
    assert "+lag(vol_z,k=3)" in out
    assert "+lead_lag(vix,horizon=5)" in out
    assert "rel_median/30" in out
    assert "tuning=0.05,0.0,taleb_only" in out
    # measured 460 to 244 characters on the real registry; the quoting is
    # what goes, so do not claim more than the ~2x it actually buys
    assert len(out) < len(_sig(7)) * 0.6
    assert "\\" not in out                       # no re-escaped JSON


def test_compact_sig_takes_a_spec_signature_too():
    """The features axis registers spec signatures, not genomes, so the entry is a
    bare list. It used to reach `.get` and kill the whole search run."""
    out = lp._compact_sig(json.dumps(["lag", ["vol_z"], [["k", 3]]]))
    assert out == "+lag(vol_z,k=3)"
    assert lp._compact_sig(json.dumps(["ratio", ["bb_pos", "rsi"], []])) == "+ratio(bb_pos,rsi)"
    assert "+lag(vol_z,k=3)" in lp._avoid_clause(
        [json.dumps(["lag", ["vol_z"], [["k", 3]]])])
    # a list of the wrong arity is still a hint, not a crash
    assert lp._compact_sig(json.dumps(["lag"])).startswith('["lag"')


def test_compact_sig_never_loses_an_unreadable_entry():
    # A hint the model cannot parse still beats pretending it was never tried.
    assert lp._compact_sig("not json at all").startswith("not json")
    assert lp._compact_sig("") == ""


def test_avoid_clause_is_budgeted_and_compacted():
    """This clause was 17.8k of a 25.9k prompt: 30 genome dumps re-escaped by
    json.dumps. On a local CPU model that alone was minutes per call."""
    entries = [_sig(i) for i in range(30)]
    raw = lp._avoid_clause.__globals__["json"].dumps(entries)
    out = lp._avoid_clause(entries)
    assert len(out) <= lp.AVOID_BUDGET + 200
    assert len(out) < len(raw) / 3           # budgeted AND compacted
    assert "\\" not in out                    # not JSON-encoded a second time
    assert "f29" in out                      # newest kept
    assert "f0" not in out                   # oldest dropped
    # compaction has to buy real breadth, not one extra line
    assert out.count("drop ") >= 8
    assert lp._avoid_clause([]) == ""


def test_a_timed_out_ollama_call_is_not_retried(monkeypatch):
    """A timeout is deterministic here, so three attempts cost three timeouts
    and learn nothing new."""
    calls = []

    class FakeTimeout(Exception):
        pass

    class FakeCompletions:
        def create(self, **kw):
            calls.append(1)
            raise FakeTimeout("timed out")

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    # The openai SDK is imported inside the call function precisely so this
    # module loads without it (see the llm_proposer docstring), and it is not a
    # requirements.txt entry - so the test must not import it either.
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(OpenAI=FakeClient,
                                              APITimeoutError=FakeTimeout))
    monkeypatch.setattr(lp, "_detect_ollama_model", lambda: "gemma4:26b")
    with pytest.raises(RuntimeError, match="timed out"):
        lp._call_ollama("hi")
    assert len(calls) == 1


def test_auto_research_reads_the_env_file_for_the_timeout(monkeypatch):
    """The configured timeout has to reach the process that calls the model.

    It did not: .env carried GTRADE_AR_LLM_TIMEOUT=3600 but only push_signals and
    ab_genomes loaded the file, so the agent fell back to the 600s default and a
    26B model on CPU was killed at ten minutes every single call.
    """
    import auto_research
    assert "load_dotenv" in open(auto_research.__file__, encoding="utf-8").read()


def test_menu_settings_win_over_the_env_file(monkeypatch):
    """load_dotenv must not override what the launcher already exported, or the
    model picked in the menu would lose to a stale line in .env."""
    monkeypatch.setenv("GTRADE_AR_LLM_TIMEOUT", "120")
    from dotenv import load_dotenv
    load_dotenv()
    assert lp._llm_timeout() == 120.0


def test_model_override_treats_auto_as_unset(monkeypatch):
    """The launcher cannot emit a blank: cmd's  set "VAR="  DELETES the variable
    and load_dotenv then refills it from .env. That is how gemma4:26b (17 GB)
    stayed pinned on a 15.7 GB machine through every menu choice and crashed
    llama-server on every wiki compile (2026-08-14). "auto" is the sentinel."""
    from core.llm_proposer import model_override
    for blank in ("auto", "AUTO", " auto ", "", "   "):
        monkeypatch.setenv("GTRADE_AR_LLM_MODEL", blank)
        assert model_override() is None, blank
    monkeypatch.delenv("GTRADE_AR_LLM_MODEL", raising=False)
    assert model_override() is None
    monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "gemma4:12b")
    assert model_override() == "gemma4:12b"


def test_the_launcher_never_emits_a_blank_model(monkeypatch):
    """Positive control on the .bat itself: a blank there is silently overridden
    by .env, so the launcher must set a real value."""
    import os
    bat = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "auto_research.bat")
    text = open(bat, encoding="utf-8", errors="replace").read()
    assert 'set "GTRADE_AR_LLM_MODEL="' not in text
    assert 'set "GTRADE_AR_LLM_MODEL=auto"' in text


class TestCredentials:
    def test_a_missing_key_stops_before_the_retry_loop(self, monkeypatch):
        """The SDK raises at CLIENT CONSTRUCTION, inside the three-attempt loop,
        so one absent string produced three identical failures here and two more
        from the caller. And its own message names neither the variable nor the
        file to put it in."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(lp.ProviderUnavailable) as e:
            lp._call_anthropic("hi")
        assert "ANTHROPIC_API_KEY" in str(e.value) and ".env" in str(e.value)
        assert "ollama" in str(e.value)

    def test_a_key_that_is_only_whitespace_does_not_count(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        with pytest.raises(lp.ProviderUnavailable):
            lp._call_anthropic("hi")

    def test_a_custom_endpoint_needs_no_openai_key(self, monkeypatch):
        """GTRADE_AR_LLM_BASE_URL points the same backend at LM Studio and
        friends, which the docstring promises and which have no key. The first
        half is the positive control: without a base URL the key IS required."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GTRADE_AR_LLM_BASE_URL", raising=False)
        with pytest.raises(lp.ProviderUnavailable):
            lp._call_openai("hi")

        monkeypatch.setenv("GTRADE_AR_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        with pytest.raises(Exception) as e:
            lp._call_openai("hi")
        assert not isinstance(e.value, lp.ProviderUnavailable), e.value

    def test_ollama_never_asks_for_a_key(self, monkeypatch):
        """A fully local, keyless setup is the whole point of the third option.

        Pointed at a closed port on purpose: the first version of this test
        reached the real Ollama on this machine and got a 200, so it asserted
        nothing about keys and would have passed with the check in place.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GTRADE_AR_LLM_BASE_URL", "http://127.0.0.1:1/v1")
        monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "whatever")
        with pytest.raises(Exception) as e:
            lp._call_ollama("hi")
        assert not isinstance(e.value, lp.ProviderUnavailable), e.value


class TestTerminalErrors:
    class _Resp:
        status_code = 400

        @staticmethod
        def json():
            return {"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "Your credit balance is too low to access the "
                           "Anthropic API."}}

    class _Status(Exception):
        """What an SDK status error looks like: a useless str(), a useful body.

        The real one stringifies as "Connection error." while the response says
        the balance is empty, which is exactly how the diagnosis was lost.
        """

        status_code = 400
        response = None

        def __str__(self):
            return "Connection error."

    def _boom(self):
        exc = self._Status()
        exc.response = self._Resp()
        return exc

    def test_the_providers_own_message_survives(self):
        assert "credit balance" in lp._api_detail(self._boom())

    def test_a_terminal_status_is_not_retried_and_is_not_a_refusal(self,
                                                                   monkeypatch):
        """Eighteen calls went out for one empty wallet: three SDK retries
        inside three loop attempts inside two judge attempts."""
        calls = []
        exc = self._boom()

        class _Client:
            class messages:
                @staticmethod
                def create(**_kw):
                    calls.append(1)
                    raise exc

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        fake = types.SimpleNamespace(Anthropic=lambda *a, **k: _Client())
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        with pytest.raises(lp.ProviderUnavailable) as e:
            lp._call_anthropic("hi")
        assert len(calls) == 1, "a deterministic 400 was retried"
        assert "credit balance" in str(e.value) and "400" in str(e.value)

    def test_a_retryable_failure_still_gets_its_three_attempts(self,
                                                              monkeypatch):
        """The positive control. 429 and a dropped connection are exactly the
        cases the loop exists for, and they must not be swept up by this."""
        calls = []

        class _Client:
            class messages:
                @staticmethod
                def create(**_kw):
                    calls.append(1)
                    raise RuntimeError("connection reset")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        fake = types.SimpleNamespace(Anthropic=lambda *a, **k: _Client())
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        with pytest.raises(RuntimeError) as e:
            lp._call_anthropic("hi")
        assert not isinstance(e.value, lp.ProviderUnavailable)
        assert len(calls) == 3
