import pytest

from core import llm_proposer as lp
from core import ollama_settings as osets


def test_every_setting_has_a_valid_default_and_a_description():
    for key, scope, dflt, check, desc in osets.SETTINGS:
        assert scope in ("env", "server") and desc
        assert dflt == "" or check(dflt), key


def test_set_writes_env_refuses_junk_and_empty_means_default(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")
    monkeypatch.setattr(osets, "ENV_PATH", str(env))
    monkeypatch.delenv("GTRADE_OLLAMA_RAM_MARGIN_MB", raising=False)
    assert osets.get("GTRADE_OLLAMA_RAM_MARGIN_MB") == "2048"
    n = [s[0] for s in osets.SETTINGS].index("GTRADE_OLLAMA_RAM_MARGIN_MB") + 1
    osets.set_value(str(n), "1500")                         # by menu number
    assert "GTRADE_OLLAMA_RAM_MARGIN_MB" in env.read_text()
    assert osets.get("GTRADE_OLLAMA_RAM_MARGIN_MB") == "1500"
    with pytest.raises(ValueError):
        osets.set_value("GTRADE_OLLAMA_RAM_MARGIN_MB", "abc")
    with pytest.raises(ValueError):
        osets.set_value("GTRADE_NOT_A_KNOB", "1")
    osets.set_value("GTRADE_OLLAMA_RAM_MARGIN_MB", "")
    assert "GTRADE_OLLAMA_RAM_MARGIN_MB" not in env.read_text()
    assert osets.get("GTRADE_OLLAMA_RAM_MARGIN_MB") == "2048"


def _capture_payload(monkeypatch):
    bodies = []

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ok"}}

    class Client:
        def __init__(self, *a, **k):
            pass

        def post(self, url, json=None, **k):
            bodies.append(json)
            return Resp()

    import httpx
    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(lp, "_ollama_unload", lambda *a, **k: None)
    return bodies


def test_context_and_mmap_reach_every_local_call_and_not_the_cloud(monkeypatch):
    bodies = _capture_payload(monkeypatch)
    monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "any-model:7b")
    monkeypatch.setenv("GTRADE_OLLAMA_NUM_CTX", "16384")
    monkeypatch.setenv("GTRADE_OLLAMA_MMAP", "1")
    monkeypatch.setenv("GTRADE_AR_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    lp._call_ollama("hi")
    assert bodies[-1]["options"]["num_ctx"] == 16384
    assert bodies[-1]["options"]["use_mmap"] is True
    monkeypatch.setenv("GTRADE_OLLAMA_MMAP", "")
    monkeypatch.setenv("GTRADE_OLLAMA_NUM_CTX", "")
    lp._call_ollama("hi")
    assert "use_mmap" not in bodies[-1]["options"] and "num_ctx" not in bodies[-1]["options"]
    monkeypatch.setenv("GTRADE_OLLAMA_NUM_CTX", "16384")
    monkeypatch.setenv("GTRADE_AR_LLM_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    lp._call_ollama("hi")
    assert "num_ctx" not in bodies[-1].get("options", {})


def test_ram_share_scales_the_need_for_any_model(monkeypatch):
    base = "http://127.0.0.1:11434/v1"
    monkeypatch.setattr(lp, "_ollama_loaded", lambda b: [])
    monkeypatch.setattr(lp, "_ollama_size_mb", lambda b, m: 17742)
    monkeypatch.setattr(lp, "_ram_free_mb", lambda: 12000)
    monkeypatch.setenv("GTRADE_OLLAMA_RAM_WAIT", "0")
    with pytest.raises(lp.ProviderUnavailable):
        lp.wait_for_ram(base, "big", sleep=lambda s: None)
    monkeypatch.setenv("GTRADE_OLLAMA_RAM_SHARE", "0.4")    # 7097 + 2000 + 2048 < 12000
    lp.wait_for_ram(base, "big", sleep=lambda s: None)
