from core import model_picker as mp


def test_a_number_picks_from_the_list_and_text_is_taken_as_is():
    models = ["gemma4:26b", "gpt-oss:120b"]
    assert mp.choose(models, "2", "") == "gpt-oss:120b"
    assert mp.choose(models, "qwen3:32b", "") == "qwen3:32b"
    assert mp.choose(models, "", "auto") == "auto"
    assert mp.choose(models, " ", "") == ""
    assert mp.choose(models, "9", "auto") == "auto"      # out of range: the default


def test_each_provider_has_its_list(monkeypatch):
    monkeypatch.setattr(mp, "_ollama_local", lambda: ["gemma4:26b"])
    monkeypatch.setattr(mp, "_ollama_cloud", lambda: ["gpt-oss:120b"])
    assert mp.models_for("ollama") == ["gemma4:26b"]
    assert mp.models_for("ollama-cloud") == ["gpt-oss:120b"]
    cc = mp.models_for("claude-code")
    assert cc[:2] == ["sonnet", "opus"] and "claude-opus-5-5" in cc
    assert "claude-opus-5-5" in mp.models_for("anthropic")
    assert mp.models_for("openai")


def test_a_dead_listing_leaves_typing_open(monkeypatch):
    def down():
        raise RuntimeError("cannot reach Ollama")

    monkeypatch.setattr(mp, "_ollama_local", down)
    assert mp.models_for("ollama") == []


def test_list_then_pick_writes_the_choice_without_reading_stdin(tmp_path, monkeypatch):
    """The .bat reads every answer itself: a Python input() on a redirected
    stdin swallowed the rest of the menu's input in a dry run."""
    monkeypatch.setattr(mp, "LIST_PATH", str(tmp_path / "models.json"))
    monkeypatch.setattr(mp, "models_for", lambda p: ["a", "b"])
    monkeypatch.setattr("builtins.input", lambda prompt="": 1 / 0)
    assert mp.main(["list", "ollama"]) == 0
    out = tmp_path / "pick.txt"
    assert mp.main(["pick", "2", str(out), "auto"]) == 0
    assert out.read_text(encoding="utf-8") == "b"
    assert mp.main(["pick", "", str(out), "auto"]) == 0
    assert out.read_text(encoding="utf-8") == "auto"
