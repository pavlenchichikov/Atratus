def test_wiki_on_env(monkeypatch):
    import core.ar_wiki as w
    monkeypatch.delenv("GTRADE_AR_WIKI", raising=False)
    assert w.wiki_on() is False
    monkeypatch.setenv("GTRADE_AR_WIKI", "1")
    assert w.wiki_on() is True


def test_wiki_summary_and_note_replicated(tmp_path, monkeypatch):
    import core.ar_wiki as w
    monkeypatch.setattr(w, "WIKI_DIR", str(tmp_path / "_ar_wiki"))
    assert w.wiki_summary() == ""                       # absent -> empty, no raise
    w.note_replicated("drops=[macro_tnx]", "neural_lift +0.6, replicated")
    s = w.wiki_summary()
    assert "REPLICATED" in s and "general" in s
    # summary truncates
    assert len(w.wiki_summary(max_chars=5)) <= 5


def _findings(monkeypatch, tmp_path, records):
    import core.ar_memory as am
    import json as _j
    p = str(tmp_path / "_ar_findings.json")
    monkeypatch.setattr(am, "FINDINGS_PATH", p)
    with open(p, "w", encoding="utf-8") as f:
        _j.dump(records, f)


def test_compile_wiki_folds_new_findings(tmp_path, monkeypatch):
    import core.ar_wiki as w
    monkeypatch.setattr(w, "WIKI_DIR", str(tmp_path / "_ar_wiki"))
    monkeypatch.setenv("GTRADE_AR_WIKI", "1")
    _findings(monkeypatch, tmp_path, [{"mode": "qd", "winners": [{"tag": "x"}]}])
    monkeypatch.setattr(w, "_backend",
                        lambda: (lambda prompt: "## features\n(high) macro drops hurt in calm\n"))
    n = w.compile_wiki()
    assert n == 1
    assert "macro drops hurt" in w.wiki_summary()
    # second call, no new findings -> no-op
    assert w.compile_wiki() == 0


def test_compile_wiki_off_or_error_is_noop(tmp_path, monkeypatch):
    import core.ar_wiki as w
    monkeypatch.setattr(w, "WIKI_DIR", str(tmp_path / "_ar_wiki"))
    _findings(monkeypatch, tmp_path, [{"mode": "qd"}])
    monkeypatch.delenv("GTRADE_AR_WIKI", raising=False)
    assert w.compile_wiki() == 0 and w.wiki_summary() == ""          # off
    monkeypatch.setenv("GTRADE_AR_WIKI", "1")
    monkeypatch.setattr(w, "_backend", lambda: (lambda p: (_ for _ in ()).throw(RuntimeError())))
    assert w.compile_wiki() == 0 and w.wiki_summary() == ""          # error -> unchanged
