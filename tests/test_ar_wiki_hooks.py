def test_compile_only_when_on(monkeypatch):
    import core.ar_wiki as w
    calls = []
    monkeypatch.setattr(w, "compile_wiki", lambda: calls.append(1) or 0)
    monkeypatch.delenv("GTRADE_AR_WIKI", raising=False)
    if w.wiki_on():
        w.compile_wiki()
    assert calls == []
    monkeypatch.setenv("GTRADE_AR_WIKI", "1")
    if w.wiki_on():
        w.compile_wiki()
    assert calls == [1]
