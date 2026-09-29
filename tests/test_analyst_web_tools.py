from core.analyst import tools


def _call(name, **args):
    return tools.call({"tool": name, "args": args}, asset="AAPL")


def test_web_search_drops_opinion_sites(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    seen = {}

    def fake_post(url, payload, key):
        seen.update(url=url, payload=payload, key=key)
        return {"results": [
            {"title": "a", "url": "https://www.tipranks.com/x", "content": "t"},
            {"title": "b", "url": "https://www.sec.gov/y", "content": "filing text"}]}

    monkeypatch.setattr(tools, "_post", fake_post)
    out = _call("web_search", query="AAPL 10-Q")
    assert [r["url"] for r in out["result"]] == ["https://www.sec.gov/y"]
    assert seen["key"] == "k" and seen["payload"]["query"] == "AAPL 10-Q"


def test_web_search_without_key_is_unavailable_and_the_menu_says_so(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert "web_search" not in [t.name for t in tools.available()]
    assert "web_search: unavailable" in tools.spec_lines(asset="AAPL")
    assert "web_fetch" in [t.name for t in tools.available()]


def test_web_fetch_refuses_an_opinion_site(monkeypatch):
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: 1 / 0)
    out = _call("web_fetch", url="https://seekingalpha.com/article/1")
    assert out["result"] == {"refused": "opinion source"}


def test_web_fetch_strips_html_and_keeps_csv_tail(monkeypatch):
    monkeypatch.setattr(tools, "_get", lambda url, headers=None:
                        "<html><script>x()</script><p>Rate  4.25%</p></html>")
    assert _call("web_fetch", url="https://cbr.ru/a")["result"]["text"] == "Rate 4.25%"
    csv_text = "date,v\n" + "\n".join("2026-09-%02d,%d" % (i, i) for i in range(1, 29))
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: csv_text)
    rows = _call("web_fetch", url="https://x.org/a.csv")["result"]["rows"]
    assert rows[0] == "date,v" and rows[-1] == "2026-09-28,28"


def test_web_budget_adds_to_the_tool_budget(monkeypatch):
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "6")
    monkeypatch.setenv("GTRADE_ANALYST_WEB_CALLS", "4")
    assert tools.max_calls() == 10
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    assert tools.max_calls() == 0
