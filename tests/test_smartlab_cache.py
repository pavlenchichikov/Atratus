"""Smart-Lab fundamentals: the router, and the last-good-copy fallback."""
import json
import os

import guru_report as G
import net

TABLE = ('<table><tr><th>Тикер</th><th>P/E</th><th>долг/EBITDA</th>'
         '<th>ДД ао, %</th></tr>'
         '<tr><td>SBER</td><td>3.7</td><td>0.5</td><td>13.6</td></tr></table>')


class _Resp:
    status_code = 200
    text = TABLE


def _point_cache(monkeypatch, tmp_path):
    p = str(tmp_path / "_smartlab_cache.json")
    monkeypatch.setattr(G, "SMARTLAB_CACHE_PATH", p)
    return p


def test_a_good_fetch_is_cached(monkeypatch, tmp_path):
    p = _point_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(net, "http_get", lambda *a, **k: _Resp())
    got = G.fetch_smartlab_data()
    assert "SBER" in got
    assert os.path.exists(p)
    assert json.load(open(p, encoding="utf-8"))["map"]["SBER"]["pe"] == 3.7


def test_a_dead_route_falls_back_to_the_cached_table(monkeypatch, tmp_path):
    """One failed request used to blank the fundamentals of every Russian name
    for the whole run, silently. Fundamentals change quarterly; yesterday's copy
    beats nothing."""
    p = _point_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(net, "http_get", lambda *a, **k: _Resp())
    G.fetch_smartlab_data()

    def dead(*a, **k):
        raise RuntimeError("no route to host")
    monkeypatch.setattr(net, "http_get", dead)
    assert "SBER" in G.fetch_smartlab_data()
    assert os.path.exists(p)


def test_without_a_cache_a_dead_route_is_empty_not_an_exception(monkeypatch, tmp_path):
    """Positive control: it is the CACHE that answers above, not the parser
    quietly succeeding on a failed request."""
    _point_cache(monkeypatch, tmp_path)

    def dead(*a, **k):
        raise RuntimeError("no route to host")
    monkeypatch.setattr(net, "http_get", dead)
    assert G.fetch_smartlab_data() == {}


def test_a_non_200_also_falls_back(monkeypatch, tmp_path):
    _point_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(net, "http_get", lambda *a, **k: _Resp())
    G.fetch_smartlab_data()

    class Bad:
        status_code = 403
        text = ""
    monkeypatch.setattr(net, "http_get", lambda *a, **k: Bad())
    assert "SBER" in G.fetch_smartlab_data()


def test_it_goes_through_the_project_router(monkeypatch, tmp_path):
    """Smart-Lab answers on one route and not the other depending on the VPN;
    a bare requests.get sees only whichever is default."""
    _point_cache(monkeypatch, tmp_path)
    seen = {}

    def spy(url, **kw):
        seen["url"] = url
        seen["route"] = kw.get("route")
        return _Resp()
    monkeypatch.setattr(net, "http_get", spy)
    G.fetch_smartlab_data()
    assert "smart-lab.ru" in seen["url"] and seen["route"] == "auto"
