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


def test_a_good_fetch_is_archived_point_in_time(monkeypatch, tmp_path):
    """Smart-Lab publishes the CURRENT table and no history, so these numbers
    cannot be model features today: joining a 2025 P/E onto a 2015 row leaks ten
    years. Archiving is the only way the history ever exists."""
    import sqlite3
    _point_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(G, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(net, "http_get", lambda *a, **k: _Resp())
    G.fetch_smartlab_data()
    con = sqlite3.connect(str(tmp_path / "market.db"))
    rows = con.execute("SELECT ticker, pe FROM fundamentals_history").fetchall()
    con.close()
    assert ("SBER", 3.7) in rows


def test_the_archive_keeps_one_row_per_ticker_per_day(monkeypatch, tmp_path):
    import sqlite3
    _point_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(G, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(net, "http_get", lambda *a, **k: _Resp())
    G.fetch_smartlab_data()
    G.fetch_smartlab_data()
    con = sqlite3.connect(str(tmp_path / "market.db"))
    n = con.execute("SELECT count(*) FROM fundamentals_history").fetchone()[0]
    con.close()
    assert n == 1


def test_a_failed_fetch_archives_nothing(monkeypatch, tmp_path):
    """Positive control: an empty map must not write a row of nulls that a
    future feature would read as a real observation."""
    import sqlite3
    _point_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(G, "BASE_DIR", str(tmp_path))

    def dead(*a, **k):
        raise RuntimeError("no route")
    monkeypatch.setattr(net, "http_get", dead)
    G.fetch_smartlab_data()
    db = tmp_path / "market.db"
    if db.exists():
        con = sqlite3.connect(str(db))
        try:
            n = con.execute("SELECT count(*) FROM fundamentals_history").fetchone()[0]
        except sqlite3.Error:
            n = 0
        con.close()
        assert n == 0
