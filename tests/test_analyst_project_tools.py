import sqlite3

import pytest

from core.analyst import store, tools

FORBIDDEN = ("prediction_log", "signals", "guru_log", "champion_registry")


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    for t, base in (("sp500", 100.0), ("gold", 50.0)):
        con.execute(f'CREATE TABLE "{t}" (Date TEXT, Open REAL, High REAL, Low REAL, Close REAL)')
        for i in range(30):
            d = "2026-08-%02d" % (i + 1)
            c = base + i
            con.execute(f'INSERT INTO "{t}" VALUES (?,?,?,?,?)', (d, c, c + 1, c - 1, c))
    con.commit()
    con.close()
    monkeypatch.setattr(store, "DB_PATH", path)
    return path


def test_the_tools_are_registered_and_rewind():
    names = {t.name for t in tools.available(today="2026-08-10", asset="SP500")}
    assert {"price_history", "compare", "sector_peers", "my_record"} <= names


def test_price_history_never_shows_today_or_later(db):
    rows = tools.call({"tool": "price_history", "args": {"days": 60}},
                      asset="SP500", today="2026-08-10")["result"]["bars"]
    assert rows and max(r["date"] for r in rows) == "2026-08-09"


def test_unknown_asset_and_short_history_do_not_raise(db):
    out = tools.call({"tool": "compare", "args": {"assets": "NOPE,GOLD", "days": 5}},
                     asset="SP500", today="2026-08-03")
    assert "error" not in out
    assert out["result"]["unknown"] == ["NOPE"]
    out = tools.call({"tool": "price_history", "args": {"days": 5}},
                     asset="ZZZZ", today="2026-08-10")
    assert "error" not in out and out["result"]["bars"] == []


def test_my_record_hides_outcomes_that_resolve_on_or_after_today(db):
    store.write_judgment({"date": "2026-08-05", "asset": "SP500", "horizon": 1,
                          "direction": "up", "conviction": 3, "thesis": "a"}, db_path=db)
    store.write_judgment({"date": "2026-08-08", "asset": "SP500", "horizon": 5,
                          "direction": "down", "conviction": 2, "thesis": "b"}, db_path=db)
    con = sqlite3.connect(db)             # the backfill's job: fill the outcomes
    con.execute("UPDATE analyst_log SET realized_ret = CASE date "
                "WHEN '2026-08-05' THEN 0.01 ELSE -0.02 END")
    con.commit()
    con.close()
    got = tools.call({"tool": "my_record", "args": {}}, asset="SP500",
                     today="2026-08-10")["result"]["judgments"]
    # 08-05 +1 bar resolves 08-06 < 08-10: shown. 08-08 +5 bars resolves 08-13: hidden.
    assert [g["date"] for g in got] == ["2026-08-05"] and got[0]["hit"] is True


def test_no_tool_touches_a_model_signal_table(db, monkeypatch):
    seen = []
    real = sqlite3.connect

    def spy(*a, **k):
        con = real(*a, **k)
        con.set_trace_callback(seen.append)
        return con

    monkeypatch.setattr(sqlite3, "connect", spy)
    for req in ({"tool": "price_history", "args": {}},
                {"tool": "compare", "args": {"assets": "GOLD"}},
                {"tool": "sector_peers", "args": {}},
                {"tool": "my_record", "args": {}}):
        tools.call(req, asset="SP500", today="2026-08-20")
    sql = " ".join(seen).lower()
    assert sql and not any(f in sql for f in FORBIDDEN)


def test_a_duplicated_bar_does_not_count_as_an_extra_day(db):
    from core.analyst import project_tools as pt
    con = sqlite3.connect(db)
    con.execute("INSERT INTO sp500 VALUES ('2026-08-09 00:00:00', 1, 1, 1, 1)")
    con.commit()
    con.close()
    # 08-08 + 2 bars resolves 08-10; before today 08-10 only 08-09 exists (twice).
    assert not pt._resolved_before("SP500", "2026-08-08", 2, "2026-08-10")
