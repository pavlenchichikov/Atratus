import json

import pytest

from core.analyst import scout

S = {"movers": [{"asset": "NVDA", "ret_1": 5.0, "ret_5": 9.0},
                {"asset": "GOLD", "ret_1": -2.0, "ret_5": -4.0}],
     "misses": [{"asset": "SBER", "date": "2026-09-20", "direction": "up", "ret": -1.0}],
     "stale": ["BTC"]}


def test_the_scout_keeps_known_assets_once_and_caps_the_list():
    reply = json.dumps({"assets": [{"asset": "NVDA", "reason": "big move"},
                                   {"asset": "NOPE", "reason": "x"},
                                   {"asset": "nvda", "reason": "again"},
                                   {"asset": "SBER", "reason": "missed"}]})
    got = scout.pick(lambda p: reply, S, max_assets=5)
    assert [g["asset"] for g in got] == ["NVDA", "SBER"] and got[0]["by"] == "scout"
    assert len(scout.pick(lambda p: reply, S, max_assets=1)) == 1


def test_a_failed_scout_falls_back_to_misses_movers_stale():
    got = scout.pick(lambda p: "no json", S, max_assets=3)
    assert [g["asset"] for g in got] == ["SBER", "NVDA", "GOLD"]
    assert all(g["by"] == "fallback" for g in got)


def test_is_stale():
    assert scout._is_stale(None, "2026-09-20", 7)
    assert scout._is_stale("2026-09-10", "2026-09-20", 7)
    assert not scout._is_stale("2026-09-18", "2026-09-20", 7)


def test_the_summary_carries_no_model_field(tmp_path, monkeypatch):
    import sqlite3

    import config
    from core.analyst import dossier as dz
    from core.analyst import store
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    for t, step in (("sp500", 1.0), ("gold", -1.0)):
        con.execute(f'CREATE TABLE "{t}" (Date TEXT, Open REAL, High REAL, Low REAL, Close REAL)')
        for i in range(1, 11):
            con.execute(f'INSERT INTO "{t}" VALUES (?,1,1,1,?)',
                        ("2026-09-%02d" % i, 100 + step * i))
    con.commit()
    con.close()
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setattr(config, "FULL_ASSET_MAP", {"SP500": "^GSPC", "GOLD": "GC=F"})
    s = scout.summary(today="2026-09-20", panel=("SP500",), db_path=path)
    flat = json.dumps(s)
    assert set(s) == {"movers", "misses", "stale"} and len(s["movers"]) == 2
    assert s["stale"] == ["SP500"]
    assert not any('"%s"' % k in flat for k in dz.FORBIDDEN_KEYS)


def test_picks_are_logged(tmp_path):
    import sqlite3
    path = str(tmp_path / "m.db")
    scout.log_picks("2026-09-20", [{"asset": "NVDA", "reason": "r", "by": "scout"}],
                    "ollama:g", db_path=path)
    assert sqlite3.connect(path).execute(
        "SELECT asset, picked_by, brain FROM analyst_auto_log").fetchall() == [
        ("NVDA", "scout", "ollama:g")]


def test_a_timeout_stops_the_scout():
    from core.llm_proposer import CallTimedOut

    def dead(p):
        raise CallTimedOut("slow")

    with pytest.raises(CallTimedOut):
        scout.pick(dead, S)
