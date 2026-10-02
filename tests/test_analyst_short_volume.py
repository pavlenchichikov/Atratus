import sqlite3

from core.analyst import store, tools


def _db(tmp_path, monkeypatch, rows):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE finra_shvol (date TEXT, asset TEXT, short REAL, total REAL)")
    con.executemany("INSERT INTO finra_shvol VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()
    monkeypatch.setattr(store, "DB_PATH", path)


def _days(n):
    import datetime

    d0 = datetime.date(2026, 6, 1)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def _call(asset, today=None):
    return tools.call({"tool": "short_volume", "args": {}}, asset=asset, today=today)


def test_short_volume_gives_the_recent_share_and_its_z(tmp_path, monkeypatch):
    days = _days(90)
    rows = [(d, "aapl", 40.0 + (i % 5), 100.0) for i, d in enumerate(days)]
    rows[-1] = (days[-1], "aapl", 70.0, 100.0)                 # an unusual last day
    _db(tmp_path, monkeypatch, rows)
    out = _call("AAPL")["result"]
    assert out["last"]["date"] == days[-1] and out["last"]["short_share_pct"] == 70.0
    assert out["z_60d"] > 3
    assert len(out["recent"]) == 10


def test_a_rewound_run_sees_only_earlier_days(tmp_path, monkeypatch):
    days = _days(90)
    _db(tmp_path, monkeypatch, [(d, "aapl", 40.0, 100.0) for d in days])
    out = _call("AAPL", today=days[50])["result"]
    assert out["last"]["date"] == days[49]


def test_it_is_offered_only_for_us_listed_names():
    names = {t.name for t in tools.available(asset="SBER")}
    assert "short_volume" not in names
    assert "short_volume" in {t.name for t in tools.available(asset="AAPL")}
