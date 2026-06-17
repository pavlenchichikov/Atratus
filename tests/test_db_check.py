"""Tests for db_check audit logic (in-memory SQLite, no real market.db)."""
import datetime as dt
import sqlite3

import db_check as dbc


def _cur():
    return sqlite3.connect(":memory:").cursor()


def _mk_price(cur, name, rows):
    cur.execute(
        f"CREATE TABLE {name} "
        "(Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL)"
    )
    cur.executemany(f"INSERT INTO {name} VALUES (?,?,?,?,?,?)", rows)


def _days_ago(n):
    return (dt.date.today() - dt.timedelta(days=n)).isoformat()


def test_price_tables_excludes_log_tables():
    cur = _cur()
    _mk_price(cur, "btc", [(_days_ago(1), 1, 2, 0.5, 1.5, 10)])
    cur.execute("CREATE TABLE guru_log (Date TEXT, note TEXT)")
    assert dbc._price_tables(cur, ["btc", "guru_log"]) == ["btc"]


def test_ohlc_separates_critical_from_minor():
    cur = _cur()
    _mk_price(cur, "x", [
        (_days_ago(3), 10, 12, 9, 11, 5),    # ok
        (_days_ago(2), 10, 12, 9, 11, -5),   # critical: Volume < 0
        (_days_ago(1), 10, 12, 9, 13, 5),    # minor: Close > High, all else fine
    ])
    out = dbc.check_ohlc(cur, ["x"])
    assert out["x"]["critical"] == 1
    assert out["x"]["minor"] == 1


def test_freshness_threshold_differs_daily_vs_weekly():
    cur = _cur()
    d14 = _days_ago(14)
    _mk_price(cur, "a", [(d14, 1, 2, 0.5, 1.5, 1)])
    _mk_price(cur, "a_weekly", [(d14, 1, 2, 0.5, 1.5, 1)])
    stale = dbc.check_freshness(cur, ["a", "a_weekly"])
    assert "a" in stale          # 14d > 7d daily limit
    assert "a_weekly" not in stale  # 14d < 21d weekly limit


def test_gaps_ignores_old_history():
    cur = _cur()
    _mk_price(cur, "old", [
        ("2000-01-01", 1, 2, 0.5, 1.5, 1),
        ("2000-06-01", 1, 2, 0.5, 1.5, 1),  # huge gap, but ends long ago
    ])
    assert "old" not in dbc.check_gaps(cur, ["old"])


def test_gaps_flags_recent_hole():
    cur = _cur()
    rows = [(_days_ago(d), 1, 2, 0.5, 1.5, 1) for d in (60, 59, 30, 29)]  # 29-day hole
    _mk_price(cur, "r", rows)
    gaps = dbc.check_gaps(cur, ["r"])
    assert "r" in gaps and gaps["r"][0] == 29


def test_low_data_flags_thin_table():
    cur = _cur()
    _mk_price(cur, "thin", [(_days_ago(i), 1, 2, 0.5, 1.5, 1) for i in range(1, 4)])
    assert dbc.check_low_data(cur, ["thin"]) == {"thin": 3}


def test_coverage_matches_registry():
    cov = dbc.check_coverage(["btc", "btc_weekly"])
    assert set(cov) == {"missing", "orphan", "no_weekly"}
    assert "btc" not in cov["orphan"]  # btc is a real config asset
