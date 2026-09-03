"""The ledger that ties an adoption to what happened live afterwards.

The gate measures a backtest. Nothing in the project could say whether an
adoption moved the live number: 49.1% over 8742 calls on 2026-09-03, with no
link to the history of adoptions. This is the missing link, and it is written at
adoption time because the "before" side cannot be reconstructed later.
"""
import json
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import adoption_ledger as al


def _db(tmp_path, rows):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,"
                " actual_next_ret REAL)")
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def test_only_scored_directional_calls_count(tmp_path):
    """WAIT is not a recommendation, and a row without an outcome is not a
    result. Counting either would dilute the number with days the system either
    declined to speak or has not been graded on yet."""
    db = _db(tmp_path, [
        ("2026-01-01", "AAA", "BUY", 0.01),    # right
        ("2026-01-02", "AAA", "SELL", 0.01),   # wrong
        ("2026-01-03", "AAA", "WAIT", 0.01),   # never counted
        ("2026-01-04", "AAA", "BUY", None),    # not graded yet
    ])
    got = al.live_accuracy(db_path=db)
    assert got == {"n": 2, "correct": 1, "accuracy": 0.5}


def test_it_can_be_asked_about_the_assets_an_adoption_touched(tmp_path):
    db = _db(tmp_path, [
        ("2026-01-01", "AAA", "BUY", 0.01),
        ("2026-01-01", "BBB", "BUY", -0.01),
        ("2026-01-02", "BBB", "SELL", -0.01),
    ])
    assert al.live_accuracy(["aaa"], db_path=db)["accuracy"] == 1.0
    assert al.live_accuracy(["BBB"], db_path=db)["accuracy"] == 0.5
    assert al.live_accuracy(db_path=db)["n"] == 3


def test_the_before_side_stops_at_the_adoption_date(tmp_path):
    """Otherwise the entry would bank days the adoption had already influenced,
    and the comparison would be with itself."""
    db = _db(tmp_path, [
        ("2026-01-01", "AAA", "BUY", 0.01),
        ("2026-01-09", "AAA", "BUY", -0.01),
    ])
    path = str(tmp_path / "ledger.json")
    e = al.record("A", ["AAA"], "measured", path=path, db_path=db, today="2026-01-05")
    assert e["before"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    after = al.live_accuracy(["AAA"], since="2026-01-05", db_path=db)
    assert after == {"n": 1, "correct": 0, "accuracy": 0.0}


def test_a_missing_journal_is_a_blank_not_a_crash(tmp_path):
    """An adoption must never be blocked by its own bookkeeping."""
    got = al.live_accuracy(db_path=str(tmp_path / "nothing.db"))
    assert got == {"n": 0, "correct": 0, "accuracy": None}
    path = str(tmp_path / "l.json")
    al.record("A", None, "", path=path, db_path=str(tmp_path / "nothing.db"))
    assert json.load(open(path, encoding="utf-8"))[0]["before"]["accuracy"] is None


def test_the_report_shows_the_counts_beside_the_percentages(tmp_path):
    """With a few dozen calls the difference means nothing, and the count is
    what makes that visible instead of inviting a story."""
    db = _db(tmp_path, [("2026-01-01", "AAA", "BUY", 0.01)])
    path = str(tmp_path / "l.json")
    al.record("A", ["AAA"], "", path=path, db_path=db, today="2026-01-05")
    text = "\n".join(al.report_lines(path=path, db_path=db))
    assert "100.0% of 1" in text
    assert "not a paired comparison" in text
    assert "No adoption" in "\n".join(al.report_lines(path=str(tmp_path / "none.json")))
