"""A trailing row is not a setup, and must never be scored as one.

Measured 2026-08-22: the first twelve resolutions in the live journal were ALL
trailing rows. Their stop belongs to a position opened days earlier and has
since ratcheted to sit near today's close, so replaying them as "enter at the
zone edge today, exit at that stop" invented a trade with a median distance of
0.56 ATR from entry to stop where the policy asks for 2.99 - and all twelve
"lost" on bar one. That read as evidence against the levels policy. It was
evidence about the journal.
"""
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import performance_tracker as pt


def _journal(tmp_path, rows):
    db = tmp_path / "market.db"
    con = sqlite3.connect(db)
    cur = con.cursor()
    pt._ensure_level_table(cur)
    pt._ensure_table(cur)
    for r in rows:
        cur.execute(
            "INSERT INTO level_log (date, asset, signal, close, atr, entry_low,"
            " entry_high, stop, trailing) VALUES (?,?,?,?,?,?,?,?,?)", r)
    con.commit()
    con.close()
    return str(db)


def test_a_trailing_row_is_closed_as_not_a_setup_and_never_priced(monkeypatch,
                                                                  tmp_path):
    db = _journal(tmp_path, [
        # close 100, atr 10, a zone around the close and a stop ratcheted to
        # 0.3 ATR below it - the shape every one of the twelve had.
        ("2026-08-18", "AAA", "BUY", 100.0, 10.0, 95.0, 105.0, 97.0, 1),
    ])
    monkeypatch.setattr(pt, "DB_PATH", db)
    out = pt.update_level_outcomes()
    assert out["not_setups"] == 1
    assert out["resolved"] == 0

    con = sqlite3.connect(db)
    row = con.execute("SELECT exit_reason, entered, ret_net FROM level_log").fetchone()
    con.close()
    assert row[0].startswith("not a setup")
    assert row[1] is None, "a row nobody could trade has no entry"
    assert row[2] is None, "and no return to average into anything"


def test_a_fresh_setup_is_still_taken_through_the_resolver(monkeypatch, tmp_path):
    """The control: the fix must not silence the rows that ARE evidence."""
    db = _journal(tmp_path, [
        ("2026-08-18", "AAA", "BUY", 100.0, 10.0, 95.0, 105.0, 80.0, 0),
    ])
    monkeypatch.setattr(pt, "DB_PATH", db)
    out = pt.update_level_outcomes()
    assert out["not_setups"] == 0
    con = sqlite3.connect(db)
    reason = con.execute("SELECT exit_reason FROM level_log").fetchone()[0]
    con.close()
    # No price history in this fixture, so it stays pending rather than being
    # marked - what matters is that it was NOT written off as "not a setup".
    assert reason is None or not reason.startswith("not a setup")
