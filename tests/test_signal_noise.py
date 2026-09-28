"""Accuracy that ignores noise-sized outcomes, and the grade table built on it."""
import sqlite3

import pandas as pd

from core import signal_noise as sn


def _db(tmp_path, rows, closes):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,
        probability REAL, actual_next_ret REAL, correct INTEGER, cb_prob REAL,
        lstm_prob REAL, tf_prob REAL, tcn_prob REAL, timing_action TEXT)""")
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    dates = pd.bdate_range("2026-01-01", periods=len(closes)).strftime("%Y-%m-%d")
    pd.DataFrame({"Date": dates, "close": closes}).to_sql("xx", con, index=False)
    con.commit()
    return path, list(dates)


def test_a_noise_sized_outcome_is_left_out_of_the_noise_free_accuracy(tmp_path):
    # every day moves 1%, so the noise band is 0.5% and a 0.1% outcome is noise
    closes = [100 * 1.01 ** (i % 2) for i in range(80)]
    path, dates = _db(tmp_path, [], closes)
    d1, d2 = dates[70], dates[71]
    rows = [(d1, "XX", "BUY", 0.7, 0.02, 1, 0.7, 0.7, 0.7, 0.7, "ENTER:+1"),    # real, right
            (d2, "XX", "BUY", 0.7, -0.001, 0, 0.7, 0.4, 0.4, 0.4, "HOLD")]      # noise, wrong
    con = sqlite3.connect(path)
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    rep = sn.noise_report(path)
    assert rep["raw"]["n"] == 2 and rep["raw"]["acc"] == 0.5
    assert rep["clean"]["n"] == 1 and rep["clean"]["acc"] == 1.0
    assert rep["per_asset"]["XX"]["n"] == 1


def test_grades_are_scored_on_clean_outcomes_only(tmp_path):
    closes = [100 * 1.01 ** (i % 2) for i in range(80)]
    path, dates = _db(tmp_path, [], closes)
    rows = [(dates[70], "XX", "BUY", 0.9, 0.02, 1, 0.9, 0.9, 0.9, 0.9, "ENTER:+1"),
            (dates[71], "XX", "BUY", 0.9, 0.0001, 0, 0.9, 0.9, 0.9, 0.9, "HOLD")]
    con = sqlite3.connect(path)
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    agree = next(g for g in sn.noise_report(path)["grades"] if g["key"] == "agree")
    four = next(r for r in agree["rows"] if r["label"] == "4 of 4")
    assert four["n"] == 1 and four["acc"] == 1.0


def test_an_empty_journal_reports_nothing_rather_than_failing(tmp_path):
    path, _ = _db(tmp_path, [], [100.0] * 80)
    rep = sn.noise_report(path)
    assert rep["raw"]["n"] == 0 and rep["grades"] == []
