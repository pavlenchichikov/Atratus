"""Healing daily bars stored before their session closed. Synthetic database
and an injected fetch: no network, no market.db."""
import sqlite3

import pandas as pd

import heal_partial_bars as H


def _db(tmp_path):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sber (Date TEXT, open REAL, close REAL, high REAL, low REAL, volume REAL)")
    rows = [("2026-07-01", 100, 101, 102, 99, 10),
            ("2026-07-02", 101, 103.5, 104, 100, 5),     # a snapshot: the day closed at 105
            ("2026-07-03", 105, 104, 106, 103, 12)]
    con.executemany("INSERT INTO sber VALUES (?,?,?,?,?,?)", rows)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, "
                "probability REAL, actual_next_ret REAL, correct INTEGER)")
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?)", [
        ("2026-06-20", "SBER", "BUY", 0.6, 0.01, 1),
        ("2026-07-01", "SBER", "BUY", 0.6, 0.0247, 1),
        ("2026-07-02", "SBER", "SELL", 0.4, 0.0048, 0),
        ("2026-07-02", "GAZP", "BUY", 0.6, 0.02, 1)])
    con.execute("CREATE TABLE level_log (date TEXT, asset TEXT, signal TEXT, trailing INTEGER, "
                "entered INTEGER, entry_date TEXT, entry_price REAL, exit_date TEXT, "
                "exit_price REAL, exit_reason TEXT, bars_held INTEGER, ret_net REAL)")
    con.executemany("INSERT INTO level_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("2026-06-10", "SBER", "BUY", 0, 1, "2026-06-10", 90, "2026-06-12", 91, "signal", 2, .01),
        ("2026-06-25", "SBER", "BUY", 0, 1, "2026-06-25", 95, "2026-07-02", 103.5, "signal", 5, .08),
        ("2026-06-26", "SBER", "BUY", 1, None, None, None, None, None,
         "not a setup: position already open", None, None)])
    con.commit()
    return con


def _fresh():
    idx = ["2026-07-01", "2026-07-02", "2026-07-03"]
    return pd.DataFrame({"open": [100, 101, 105], "close": [101, 105, 104], "high": [102, 106, 106],
                         "low": [99, 100, 103], "volume": [10, 20, 12]}, index=idx)


def test_the_scan_finds_only_the_snapshot(tmp_path):
    con = _db(tmp_path)
    assert H.partial_dates(con, "sber", _fresh()) == ["2026-07-02"]


def test_the_scan_changes_nothing(tmp_path):
    con = _db(tmp_path)
    H.partial_dates(con, "sber", _fresh())
    assert con.execute("SELECT close FROM sber WHERE Date='2026-07-02'").fetchone()[0] == 103.5


def test_apply_replaces_the_bar_and_resets_what_it_fed(tmp_path):
    con = _db(tmp_path)
    out = H.heal_asset(con, "SBER", "sber", _fresh(), ["2026-07-02"])
    assert con.execute("SELECT close, high, volume FROM sber WHERE Date='2026-07-02'").fetchone() \
        == (105, 106, 20)
    assert con.execute("SELECT count(*) FROM sber").fetchone()[0] == 3
    # the rows scored on or after the day before the healed bar are reset, others kept
    got = dict(con.execute("SELECT date || asset, actual_next_ret FROM prediction_log"))
    assert got["2026-06-20SBER"] == 0.01
    assert got["2026-07-01SBER"] is None and got["2026-07-02SBER"] is None
    assert got["2026-07-02GAZP"] == 0.02
    # a trade that exited on the healed bar is re-resolved; one that closed before is kept;
    # a "not a setup" row stays what it is
    lv = {r[0]: r[1] for r in con.execute("SELECT date, exit_reason FROM level_log")}
    assert lv["2026-06-10"] == "signal"
    assert lv["2026-06-25"] is None
    assert lv["2026-06-26"].startswith("not a setup")
    assert out == {"bars": 1, "dropped": 0, "predictions": 2, "levels": 1}


def test_a_bar_newer_than_the_last_finished_session_is_removed(tmp_path):
    """A snapshot of TODAY has no fresh counterpart yet, so comparing prices
    cannot find it, and data_engine resumes from the day after it. It goes, so
    the next data update fetches that day once it has closed."""
    con = _db(tmp_path)
    con.execute("INSERT INTO sber VALUES ('2026-07-04', 104, 104.2, 104.5, 103.9, 1)")
    con.commit()
    assert H.unfinished_dates(con, "sber", _fresh(), today="2026-07-04") == ["2026-07-04"]
    H.heal_asset(con, "SBER", "sber", _fresh(), ["2026-07-02"], drop=["2026-07-04"])
    dates = [r[0] for r in con.execute("SELECT Date FROM sber ORDER BY Date")]
    assert dates == ["2026-07-01", "2026-07-02", "2026-07-03"]


def test_a_source_with_a_hole_does_not_erase_what_we_stored(tmp_path):
    """TON: Yahoo now serves 2026-06-16 onward empty. Its last bar looked like
    the last finished session, and 62 real stored bars were dropped as
    unfinished. A source that has stopped is a vendor gap, not a live session."""
    con = _db(tmp_path)
    con.execute("INSERT INTO sber VALUES ('2026-07-20', 104, 104.2, 104.5, 103.9, 1)")
    con.commit()
    assert H.unfinished_dates(con, "sber", _fresh(), today="2026-07-20") == []
    assert H.unfinished_dates(con, "sber", _fresh(), today="2026-07-04") == ["2026-07-20"]


def test_apply_refuses_while_training_runs(monkeypatch):
    monkeypatch.setattr(H, "_training_active", lambda: (True, 3.0))
    assert H.main(["--apply"]) == 2


def test_the_operator_can_state_that_training_has_stopped(tmp_path, monkeypatch):
    """The check reads file ages, so it refuses for an hour after a run is
    killed. The flag is an explicit statement, never a default."""
    monkeypatch.setattr(H, "_training_active", lambda: (True, 3.0))
    monkeypatch.setattr(H, "DB_PATH", str(tmp_path / "m.db"))
    monkeypatch.setattr(H, "fresh_bars", lambda asset, since: None)
    assert H.main(["--apply", "--training-stopped", "--assets", "SBER"]) == 0
