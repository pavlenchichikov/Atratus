"""What the issued levels actually did.

The product tells a person where to enter and where to bail, and until now
nothing recorded either the instruction or its outcome, so "did the levels make
money" could not be answered for a single day. These tests pin the two halves:
the row written on the day, and the reconcile that walks it forward over bars.

The trade ends on the stop or on the signal turning away from the side it was
issued for, which is the rule the asset card states and the one
core/positions.py already uses for a segment.
"""

import pandas as pd
import pytest

import performance_tracker as pt


def _bars(rows):
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df.index = pd.to_datetime(df.pop("date")).values
    return df


LEG = 0.0025          # COMMISSION + SLIPPAGE, two legs charged on a closed trade


def _resolve(bars, shown, signal="BUY", entry=(99.0, 101.0), stop=96.0):
    return pt._resolve_level_row(("2026-01-01", signal, entry[0], entry[1], stop),
                                 bars, shown, LEG)


def test_a_long_that_fills_and_stops_out_loses_the_move_and_the_costs():
    bars = _bars([
        ("2026-01-02", 100.0, 101.0, 99.5, 100.0),   # fills at the top of the zone
        ("2026-01-05", 98.0, 98.5, 95.0, 95.5),      # trades through the stop
    ])
    out = _resolve(bars, {})
    assert out["entered"] == 1 and out["exit_reason"] == "stop"
    assert out["entry_price"] == 100.0        # min(open, entry_high), the worse edge
    assert out["exit_price"] == 96.0
    assert out["ret_net"] == pytest.approx((96.0 - 100.0) / 100.0 - 2 * LEG)


def test_a_gap_through_the_stop_fills_at_the_gap_not_at_the_stop():
    """Anything else would report a loss the trade could not have taken."""
    bars = _bars([
        ("2026-01-02", 100.0, 101.0, 99.5, 100.0),
        ("2026-01-05", 90.0, 92.0, 89.0, 91.0),      # opens far below the stop
    ])
    out = _resolve(bars, {})
    assert out["exit_price"] == 90.0
    assert out["ret_net"] < (96.0 - 100.0) / 100.0


def test_a_signal_that_turns_away_closes_the_trade_at_that_close():
    bars = _bars([
        ("2026-01-02", 100.0, 101.0, 99.5, 100.0),
        ("2026-01-05", 103.0, 104.0, 102.0, 103.5),
    ])
    out = _resolve(bars, {"2026-01-05": "WAIT"})
    assert out["exit_reason"] == "signal"
    assert out["exit_price"] == 103.5
    assert out["ret_net"] == pytest.approx((103.5 - 100.0) / 100.0 - 2 * LEG)


def test_a_zone_never_touched_before_the_signal_turns_is_not_a_trade():
    bars = _bars([
        ("2026-01-02", 110.0, 112.0, 109.0, 111.0),  # never comes back to the zone
        ("2026-01-05", 115.0, 116.0, 114.0, 115.0),
    ])
    out = _resolve(bars, {"2026-01-05": "SELL"})
    assert out["entered"] == 0
    assert out["exit_reason"] == "no_entry"
    assert out["ret_net"] == 0.0


def test_a_trade_still_running_stays_unresolved():
    """A row scored while the position is open would make the result a function
    of when the report was run."""
    bars = _bars([
        ("2026-01-02", 100.0, 101.0, 99.5, 100.0),
        ("2026-01-05", 101.0, 102.0, 100.5, 101.5),
    ])
    assert _resolve(bars, {}) is None


def test_a_short_fills_and_stops_on_the_other_side():
    bars = _bars([
        ("2026-01-02", 100.0, 100.5, 99.0, 100.0),
        ("2026-01-05", 105.0, 106.0, 104.0, 105.5),
    ])
    out = _resolve(bars, {}, signal="SELL", entry=(99.0, 101.0), stop=104.0)
    assert out["entered"] == 1 and out["exit_reason"] == "stop"
    assert out["entry_price"] == 100.0        # max(open, entry_low)
    assert out["exit_price"] == 105.0         # gapped past the stop
    assert out["ret_net"] == pytest.approx(-(105.0 - 100.0) / 100.0 - 2 * LEG)


def test_a_wait_row_has_no_side_and_is_never_scored():
    bars = _bars([("2026-01-02", 100.0, 101.0, 99.0, 100.0)])
    assert _resolve(bars, {}, signal="WAIT") is None


def test_costs_are_charged_on_both_legs():
    """One leg would quietly halve the cost of every trade in the reward."""
    bars = _bars([
        ("2026-01-02", 100.0, 101.0, 99.5, 100.0),
        ("2026-01-05", 100.0, 101.0, 99.9, 100.0),
    ])
    out = _resolve(bars, {"2026-01-05": "WAIT"})
    assert out["ret_net"] == pytest.approx(-2 * LEG)


# --- the row written on the day ---------------------------------------------

def _db(tmp_path):
    import sqlite3
    db = str(tmp_path / "t.db")
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("CREATE TABLE btc (Date TEXT, open REAL, high REAL, low REAL, "
                "close REAL, volume REAL)")
    cur.execute("INSERT INTO btc VALUES ('2026-06-12',1,1,1,100.0,1)")
    con.commit()
    con.close()
    return db


OK = {"close": 100.0, "atr": 2.0, "entry_low": 99.0, "entry_high": 101.0,
      "stop": 96.0, "trailing": True, "status": "ok"}


def _rows(db):
    """Rows in level_log. A missing table counts as none: a call that stores
    nothing has no reason to create one."""
    import sqlite3
    con = sqlite3.connect(db)
    try:
        return con.execute(
            "SELECT asset, signal, stop, trailing FROM level_log").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def test_the_issued_levels_are_written_once_a_day(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    assert pt.log_levels("BTC", "BUY", date="2026-06-12", row=OK) is True
    assert pt.log_levels("BTC", "BUY", date="2026-06-12", row=OK) is False
    assert _rows(db) == [("BTC", "BUY", 96.0, 1)]


def test_the_stored_stop_is_the_one_that_was_shown(tmp_path, monkeypatch):
    """A trailing stop is what the card displays once a position is open, so the
    outcome has to be measured against that number and not the untrailed one."""
    db = _db(tmp_path)
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    pt.log_levels("BTC", "BUY", date="2026-06-12", row=OK)
    assert _rows(db)[0][3] == 1


def test_a_row_with_no_levels_is_not_stored(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    blank = dict(OK, status="no_signal")
    assert pt.log_levels("BTC", "WAIT", date="2026-06-12", row=blank) is False
    assert _rows(db) == []


def test_a_day_with_no_bar_is_never_logged(tmp_path, monkeypatch):
    """It could never be reconciled, so it would sit pending forever."""
    db = _db(tmp_path)
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    assert pt.log_levels("BTC", "BUY", date="2026-06-13", row=OK) is False
    assert _rows(db) == []


# --- reading the journal back ----------------------------------------------

import sqlite3


def _journal(tmp_path, rows):
    """A level_log with `rows` in it, as (entered, exit_reason, ret_net) triples."""
    path = tmp_path / "j.db"
    con = sqlite3.connect(path)
    pt._ensure_level_table(con.cursor())
    for i, (entered, reason, ret) in enumerate(rows):
        con.execute(
            "INSERT INTO level_log (date, asset, signal, close, atr, entry_low, "
            "entry_high, stop, trailing, entered, exit_reason, ret_net) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-01-%02d" % (i + 1), "SP500", "BUY", 100.0, 2.0, 99.0, 101.0,
             96.0, 0, entered, reason, ret))
    con.commit()
    con.close()
    return path


def test_the_journal_reports_nothing_rather_than_zero_while_it_is_young(tmp_path, monkeypatch):
    """A zero would read as a flat result. Nothing resolved is not a flat result."""
    path = _journal(tmp_path, [(None, None, None)] * 3)
    monkeypatch.setattr(pt, "_conn", lambda: sqlite3.connect(path))
    s = pt.level_summary()
    assert (s["issued"], s["resolved"], s["pending"]) == (3, 0, 3)
    assert s["avg_ret"] is None and s["win_pct"] is None
    assert "no outcome to report" in " ".join(pt.level_summary_lines(s))


def test_an_issue_that_never_filled_is_not_counted_as_a_losing_trade(tmp_path, monkeypatch):
    """A wide entry zone makes no_entry common, and folding those into the return
    would report a strategy that took trades it never took."""
    path = _journal(tmp_path, [
        (1, "stop", -0.05),        # filled, lost
        (1, "signal", +0.03),      # filled, won
        (0, "no_entry", 0.0),      # never reached the zone
        (None, None, None),        # still open
    ])
    monkeypatch.setattr(pt, "_conn", lambda: sqlite3.connect(path))
    s = pt.level_summary()
    assert (s["issued"], s["resolved"], s["pending"]) == (4, 3, 1)
    assert (s["entered"], s["no_entry"]) == (2, 1)
    assert (s["stopped"], s["flipped"]) == (1, 1)
    assert s["avg_ret"] == pytest.approx(-0.01)     # over the two FILLED trades
    assert s["win_pct"] == pytest.approx(50.0)      # not 33.3 over all three
    text = " ".join(pt.level_summary_lines(s))
    assert "-0.0100" in text and "50.0%" in text
