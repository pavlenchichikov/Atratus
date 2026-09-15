"""The reach journal: what the card quoted, and whether the session touched it.

No network and no real database: every test points performance_tracker at a
temp market.db and feeds quotes in directly, the way test_performance_tracker
does for the other two journals.
"""

import sqlite3

import pytest


@pytest.fixture
def pt(tmp_path, monkeypatch):
    import performance_tracker as module
    db = str(tmp_path / "market.db")
    monkeypatch.setattr(module, "DB_PATH", db)
    monkeypatch.setattr(module, "_ENGINE", None)
    return module


def _quote(session="2026-09-11", side="upper", level=100.0, p=0.65):
    return {"session": session, "side": side, "level": level, "k": 0.5,
            "dist": 0.4, "probability": p, "cutoff": "2025-10-13"}


def test_a_quote_is_stored_with_no_outcome_yet(pt):
    assert pt.log_intraday_reach("AAPL", [_quote()]) == 1
    with sqlite3.connect(pt.DB_PATH) as con:
        row = con.execute(
            "SELECT session, asset, side, level, probability, reached "
            "FROM intraday_reach_log").fetchone()
    assert row[:3] == ("2026-09-11", "AAPL", "upper")
    assert row[5] is None, "a fresh quote must not carry an outcome"


def test_a_second_call_does_not_restate_the_quote(pt):
    """The odds are only true at the session's first bar. A later call in the
    same session would quote from the middle of it and overwrite the row the
    calibration actually applies to."""
    pt.log_intraday_reach("AAPL", [_quote(p=0.65)])
    assert pt.log_intraday_reach("AAPL", [_quote(p=0.20)]) == 0
    with sqlite3.connect(pt.DB_PATH) as con:
        p = con.execute("SELECT probability FROM intraday_reach_log").fetchone()[0]
    assert p == 0.65, "the first-bar quote was overwritten"


def test_both_sides_are_separate_rows(pt):
    n = pt.log_intraday_reach("AAPL", [_quote(side="upper"),
                                       _quote(side="lower", level=90.0)])
    assert n == 2
    with sqlite3.connect(pt.DB_PATH) as con:
        sides = sorted(r[0] for r in con.execute(
            "SELECT side FROM intraday_reach_log"))
    assert sides == ["lower", "upper"]


def _bars(sessions, hi, lo):
    """Hourly bars for `sessions` days, each with the given high and low."""
    import pandas as pd
    out = []
    day0 = pd.Timestamp("2026-09-11", tz="UTC")
    for d in range(sessions):
        t0 = day0 + pd.Timedelta(days=d, hours=13)
        for h in range(6):
            out.append({"ts": (t0 + pd.Timedelta(hours=h)).isoformat(),
                        "open": 100.0, "high": hi, "low": lo,
                        "close": 100.0, "volume": 10.0})
    return out


def _point_store(pt, monkeypatch, bars, tz="UTC"):
    import intraday_fetch
    monkeypatch.setattr(intraday_fetch, "load", lambda a, db_path=None: bars)
    monkeypatch.setattr(intraday_fetch, "load_tz", lambda a, db_path=None: tz)


def test_a_running_session_is_left_pending_not_scored_as_a_miss(pt, monkeypatch):
    """The trap this guard exists for. Judging the session still in progress
    records every untouched level as a miss, and the measured calibration then
    reads far below the truth for reasons that are purely bookkeeping."""
    _point_store(pt, monkeypatch, _bars(1, hi=101.0, lo=99.0))
    pt.log_intraday_reach("AAPL", [_quote(session="2026-09-11", level=200.0)])
    res = pt.update_intraday_reach()
    assert res["scored"] == 0 and res["pending"] == 1
    with sqlite3.connect(pt.DB_PATH) as con:
        assert con.execute(
            "SELECT reached FROM intraday_reach_log").fetchone()[0] is None


def test_a_closed_session_is_scored_on_its_own_high_and_low(pt, monkeypatch):
    # two sessions, so the first one has a later one behind it and counts closed
    _point_store(pt, monkeypatch, _bars(2, hi=105.0, lo=95.0))
    pt.log_intraday_reach("AAPL", [_quote(session="2026-09-11", side="upper",
                                          level=104.0),
                                   _quote(session="2026-09-11", side="lower",
                                          level=90.0)])
    res = pt.update_intraday_reach()
    assert res["scored"] == 2
    with sqlite3.connect(pt.DB_PATH) as con:
        got = dict(con.execute(
            "SELECT side, reached FROM intraday_reach_log").fetchall())
    assert got["upper"] == 1, "high 105 should have touched 104"
    assert got["lower"] == 0, "low 95 never reached 90"


def test_scoring_is_idempotent(pt, monkeypatch):
    _point_store(pt, monkeypatch, _bars(2, hi=105.0, lo=95.0))
    pt.log_intraday_reach("AAPL", [_quote(session="2026-09-11", level=104.0)])
    first = pt.update_intraday_reach()
    second = pt.update_intraday_reach()
    assert first["scored"] == 1
    assert second["scored"] == 0, "a scored row was scored again"


def test_the_summary_reports_calibration_not_a_hit_rate_alone(pt, monkeypatch):
    """A reach quote has no side to be right or wrong about, so the number that
    matters is whether 65% means 65%. Brier and the bands say that; a bare hit
    rate does not."""
    _point_store(pt, monkeypatch, _bars(2, hi=105.0, lo=95.0))
    pt.log_intraday_reach("AAPL", [_quote(session="2026-09-11", side="upper",
                                          level=104.0, p=0.9),
                                   _quote(session="2026-09-11", side="lower",
                                          level=90.0, p=0.1)])
    pt.update_intraday_reach()
    s = pt.intraday_reach_summary()
    assert s["scored"] == 2 and s["quoted"] == 2
    assert s["brier"] == pytest.approx(0.01, abs=1e-9)
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["by_side"]["upper"]["realised"] == 1.0
    assert s["by_side"]["lower"]["realised"] == 0.0


def test_an_empty_journal_says_so_instead_of_reporting_zero(pt):
    s = pt.intraday_reach_summary()
    assert s["quoted"] == 0 and s["brier"] is None
    assert "No reach quotes stored yet" in " ".join(pt.intraday_reach_lines(s))
