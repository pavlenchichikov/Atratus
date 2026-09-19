"""The intraday question: parse, store, backfill and score, all offline.

No test here reaches a network or a provider: the dossier's context sources and
the provider are monkeypatched exactly as in test_analyst_cli.py.
"""

import json
import sqlite3

import analyst
from core.analyst import agent, intraday, store

SESSION_REPLY = {
    "direction": "up", "gap": "down", "conviction": 3, "vol_regime": "elevated",
    "stand_aside": True, "stand_aside_reason": "CBR decision inside the session.",
    "key_risk": "a hawkish surprise", "thesis": "Wide day, fade the open.",
    "evidence": ["close", "atr_pct"]}


def test_a_session_reply_must_carry_the_gap_and_stand_aside():
    why = []
    no_gap = {k: v for k, v in SESSION_REPLY.items() if k != "gap"}
    assert agent.parse_judgment(json.dumps(no_gap), session=True, why=why) is None
    assert "gap" in why[0]
    why = []
    not_bool = {**SESSION_REPLY, "stand_aside": "yes"}
    assert agent.parse_judgment(json.dumps(not_bool), session=True, why=why) is None
    assert "stand_aside" in why[0]

    j = agent.parse_judgment(json.dumps(SESSION_REPLY), session=True)
    assert j["gap"] == "down" and j["stand_aside"] is True
    assert j["stand_aside_reason"].startswith("CBR")


def test_the_daily_question_is_untouched():
    j = agent.parse_judgment(json.dumps(SESSION_REPLY))
    assert "gap" not in j and "stand_aside" not in j
    prompt = agent.prompt_for({"close": 1.0}, horizon=1)
    assert "INTRADAY" not in prompt and "the next trading day" in prompt
    session = agent.prompt_for({"close": 1.0}, session=True)
    assert "INTRADAY" in session and "from its open to its close" in session


def _db(tmp_path, n=80, start="2026-01-01"):
    """SBER and SP500 with n weekday bars from `start`; SBER's ranges alternate
    so its history has three distinct range classes. Weekdays only, because
    the analyst drops Saturday and Sunday for every non-crypto asset."""
    import datetime as dt

    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    for table in ("sber", "sp500"):
        con.execute('CREATE TABLE %s (Date TEXT, Open REAL, Close REAL, '
                    'High REAL, Low REAL)' % table)
    days = (dt.date.fromisoformat(start) + dt.timedelta(days=k) for k in range(2 * n))
    for i, day in enumerate([d.isoformat() for d in days if d.weekday() < 5][:n]):
        width = (0.5, 1.0, 2.0)[i % 3]
        con.execute("INSERT INTO sber VALUES (?,?,?,?,?)",
                    (day, 100.0, 100.0 + (0.5 if i % 2 else -0.5),
                     100.0 + width, 100.0 - width))
        con.execute("INSERT INTO sp500 VALUES (?,?,?,?,?)",
                    (day, 5000.0, 5000.0 + i, 5010.0 + i, 4990.0))
    con.commit()
    con.close()
    return path


def _offline(monkeypatch, db):
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr("core.track_record.DB_PATH", db)
    monkeypatch.setattr("core.dashboard.guru_for_asset",
                        lambda asset, db_path=None: None)
    monkeypatch.setattr("core.events.earnings_for",
                        lambda symbols_by_asset, session=None, fetch=None: {})
    monkeypatch.setattr("core.events.load_macro", lambda path=None: [])


def test_the_us_lead_is_read_only_from_the_same_date(monkeypatch, tmp_path):
    db = _db(tmp_path, n=10)
    _offline(monkeypatch, db)
    # Monday 01-05 reads its return from Friday 01-02, the bar before it.
    got = intraday.us_last_session_ret("2026-01-05", db_path=db)
    assert abs(got - (5002.0 - 5001.0) / 5001.0) < 1e-12
    assert intraday.us_last_session_ret("2026-02-01", db_path=db) is None


def test_a_run_writes_a_row_that_waits_for_a_finished_session(monkeypatch, tmp_path):
    db = _db(tmp_path, n=28)          # last bar Monday 2026-02-09
    _offline(monkeypatch, db)
    monkeypatch.setattr(analyst, "_provider_call",
                        lambda: (lambda prompt: json.dumps(SESSION_REPLY)))
    assert analyst.main(["intraday", "--assets", "SBER"]) == 0
    row = intraday.scored_rows(db) or None
    assert row is None and intraday.pending_count(db) == 1

    # The next bar exists, but it is dated today: possibly still trading.
    con = sqlite3.connect(db)
    con.execute("INSERT INTO sber VALUES ('2026-02-10', 100, 101, 102, 99)")
    con.commit()
    con.close()
    assert intraday.backfill(db, today="2026-02-10") == 0
    assert intraday.backfill(db, today="2026-02-11") == 1
    r = intraday.scored_rows(db)[0]
    assert (r["date"], r["session_date"]) == ("2026-02-09", "2026-02-10")
    assert r["stand_aside"] == 1 and r["gap"] == "down"


def _judged(db, date, session, direction="up", vol="normal", stand=0,
            calendar=None, o=100.0, c=101.0, h=102.0, low=99.0):
    intraday.write({"date": date, "asset": "SBER", "direction": direction,
                    "gap": "up", "conviction": 3, "vol_regime": vol,
                    "stand_aside": stand, "close_at_signal": 100.0,
                    "calendar_json": json.dumps(calendar or {})}, db)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE analyst_intraday_log SET session_date=?, "
                    "session_open=?, session_high=?, session_low=?, "
                    "session_close=? WHERE date=?", (session, o, h, low, c, date))


def test_score_counts_each_question_and_holds_under_the_floor(monkeypatch, tmp_path):
    db = _db(tmp_path, n=120, start="2025-12-01")   # 60+ sessions before 03-11
    _offline(monkeypatch, db)
    monkeypatch.setattr(intraday, "_lead_rule", lambda rows, db_path: {})
    # Right, wrong, right on direction; a wide stand-aside day; an event day.
    _judged(db, "2026-03-10", "2026-03-11", "up", c=101.0)
    _judged(db, "2026-03-11", "2026-03-12", "down", c=101.0)
    _judged(db, "2026-03-12", "2026-03-13", "down", c=99.0, vol="elevated",
            stand=1, h=104.0, low=96.0,
            calendar={"macro_events": [{"name": "CBR", "date": "2026-03-13"}]})

    s = intraday.score(db)
    assert s["direction"]["n"] == 3 and s["direction"]["hit_rate"] == round(2 / 3, 4)
    assert s["range"]["n"] == 3
    a = s["stand_aside"]
    assert a["n_on"] == 1 and a["n_off"] == 2
    assert a["surprise_on"] > a["surprise_off"]
    assert a["calendar"]["n_on"] == 1, "the event is matched on the SESSION date"

    v = intraday.verdicts(s)
    assert all(e["verdict"] == "HOLD" for e in v.values())
    assert "100 scored sessions" in v["direction"]["missing"]


def test_the_paired_test_only_counts_rows_where_exactly_one_was_right():
    got = intraday._paired([True, True, False, True], [True, False, True, False])
    assert (got["wins"], got["losses"]) == (2, 1)


def test_a_ship_needs_every_condition(monkeypatch):
    s = {"direction": {"n": 150, "p_vs_coin": 0.01},
         "gap": {"n": 150, "p": 0.2},
         "range": {"n": 50, "p": 0.01},
         "stand_aside": {"n_on": 30, "n_off": 120, "surprise_on": 1.6,
                         "surprise_off": 0.9, "p": 0.001,
                         "calendar": {"surprise_on": 1.2, "surprise_off": 1.0}}}
    v = intraday.verdicts(s)
    assert v["direction"]["verdict"] == "SHIP"
    assert v["gap"]["verdict"] == "HOLD"
    assert v["range"]["missing"] == ["100 scored sessions"]
    assert v["stand_aside"]["verdict"] == "SHIP"
