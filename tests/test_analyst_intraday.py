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

    assert s["range"]["har_hit_rate"] is not None, "range is paired with HAR"

    v = intraday.verdicts(s)
    assert v["gap"]["verdict"] == "INFO", "the gap is not a scored claim"
    assert all(e["verdict"] == "HOLD" for k, e in v.items() if k != "gap")
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
    assert v["gap"]["verdict"] == "INFO"
    assert v["range"]["missing"] == ["100 scored sessions"]
    s["range"] = {"n": 150, "p": 0.3}
    assert intraday.verdicts(s)["range"]["missing"] == ["better than HAR, p < 0.05"]
    assert v["stand_aside"]["verdict"] == "SHIP"


def test_session_range_view_classes_har_on_the_same_cut_offs(tmp_path):
    db = _db(tmp_path, n=80)
    bars = intraday.ohlc_series("SBER", days=400, db_path=db)
    assert intraday.session_range_view(bars[:59]) is None, "needs 60 sessions"
    v = intraday.session_range_view(bars)
    assert v["calm_below_pct"] < v["usual_pct"] < v["elevated_above_pct"]
    assert v["har_typical_pct"] < v["har_wide_pct"]
    assert v["har_class"] in ("calm", "normal", "elevated")
    last = bars[-1]
    assert v["last_session_pct"] == round(100 * (last["high"] - last["low"]) / last["open"], 3)


def test_the_dossier_carries_session_range(monkeypatch, tmp_path):
    db = _db(tmp_path, n=80)
    _offline(monkeypatch, db)
    d = intraday.build("SBER", db_path=db)
    assert d["session_range"]["har_class"] in ("calm", "normal", "elevated")
    prompt = agent.prompt_for(d, session=True)
    assert "har_class" in prompt and "session_plan" in prompt
    assert "tend to follow it" not in prompt, "no nudge toward the closed lead"


def test_a_session_plan_is_kept_and_shown(monkeypatch, tmp_path, capsys):
    db = _db(tmp_path, n=80)
    _offline(monkeypatch, db)
    reply = {**SESSION_REPLY, "session_plan": "Wait out the first hour."}
    monkeypatch.setattr(analyst, "_provider_call",
                        lambda: (lambda prompt: json.dumps(reply)))
    assert analyst.main(["intraday", "--assets", "SBER"]) == 0
    out = capsys.readouterr().out
    assert "plan:  Wait out the first hour." in out
    assert "range call: analyst wide, HAR" in out and "last close" in out
    assert intraday.recent(1, db)[0]["session_plan"] == "Wait out the first hour."


def test_rewinds_stop_at_the_knowledge_cutoff(monkeypatch, tmp_path, capsys):
    db = _db(tmp_path, n=80)
    _offline(monkeypatch, db)
    monkeypatch.setenv("GTRADE_ANALYST_REWIND_FROM", "2026-04-01")
    assert analyst.main(["intraday", "--assets", "SBER", "--as-of", "2026-03-02"]) == 1
    assert "before 2026-04-01" in capsys.readouterr().out
    monkeypatch.setattr(analyst, "_recent_dates",
                        lambda n: ["2026-03-30", "2026-03-31", "2026-04-01"])
    seen = []
    real = analyst.cmd_intraday

    def one(args):
        if not args.back:
            seen.append(args.as_of)
            return 0
        return real(args)
    monkeypatch.setattr(analyst, "cmd_intraday", one)
    args = analyst.build_parser().parse_args(["intraday", "--back", "3"])
    assert real(args) == 0
    assert seen == ["2026-04-01"]


def test_the_panel_is_the_watchlist_unless_set(monkeypatch, tmp_path):
    wl = tmp_path / "watchlist.json"
    wl.write_text(json.dumps({"default": ["sber", "BTC"]}))
    monkeypatch.setattr(intraday, "WATCHLIST", str(wl))
    monkeypatch.delenv("GTRADE_ANALYST_INTRADAY_PANEL", raising=False)
    assert intraday.panel_assets() == ["SBER", "BTC"]
    monkeypatch.setenv("GTRADE_ANALYST_INTRADAY_PANEL", "gold")
    assert intraday.panel_assets() == ["GOLD"]
    monkeypatch.delenv("GTRADE_ANALYST_INTRADAY_PANEL")
    monkeypatch.setattr(intraday, "WATCHLIST", str(tmp_path / "none.json"))
    assert intraday.panel_assets() == list(intraday.PANEL)


def test_the_range_is_paired_with_har_not_with_always_normal(monkeypatch, tmp_path):
    db = _db(tmp_path, n=120, start="2025-12-01")
    _offline(monkeypatch, db)
    monkeypatch.setattr(intraday, "_lead_rule", lambda rows, db_path: {})
    _judged(db, "2026-03-10", "2026-03-11", vol="normal")
    _judged(db, "2026-03-11", "2026-03-12", vol="elevated", h=104.0, low=96.0)
    # A HAR that is never right: every row the analyst got right is a win.
    monkeypatch.setattr(intraday, "session_range_view", lambda bars: {"har_class": "never"})
    r = intraday.score(db)["range"]
    assert r["har_hit_rate"] == 0.0
    assert (r["wins"], r["losses"]) == (round(r["hit_rate"] * r["n"]), 0)
