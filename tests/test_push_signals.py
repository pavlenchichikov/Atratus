"""Unit tests for push_signals.build_payload (pure, no network / no DB)."""

from core import digest, timing_policy
from push_signals import _rest_base, build_payload


def test_rest_base_tolerates_pasted_path():
    want = "https://x.supabase.co/rest/v1"
    assert _rest_base("https://x.supabase.co") == want
    assert _rest_base("https://x.supabase.co/") == want
    assert _rest_base("https://x.supabase.co/rest/v1") == want
    assert _rest_base("https://x.supabase.co/rest/v1/") == want


def _sig(asset, signal, prob, acc=None, date="2026-07-12"):
    return {
        "asset": asset,
        "date": date,
        "signal": signal,
        "probability": prob,
        "acc": {"n": 10, "correct": 6, "acc": acc},
    }


def test_counts_and_breadth():
    sigs = [
        _sig("BTC", "BUY", 0.62, 0.6),
        _sig("ETH", "WAIT", 0.51, None),
        _sig("SBER", "SELL", 0.38, 0.5),
    ]
    rows, stats = build_payload(sigs)
    assert len(rows) == 3
    assert (stats["n_buy"], stats["n_sell"], stats["n_wait"]) == (1, 1, 1)
    # breadth = actionable / total = 2/3
    assert abs(stats["breadth"] - 2 / 3) < 1e-9
    # accuracy = mean of present accs (0.6, 0.5)
    assert abs(stats["accuracy"] - 0.55) < 1e-9
    assert stats["snapshot_date"] == "2026-07-12"


def test_row_mapping_and_missing_action():
    rows, stats = build_payload([{"asset": "GOLD", "date": "2026-07-11", "probability": 0.58}])
    row = rows[0]
    assert row["asset"] == "GOLD"
    assert row["action"] == "WAIT"        # missing signal - WAIT
    assert row["prob"] == 0.58
    assert row["mode"] is None and row["taleb"] is None
    assert stats["n_wait"] == 1
    assert stats["accuracy"] is None       # no accuracies present


def test_empty_snapshot_uses_today():
    rows, stats = build_payload([])
    assert rows == []
    assert stats["n_buy"] == stats["n_sell"] == stats["n_wait"] == 0
    assert stats["breadth"] == 0.0
    assert stats["snapshot_date"]          # today's date, non-empty

import sqlite3

import push_signals


def ev(asset, kind, from_signal="WAIT", to_signal="BUY", conf=None,
       from_timing=None, to_timing=None, date="2026-07-24"):
    return digest.DigestEvent(asset=asset, kind=kind, from_signal=from_signal,
                              to_signal=to_signal, from_timing=from_timing,
                              to_timing=to_timing, confidence=conf, date=date)


def _seed_history_db(tmp_path):
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE btc (Date TEXT, open REAL, close REAL, high REAL,'
                ' low REAL, volume REAL)')
    for i in range(200):
        d = f"2026-{i:03d}"  # synthetic ascending dates, 8 chars (str[:10]-safe)
        con.execute("INSERT INTO btc VALUES (?, 1, 2, 3, 0.5, 9)", (d,))
    con.execute('CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,'
                ' probability REAL, actual_next_ret REAL, correct INTEGER,'
                ' timing_action TEXT, timing_reason TEXT)')
    for i in range(120):
        con.execute("INSERT INTO prediction_log VALUES "
                    "(?, 'BTC', 'BUY', 0.6, 0.01, 1, 'STAY_OUT', 'confirm')",
                    (f"2026-{i:03d}",))
    con.commit()
    con.close()
    return db


def test_fetch_history_rows_limits_and_order(tmp_path, monkeypatch):
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP", {"BTC": "BTC-USD"},
                        raising=False)
    db = _seed_history_db(tmp_path)
    bars, hist = push_signals.fetch_history_rows(db_path=db, bar_limit=180,
                                                 sig_limit=90)
    assert len(bars) == 180 and len(hist) == 90
    # ascending after the DESC-limit read
    assert bars[0]["date"] < bars[-1]["date"]
    assert bars[-1]["date"] == "2026-199"
    assert set(bars[0]) == {"asset", "date", "open", "high", "low", "close"}
    assert set(hist[0]) == {"asset", "date", "signal", "prob", "actual_next_ret",
                            "correct", "timing_action", "timing_label"}


def test_fetch_history_rows_skips_missing_table(tmp_path, monkeypatch):
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP",
                        {"BTC": "BTC-USD", "NEWCO": "NEW"}, raising=False)
    db = _seed_history_db(tmp_path)
    bars, hist = push_signals.fetch_history_rows(db_path=db)
    assert {b["asset"] for b in bars} == {"BTC"}  # NEWCO has no table - skipped


def test_fetch_history_rows_timing_label_on(tmp_path, monkeypatch):
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP", {"BTC": "BTC-USD"},
                        raising=False)
    monkeypatch.setattr(timing_policy, "timing_on", lambda: True)
    monkeypatch.setattr(timing_policy, "load_policy", lambda path=None: object())
    db = _seed_history_db(tmp_path)
    _bars, hist = push_signals.fetch_history_rows(db_path=db)
    assert hist[0]["timing_action"] == "STAY_OUT"
    assert hist[0]["timing_label"] == "policy: waiting for confirmation"


def test_fetch_history_rows_timing_none_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP", {"BTC": "BTC-USD"},
                        raising=False)
    monkeypatch.setattr(timing_policy, "timing_on", lambda: False)
    db = _seed_history_db(tmp_path)
    _bars, hist = push_signals.fetch_history_rows(db_path=db)
    assert hist[0]["timing_action"] is None
    assert hist[0]["timing_label"] is None


def test_push_history_full_refresh(monkeypatch):
    calls = []

    class R:
        status_code = 200
        def raise_for_status(self):
            pass

    # All Supabase traffic now flows through requests.request (proxy + retry).
    monkeypatch.setattr(push_signals.requests, "request",
                        lambda method, url, **kw:
                        calls.append((method, url)) or R())
    bars = [{"asset": "BTC", "date": "2026-07-13"}] * (push_signals.CHUNK * 3)
    push_signals.push_history("https://x.supabase.co", "k", bars, [], [], None)
    deletes = [u for m, u in calls if m == "DELETE"]
    posts = [u for m, u in calls if m == "POST"]
    assert sum("/bars" in u for u in deletes) == 1
    assert sum("/bars" in u for u in posts) == 3          # CHUNK*3 rows / CHUNK
    assert sum("/signal_history" in u for u in deletes) == 1
    assert sum("/guru?" in u or u.endswith("/guru") for u in deletes) == 1
    assert not any("/guru_stats" in u for u in posts)      # guru_stats None - skipped


def test_fetch_guru_rows_maps_dashboard_payload(monkeypatch):
    from core import dashboard

    monkeypatch.setattr(dashboard, "guru_latest", lambda: [{
        "asset": "AAPL", "date": "2026-07-12", "verdict": "BUY", "pct": 75.0,
        "lynch": 2, "buffett": 2, "graham": 1, "munger": 1,
        "source": "yf", "correct_5d": 1,
    }])
    monkeypatch.setattr(dashboard, "guru_accuracy", lambda: {
        "council": {"accuracy": 0.61, "total": 40, "correct": 24,
                    "avg_return": 0.01, "by_verdict": {}, "horizon": "60d"},
        "individual": {},
    })
    rows, stats = push_signals.fetch_guru_rows()
    assert rows == [{"asset": "AAPL", "verdict": "BUY", "council_pct": 75.0,
                     "lynch": 2, "buffett": 2, "graham": 1, "munger": 1,
                     "source": "yf", "date": "2026-07-12", "correct_5d": 1}]
    assert stats == {"id": 1, "accuracy": 0.61, "n": 40, "horizon": "60d"}




def test_build_push_text_names_signal_events():
    events = [
        ev("SBER", digest.FLIP, "BUY", "SELL", 0.62),
        ev("GAZP", digest.ENTRY_BUY, "WAIT", "BUY", 0.61),
        ev("DAX", digest.EXIT, "BUY", "WAIT", None),
    ]
    title, body = push_signals.build_push_text(events)
    assert title == "Atratus: 3 changes"
    assert body == "SBER BUY>SELL | GAZP BUY 61% | DAX exit"


def test_build_push_text_singular_title():
    title, _body = push_signals.build_push_text([ev("GAZP", digest.ENTRY_BUY,
                                                    conf=0.6)])
    assert title == "Atratus: 1 change"


def test_build_push_text_truncates_and_counts_the_rest():
    events = [ev(f"A{i}", digest.ENTRY_BUY, conf=0.9 - i / 100) for i in range(6)]
    title, body = push_signals.build_push_text(events)
    assert title == "Atratus: 6 changes"
    assert body == "A0 BUY 90% | A1 BUY 89% | A2 BUY 88% | A3 BUY 87% | +2 more"


def test_build_push_text_appends_timing_counter():
    events = [ev("SBER", digest.FLIP, "BUY", "SELL", 0.62)]
    events += [ev(f"T{i}", digest.TIMING_CHANGE, "BUY", "BUY", 0.55,
                  None, "policy: entering") for i in range(7)]
    title, body = push_signals.build_push_text(events)
    assert title == "Atratus: 8 changes"
    assert body == "SBER BUY>SELL | +7 timing"


def test_build_push_text_entry_without_confidence_omits_percent():
    title, body = push_signals.build_push_text([ev("GOLD", digest.ENTRY_SELL,
                                                   "WAIT", "SELL", None)])
    assert (title, body) == ("Atratus: 1 change", "GOLD SELL")


def test_build_push_text_is_ascii_only():
    events = [ev("SBER", digest.FLIP, "BUY", "SELL", 0.62),
              ev("T1", digest.TIMING_CHANGE, "BUY", "BUY", 0.55)]
    title, body = push_signals.build_push_text(events)
    # Guards the repo convention: no arrows, dashes or smart quotes in copy.
    (title + body).encode("ascii")
    assert "->" not in body


def test_send_push_noop_without_creds(monkeypatch):
    monkeypatch.delenv("GTRADE_FCM_CREDS", raising=False)
    # must not touch the network or firebase at all
    assert push_signals.send_push("https://x.supabase.co", "k", [], {}) == 0


def test_build_payload_includes_timing_label_on_divergence(monkeypatch):
    monkeypatch.setattr(timing_policy, "timing_on", lambda: True)
    monkeypatch.setattr(timing_policy, "load_policy", lambda path=None: object())
    sigs = [{"asset": "BTC", "signal": "BUY", "probability": 0.6,
             "acc": {"acc": None}, "date": "2026-07-23",
             "timing_action": "STAY_OUT", "timing_reason": "confirm"}]
    rows, _stats = build_payload(sigs)
    row = rows[0]
    assert row["timing_action"] == "STAY_OUT"
    assert row["timing_reason"] == "confirm"
    assert row["timing_label"] == "policy: waiting for confirmation"


def test_build_payload_timing_label_none_when_aligned(monkeypatch):
    monkeypatch.setattr(timing_policy, "timing_on", lambda: True)
    monkeypatch.setattr(timing_policy, "load_policy", lambda path=None: object())
    sigs = [{"asset": "BTC", "signal": "BUY", "probability": 0.6,
             "acc": {"acc": None}, "date": "2026-07-23",
             "timing_action": "HOLD", "timing_reason": "ok"}]
    rows, _stats = build_payload(sigs)
    assert rows[0]["timing_label"] is None


def test_build_payload_timing_none_when_absent():
    sigs = [{"asset": "BTC", "signal": "BUY", "probability": 0.6,
             "acc": {"acc": None}, "date": "2026-07-23"}]
    rows, _stats = build_payload(sigs)
    assert rows[0]["timing_action"] is None
    assert rows[0]["timing_label"] is None


def test_build_payload_no_timing_when_flag_off(monkeypatch):
    monkeypatch.setattr(timing_policy, "timing_on", lambda: False)
    sigs = [{"asset": "BTC", "signal": "BUY", "probability": 0.6,
             "acc": {"acc": None}, "date": "2026-07-23",
             "timing_action": "STAY_OUT", "timing_reason": "confirm"}]
    rows, _stats = build_payload(sigs)
    row = rows[0]
    assert row["timing_action"] is None
    assert row["timing_reason"] is None
    assert row["timing_label"] is None


def test_event_hash_ignores_order_but_not_content():
    a = ev("AAA", digest.ENTRY_BUY, conf=0.7)
    b = ev("BBB", digest.EXIT, "BUY", "WAIT")
    assert push_signals._event_hash([a, b]) == push_signals._event_hash([b, a])
    other = ev("AAA", digest.ENTRY_SELL, "WAIT", "SELL", 0.7)
    assert push_signals._event_hash([a]) != push_signals._event_hash([other])


def test_event_hash_includes_the_date():
    # Same events on a later day must NOT be mistaken for a duplicate.
    today = ev("AAA", digest.ENTRY_BUY, conf=0.7, date="2026-07-24")
    tomorrow = ev("AAA", digest.ENTRY_BUY, conf=0.7, date="2026-07-25")
    assert push_signals._event_hash([today]) != push_signals._event_hash([tomorrow])


def test_event_hash_includes_timing_labels():
    before = ev("AAA", digest.TIMING_CHANGE, "BUY", "BUY", 0.7, None, "policy: entering")
    after = ev("AAA", digest.TIMING_CHANGE, "BUY", "BUY", 0.7, None, "policy: exiting")
    assert push_signals._event_hash([before]) != push_signals._event_hash([after])


def test_push_state_roundtrip(tmp_path):
    path = str(tmp_path / "push_state.json")
    assert push_signals._load_push_state(path) == {}      # missing file
    push_signals._save_push_state("abc", "2026-07-24", path)
    state = push_signals._load_push_state(path)
    assert state["hash"] == "abc"
    assert state["snapshot_date"] == "2026-07-24"
    assert state["sent_at"]


def test_load_push_state_tolerates_garbage(tmp_path):
    path = tmp_path / "push_state.json"
    path.write_text("{not json", encoding="utf-8")
    assert push_signals._load_push_state(str(path)) == {}


def test_save_push_state_never_raises(tmp_path, capsys):
    # A directory where the file should be: open() fails, and that must not
    # break a run whose push already succeeded.
    path = tmp_path / "push_state.json"
    path.mkdir()
    push_signals._save_push_state("abc", "2026-07-24", str(path))
    assert "could not write" in capsys.readouterr().out
