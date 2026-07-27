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

import datetime
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


def test_digest_rows_drops_a_baseline_outside_the_window():
    today = datetime.date(2026, 7, 24)
    hist = [
        # baseline ~60 days back: outside the client's 21-day Today window.
        {"asset": "OLD", "date": "2026-05-25", "signal": "WAIT", "prob": None,
         "timing_label": None},
        {"asset": "OLD", "date": "2026-07-24", "signal": "BUY", "prob": 0.7,
         "timing_label": None},
        # baseline 14 days back: inside the window.
        {"asset": "NEW", "date": "2026-07-10", "signal": "WAIT", "prob": None,
         "timing_label": None},
        {"asset": "NEW", "date": "2026-07-24", "signal": "BUY", "prob": 0.7,
         "timing_label": None},
    ]
    filtered = push_signals._digest_rows(hist, today=today)
    events = digest.build_digest(filtered)
    kinds = {e.asset: e.kind for e in events}
    # OLD's baseline row got filtered out, leaving a single row for OLD -
    # build_digest requires >=2 rows per asset, so no event, matching what
    # the client's 21-day window would also fail to show.
    assert "OLD" not in kinds
    assert kinds["NEW"] == digest.ENTRY_BUY


def test_digest_rows_keeps_the_full_row_at_the_window_boundary():
    today = datetime.date(2026, 7, 24)
    # Exactly 21 days back: the floor date itself must be kept (>=, not >).
    hist = [{"asset": "AAA", "date": "2026-07-03", "signal": "WAIT",
             "prob": None, "timing_label": None}]
    filtered = push_signals._digest_rows(hist, today=today)
    assert filtered == hist


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
    assert push_signals.send_push("https://x.supabase.co", "k", []) == 0


def _fake_creds(monkeypatch, tmp_path):
    creds = tmp_path / "creds.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GTRADE_FCM_CREDS", str(creds))
    calls = []
    monkeypatch.setattr(push_signals.requests, "request",
                        lambda method, url, **kw: calls.append((method, url)))
    return calls


def test_send_push_silent_without_signal_events(monkeypatch, tmp_path):
    calls = _fake_creds(monkeypatch, tmp_path)
    timing_only = [ev("T1", digest.TIMING_CHANGE, "BUY", "BUY", 0.55,
                      None, "policy: entering")]
    # No firebase import, no token RPC: a timing-only diff is not push-worthy.
    assert push_signals.send_push("https://x.supabase.co", "k", timing_only) == 0
    assert push_signals.send_push("https://x.supabase.co", "k", []) == 0
    assert calls == []


def test_send_push_skips_a_repeat_of_the_last_push(monkeypatch, tmp_path):
    calls = _fake_creds(monkeypatch, tmp_path)
    events = [ev("SBER", digest.FLIP, "BUY", "SELL", 0.62)]
    state = tmp_path / "push_state.json"
    monkeypatch.setattr(push_signals, "PUSH_STATE", str(state))
    push_signals._save_push_state(push_signals._event_hash(events),
                                  "2026-07-24", str(state))
    assert push_signals.send_push("https://x.supabase.co", "k", events) == 0
    assert calls == []      # dedupe happens before any network or firebase use


def _fake_firebase(monkeypatch, success_count=1):
    """Stand-in for firebase_admin so send_push runs without creds or network."""
    import sys
    import types

    seen = {}

    class _Resp:
        pass

    _Resp.success_count = success_count
    _Resp.responses = []

    messaging = types.SimpleNamespace(
        Notification=lambda title, body: {"title": title, "body": body},
        MulticastMessage=lambda notification, data, tokens: seen.update(
            notification=notification, data=data, tokens=tokens) or "msg",
        send_each_for_multicast=lambda msg: _Resp(),
    )
    fake = types.ModuleType("firebase_admin")
    fake._apps = ["already-initialised"]     # skips credentials.Certificate
    fake.credentials = types.SimpleNamespace(Certificate=lambda path: path)
    fake.messaging = messaging
    monkeypatch.setitem(sys.modules, "firebase_admin", fake)
    monkeypatch.setitem(sys.modules, "firebase_admin.messaging", messaging)
    return seen


def test_send_push_carries_the_today_deeplink_and_records_state(monkeypatch,
                                                               tmp_path):
    creds = tmp_path / "creds.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GTRADE_FCM_CREDS", str(creds))
    state = tmp_path / "push_state.json"
    monkeypatch.setattr(push_signals, "PUSH_STATE", str(state))

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"token": "t1"}]

    monkeypatch.setattr(push_signals.requests, "request",
                        lambda method, url, **kw: R())
    seen = _fake_firebase(monkeypatch)
    events = [ev("SBER", digest.FLIP, "BUY", "SELL", 0.62)]

    assert push_signals.send_push("https://x.supabase.co", "k", events) == 1
    assert seen["data"] == {"screen": "today"}          # the Today deep-link
    assert seen["tokens"] == ["t1"]
    assert seen["notification"]["title"] == "Atratus: 1 change"
    assert push_signals._load_push_state(str(state))["hash"] == \
        push_signals._event_hash(events)
    # The fingerprint is now on disk, so an identical second run is deduped.
    assert push_signals.send_push("https://x.supabase.co", "k", events) == 0


def test_send_push_notes_when_no_tokens(monkeypatch, tmp_path, capsys):
    creds = tmp_path / "creds.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GTRADE_FCM_CREDS", str(creds))
    _fake_firebase(monkeypatch)

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return []      # allow-list RPC returned no tokens

    monkeypatch.setattr(push_signals.requests, "request",
                        lambda method, url, **kw: R())
    events = [ev("SBER", digest.FLIP, "BUY", "SELL", 0.62)]

    assert push_signals.send_push("https://x.supabase.co", "k", events) == 0
    assert "no allow-listed device tokens" in capsys.readouterr().out


def test_send_push_warns_on_zero_delivery(monkeypatch, tmp_path, capsys):
    creds = tmp_path / "creds.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GTRADE_FCM_CREDS", str(creds))
    state = tmp_path / "push_state.json"
    monkeypatch.setattr(push_signals, "PUSH_STATE", str(state))

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"token": "t1"}, {"token": "t2"}]

    monkeypatch.setattr(push_signals.requests, "request",
                        lambda method, url, **kw: R())
    _fake_firebase(monkeypatch, success_count=0)
    events = [ev("SBER", digest.FLIP, "BUY", "SELL", 0.62)]

    assert push_signals.send_push("https://x.supabase.co", "k", events) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "0 of 2 device" in out
    # A total failure must not save the fingerprint, so the next run retries.
    assert push_signals._load_push_state(str(state)) == {}


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


def test_push_news_deletes_on_id_not_asset(monkeypatch):
    # General-feed rows carry a null asset; an asset-based delete filter would
    # leave them behind to accumulate run after run.
    calls = []

    def fake_send(method, url, **kw):
        calls.append((method, url))
        return None

    monkeypatch.setattr(push_signals, "_send", fake_send)
    rows = [{"id": "a1", "asset": None, "date": "2026-07-26", "title": "t"}]
    push_signals.push_news("https://p.supabase.co", "k", rows)
    methods = [m for m, _ in calls]
    assert methods == ["DELETE", "POST"]
    assert "news?id=not.is.null" in calls[0][1]
    assert "on_conflict=id" in calls[1][1]


def test_push_news_sends_nothing_but_the_delete_when_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(push_signals, "_send",
                        lambda method, url, **kw: calls.append(method))
    push_signals.push_news("https://p.supabase.co", "k", [])
    assert calls == ["DELETE"]


def test_fetch_news_rows_survives_a_failing_general_digest(monkeypatch):
    import news_analyzer

    def boom(*a, **kw):
        raise RuntimeError("rss down")

    monkeypatch.setattr(news_analyzer, "fetch_authority_digest", boom)
    monkeypatch.setattr(news_analyzer, "fetch_news",
                        lambda asset, **kw: [{"title": f"{asset} news",
                                              "link": f"http://x/{asset}",
                                              "published": "", "source": "R",
                                              "weighted_score": 0.2,
                                              "sentiment_label": "POSITIVE"}])
    hist = [{"asset": "BTC", "date": "2026-07-26", "signal": "BUY", "prob": 0.8}]
    rows, items = push_signals.fetch_news_rows(hist,
                                               today=datetime.date(2026, 7, 26))
    # The general feed failed, the per-asset fetch still delivered.
    assert [r["asset"] for r in rows] == ["BTC"]
    assert list(items) == ["BTC"]


def test_fetch_news_rows_skips_one_failing_asset(monkeypatch):
    import news_analyzer

    monkeypatch.setattr(news_analyzer, "fetch_authority_digest", lambda **kw: [])

    def per_asset(asset, **kw):
        if asset == "BAD":
            raise RuntimeError("blocked")
        return [{"title": f"{asset} news", "link": f"http://x/{asset}",
                 "published": "", "source": "R", "weighted_score": 0.2,
                 "sentiment_label": "POSITIVE"}]

    monkeypatch.setattr(news_analyzer, "fetch_news", per_asset)
    hist = [{"asset": "BAD", "date": "2026-07-26", "signal": "BUY", "prob": 0.9},
            {"asset": "OK", "date": "2026-07-26", "signal": "BUY", "prob": 0.8}]
    rows, items = push_signals.fetch_news_rows(hist,
                                               today=datetime.date(2026, 7, 26))
    assert [r["asset"] for r in rows] == ["OK"]
    assert list(items) == ["OK"]


def test_fetch_news_context_groups_bars_by_asset():
    # 25 closes: enough returns for a sigma, and every date stays a real July
    # day so the published timestamp below can match the last bar.
    bars = []
    for asset, closes in (("A", [100.0] * 24 + [130.0]), ("B", [50.0] * 25)):
        for i, c in enumerate(closes):
            bars.append({"asset": asset, "date": f"2026-07-{i + 1:02d}",
                         "close": c})
    items = {"A": [{"title": "t", "published": "25 Jul 2026 09:00:00 GMT",
                    "weighted_score": 0.6}]}
    rows = push_signals.fetch_news_context(bars, items)
    by_asset = {r["asset"]: r for r in rows}
    assert set(by_asset) == {"A", "B"}
    assert by_asset["A"]["notable"] is True
    assert by_asset["B"]["notable"] is False
    # Proves the per-asset items actually reached context_row, not just the bars.
    assert by_asset["A"]["consistency"] == "consistent"
    assert by_asset["B"]["consistency"] == "no_news"


def test_fetch_news_context_skips_an_asset_without_bars():
    rows = push_signals.fetch_news_context([], {"A": []})
    assert rows == []


def test_push_news_context_deletes_on_asset(monkeypatch):
    # asset is this table's primary key and never null, unlike in `news`.
    calls = []
    monkeypatch.setattr(push_signals, "_send",
                        lambda method, url, **kw: calls.append((method, url)))
    push_signals.push_news_context("https://p.supabase.co", "k",
                                   [{"asset": "A", "date": "2026-07-26"}])
    assert [m for m, _ in calls] == ["DELETE", "POST"]
    assert "news_context?asset=not.is.null" in calls[0][1]
    assert "on_conflict=asset" in calls[1][1]


from core import events as core_events


def fetch_event_rows_today(module, assets):
    """fetch_event_rows with a pinned today, so the horizon tests are stable."""
    return module.fetch_event_rows(assets, today=datetime.date(2026, 7, 26))


def test_fetch_event_rows_merges_earnings_and_macro(monkeypatch, tmp_path):
    macro = tmp_path / "macro.json"
    macro.write_text('[{"date": "2026-07-29", "name": "FOMC",'
                     ' "importance": "high"}]', encoding="utf-8")
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP", {"AAPL": "AAPL"},
                        raising=False)
    monkeypatch.setattr(core_events, "MACRO_PATH", str(macro))
    monkeypatch.setattr(core_events, "_yf_calendar",
                        lambda symbol, session: {
                            "Earnings Date": ["2026-08-05", "2026-08-09"]})
    monkeypatch.setattr(push_signals, "yf_session", lambda: None,
                        raising=False)
    rows = fetch_event_rows_today(push_signals, ["AAPL"])
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["earnings", "macro"]
    earn = next(r for r in rows if r["kind"] == "earnings")
    assert earn["confirmed"] is False


def test_fetch_event_rows_drops_events_past_the_horizon(monkeypatch, tmp_path):
    macro = tmp_path / "macro.json"
    # Well beyond the 14-day horizon from the pinned today below.
    macro.write_text('[{"date": "2026-12-01", "name": "far"}]',
                     encoding="utf-8")
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP", {}, raising=False)
    monkeypatch.setattr(core_events, "MACRO_PATH", str(macro))
    monkeypatch.setattr(push_signals, "yf_session", lambda: None,
                        raising=False)
    assert fetch_event_rows_today(push_signals, []) == []


def test_fetch_event_rows_survives_a_dead_session(monkeypatch, tmp_path):
    macro = tmp_path / "macro.json"
    macro.write_text('[{"date": "2026-07-29", "name": "FOMC"}]',
                     encoding="utf-8")
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP", {"AAPL": "AAPL"},
                        raising=False)
    monkeypatch.setattr(core_events, "MACRO_PATH", str(macro))

    def boom():
        raise RuntimeError("no route")

    monkeypatch.setattr(push_signals, "yf_session", boom, raising=False)
    # Macro still lands: the two sources are independent.
    rows = fetch_event_rows_today(push_signals, ["AAPL"])
    assert [r["kind"] for r in rows] == ["macro"]


def test_push_events_deletes_on_id(monkeypatch):
    # Macro rows carry a null asset, so an asset filter would orphan them.
    calls = []
    monkeypatch.setattr(push_signals, "_send",
                        lambda method, url, **kw: calls.append((method, url)))
    push_signals.push_events("https://p.supabase.co", "k",
                             [{"id": "e1", "kind": "macro"}])
    assert [m for m, _ in calls] == ["DELETE", "POST"]
    assert "events?id=not.is.null" in calls[0][1]
    assert "on_conflict=id" in calls[1][1]
