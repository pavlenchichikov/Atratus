"""Tests for data_engine._retry_failed (end-of-pass re-fetch of errored assets)."""
import threading

import data_engine as de
import net


def _counting_fetch(behavior):
    """Build a fetch_fn(name, sym) returning (name, status, bars) that records
    call counts per name. behavior(name, call_n) returns (status, bars)."""
    calls = {}
    lock = threading.Lock()

    def fetch(n, sym):
        with lock:
            calls[n] = calls.get(n, 0) + 1
            k = calls[n]
        st, b = behavior(n, k)
        return n, st, b

    fetch.calls = calls
    return fetch


def test_transient_errors_cleared_on_first_sweep():
    results = [("A", "NEW", 1), ("B", "ERR", 0), ("C", "ERR", 0)]
    fetch = _counting_fetch(lambda n, k: ("NEW", 1))
    de._retry_failed(fetch, {"B": "B", "C": "C"}, results,
                     sweeps=2, workers=4, label="t")
    statuses = {n: st for n, st, _ in results}
    assert statuses == {"A": "NEW", "B": "NEW", "C": "NEW"}
    # one sweep was enough: each failed asset fetched exactly once
    assert fetch.calls == {"B": 1, "C": 1}


def test_persistent_error_retried_each_sweep_then_kept():
    results = [("A", "NEW", 1), ("B", "ERR", 0)]
    fetch = _counting_fetch(lambda n, k: ("ERR", 0))
    de._retry_failed(fetch, {"B": "B"}, results,
                     sweeps=3, workers=2, label="t")
    assert {n: st for n, st, _ in results}["B"] == "ERR"
    assert fetch.calls["B"] == 3  # retried on every sweep


def test_no_errors_means_no_fetch():
    results = [("A", "NEW", 1), ("B", "UP_TO_DATE", 0)]
    fetch = _counting_fetch(lambda n, k: ("NEW", 1))
    de._retry_failed(fetch, {}, results, sweeps=2, workers=2, label="t")
    assert fetch.calls == {}
    assert {n: st for n, st, _ in results} == {"A": "NEW", "B": "UP_TO_DATE"}


def test_partial_recovery_stops_when_clean():
    results = [("B", "ERR", 0), ("C", "ERR", 0)]
    # B clears on its first retry, C never does
    fetch = _counting_fetch(lambda n, k: ("NEW", 2) if n == "B" else ("ERR", 0))
    de._retry_failed(fetch, {"B": "B", "C": "C"}, results,
                     sweeps=3, workers=2, label="t")
    statuses = {n: st for n, st, _ in results}
    assert statuses["B"] == "NEW"
    assert statuses["C"] == "ERR"
    assert fetch.calls["B"] == 1      # B fetched only until it cleared
    assert fetch.calls["C"] == 3      # C retried every sweep


def test_bars_updated_on_recovery():
    results = [("B", "ERR", 0)]
    fetch = _counting_fetch(lambda n, k: ("NEW", 7))
    de._retry_failed(fetch, {"B": "B"}, results, sweeps=1, workers=1, label="t")
    assert results[0] == ("B", "NEW", 7)


# --- startup route-status banner (probes real reachability) ------------------

def _patch_reach(monkeypatch, ok):
    """Make net.http_get (used by the banner probe) succeed or fail."""
    def fake(url, *a, **k):
        if ok:
            return type("R", (), {"status_code": 200})()
        raise ConnectionError("unreachable")
    monkeypatch.setattr(net, "http_get", fake)


def test_route_status_yahoo_reachable(monkeypatch):
    # full-tunnel VPN: no SOCKS5 proxy, but Yahoo reachable on the direct route
    monkeypatch.setattr(net, "is_proxy_alive", lambda *a, **k: False)
    _patch_reach(monkeypatch, True)
    line = de.route_status_line()
    assert "OK" in line and "reachable" in line.lower()
    assert "fail" not in line.lower()  # must NOT cry wolf when it works


def test_route_status_yahoo_unreachable(monkeypatch):
    monkeypatch.setattr(net, "is_proxy_alive", lambda *a, **k: False)
    _patch_reach(monkeypatch, False)
    line = de.route_status_line()
    assert "UNREACHABLE" in line


def test_route_status_reports_proxy_state(monkeypatch):
    monkeypatch.setattr(net, "SOCKS5_PROXY", "socks5h://127.0.0.1:12334")
    monkeypatch.setattr(net, "is_proxy_alive", lambda *a, **k: True)
    _patch_reach(monkeypatch, True)
    assert "proxy up" in de.route_status_line().lower()


# --- MOEX skips the network call when already current ------------------------

def test_moex_skips_fetch_when_already_current(monkeypatch):
    import datetime as _dt
    called = []
    monkeypatch.setattr(net, "http_get", lambda *a, **k: called.append(1))
    # last bar is today: next bar would start in the future, so no network call
    assert de.fetch_moex_smart("SBER", _dt.datetime.now()) is None
    assert called == []
    # weekly fetcher has the same guard
    assert de.fetch_moex_weekly("SBER", _dt.datetime.now()) is None
    assert called == []


def test_moex_fetches_when_stale(monkeypatch):
    import datetime as _dt
    called = []
    def fake(*a, **k):
        called.append(1)
        raise ConnectionError("net hit")
    monkeypatch.setattr(net, "http_get", fake)
    de.fetch_moex_smart("SBER", _dt.datetime.now() - _dt.timedelta(days=7))
    assert called  # stale data: network attempted


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _weekly_payload(dates, closes=None):
    import datetime as dt
    ts = [int(dt.datetime.strptime(d, "%Y-%m-%d").timestamp()) for d in dates]
    n = len(ts)
    closes = closes or [100.0 + i for i in range(n)]
    return {"chart": {"result": [{
        "timestamp": ts,
        "indicators": {"quote": [{"open": closes, "close": closes,
                                  "high": closes, "low": closes,
                                  "volume": [1.0] * n}]}}]}}


def test_the_week_in_progress_is_not_stored(monkeypatch):
    """Yahoo appends a bar for the week IN PROGRESS, stamped with the day of the
    request rather than the week's start. A daily update wrote one such row per
    day: measured 2026-09-08, 665 of the 850 weekly tables carried daily-spaced
    rows dated after 2026-03-04."""
    import datetime as dt
    today = dt.date.today()
    days = [today - dt.timedelta(days=d) for d in (21, 14, 7, 0)]
    payload = _weekly_payload([d.isoformat() for d in days])
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(payload))
    got = de.fetch_yahoo_weekly("AAPL", None)
    assert got is not None
    assert got.index[-1].date() == days[-2], "kept a week that has not finished"
    assert len(got) == 3


def test_a_week_that_finished_is_kept(monkeypatch):
    """Positive control: the rule is the bar's AGE, not "drop the last row"."""
    import datetime as dt
    today = dt.date.today()
    days = [today - dt.timedelta(days=d) for d in (28, 21, 14, 8)]
    payload = _weekly_payload([d.isoformat() for d in days])
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(payload))
    got = de.fetch_yahoo_weekly("AAPL", None)
    assert len(got) == 4 and got.index[-1].date() == days[-1]


def test_an_eight_day_gap_does_not_hide_an_unfinished_week(monkeypatch):
    """The rule this replaced compared the last two stamps and required a gap
    under seven days. aapl_weekly held 08-17, 08-24, 08-31 and then 09-08: an
    8-day gap, so the in-progress bar sailed straight through it."""
    import datetime as dt
    today = dt.date.today()
    days = [today - dt.timedelta(days=d) for d in (22, 15, 8, 0)]
    payload = _weekly_payload([d.isoformat() for d in days])
    monkeypatch.setattr(net, "http_get", lambda url, **kw: _Resp(payload))
    got = de.fetch_yahoo_weekly("AAPL", None)
    assert len(got) == 3 and got.index[-1].date() == days[-2]


def _moex_payload(dates):
    cols = ["open", "close", "high", "low", "value", "volume", "begin", "end"]
    data = [[1.0, 1.0, 1.0, 1.0, 0.0, 5.0, d + " 00:00:00", d + " 23:59:59"]
            for d in dates]
    return {"candles": {"columns": cols, "data": data}}


def test_the_moex_weekly_fetch_also_refuses_the_week_in_progress(monkeypatch):
    """The Yahoo fix left this second copy of the defect in place, and it said so
    plainly: after that fix exactly 181 tables still carried an unfinished week,
    and there are exactly 181 MOEX names."""
    import datetime as dt
    today = dt.date.today()
    days = [(today - dt.timedelta(days=d)).isoformat() for d in (21, 14, 7, 0)]
    monkeypatch.setattr(net, "http_get",
                        lambda url, **kw: _Resp(_moex_payload(days)))
    got = de.fetch_moex_weekly("SBER", None)
    assert got is not None
    assert str(got.index[-1].date()) == days[-2], "kept the week in progress"
    assert len(got) == 3


def test_preflight_repairs_the_weekly_tables_and_says_so(monkeypatch, capsys):
    """The repair runs BEFORE the fetch so the same run refills what it deleted.
    Run afterwards, the tables stay short until tomorrow and a trainer started in
    between reads the hole."""
    import repair_aggregated_bars as agg
    import repair_weekly_tables as wk
    monkeypatch.setattr(de, "AUTO_REPAIR", True)
    monkeypatch.setattr(de, "_training_looks_active", lambda: (False, None))
    monkeypatch.setattr(wk, "cut_plan", lambda db, **kw: [("a_weekly", "2026-01-01", 3)])
    monkeypatch.setattr(wk, "apply_cuts", lambda db, plan: 3)
    monkeypatch.setattr(agg, "one_pass", lambda db, **kw: (0, 0, 0, 0))
    de._preflight_repair(60)
    out = capsys.readouterr().out
    assert "1 table(s) cut, 3 row(s) removed" in out


def test_preflight_says_clean_when_it_is(monkeypatch, capsys):
    """Positive control for the test above: the same code path with nothing to
    do must report nothing, not the same line."""
    import repair_aggregated_bars as agg
    import repair_weekly_tables as wk
    monkeypatch.setattr(de, "AUTO_REPAIR", True)
    monkeypatch.setattr(de, "_training_looks_active", lambda: (False, None))
    monkeypatch.setattr(wk, "cut_plan", lambda db, **kw: [])
    monkeypatch.setattr(agg, "one_pass", lambda db, **kw: (0, 0, 0, 0))
    de._preflight_repair(60)
    out = capsys.readouterr().out
    assert "Weekly  : clean" in out and "row(s) removed" not in out


def test_preflight_repeats_until_the_daily_scan_settles(monkeypatch, capsys):
    """The daily filter is relative to each asset's own median, so removing the
    worst rows exposes the next layer. One pass is never enough: four were
    needed on 2026-09-08."""
    import repair_aggregated_bars as agg
    import repair_weekly_tables as wk
    monkeypatch.setattr(de, "AUTO_REPAIR", True)
    monkeypatch.setattr(de, "_training_looks_active", lambda: (False, None))
    monkeypatch.setattr(wk, "cut_plan", lambda db, **kw: [])
    passes = iter([(5, 1, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0)])
    monkeypatch.setattr(agg, "one_pass", lambda db, **kw: next(passes))
    de._preflight_repair(60)
    out = capsys.readouterr().out
    assert "7 aggregated row(s) rewritten, 1 phantom day(s) dropped, in 3 pass(es)" in out


def test_a_broken_repair_never_blocks_the_data_update(monkeypatch, capsys):
    import repair_weekly_tables as wk
    monkeypatch.setattr(de, "AUTO_REPAIR", True)
    monkeypatch.setattr(de, "_training_looks_active", lambda: (False, None))

    def boom(*a, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(wk, "cut_plan", boom)
    de._preflight_repair(60)          # must not raise
    assert "scan failed" in capsys.readouterr().out


def test_preflight_can_be_switched_off(monkeypatch, capsys):
    monkeypatch.setattr(de, "AUTO_REPAIR", False)
    de._preflight_repair(60)
    assert "GTRADE_AUTO_REPAIR=0" in capsys.readouterr().out


def test_training_activity_is_seen_on_either_marker(tmp_path, monkeypatch):
    """Both markers must count. The first version folded them with an expression
    that kept only the first one it found and silently ignored the rest."""
    import os
    import time
    monkeypatch.setattr(de, "BASE_DIR", str(tmp_path))
    (tmp_path / "_chunk_progress.txt").write_text("old", encoding="utf-8")
    os.utime(tmp_path / "_chunk_progress.txt", (time.time() - 7200,) * 2)
    assert de._training_looks_active()[0] is False

    logs = tmp_path / "_chunk_logs"
    logs.mkdir()
    (logs / "chunk_01.log").write_text("fresh", encoding="utf-8")
    assert de._training_looks_active()[0] is True


def test_the_repair_stands_down_while_training_runs(monkeypatch, capsys):
    """Skipping a repair costs a day. Rewriting the bars a 24-hour training run
    is reading costs the day and the champions."""
    monkeypatch.setattr(de, "AUTO_REPAIR", True)
    monkeypatch.setattr(de, "_training_looks_active", lambda: (True, 3.0))
    de._preflight_repair(60)
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "3 minute(s) ago" in out
    assert "clean" not in out
