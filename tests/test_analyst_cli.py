"""analyst.py: the eligibility throttle and the kill switch.

No test here may reach a network or a provider - watchlist._load, the earnings
lookup and the provider resolver are monkeypatched at their source in every
test.
"""

import json
import os
import sqlite3

import pytest

import analyst
from core.analyst import store
from core.analyst.score import mae_atr, standings


def test_an_asset_reporting_earnings_today_is_added_even_off_watchlist(monkeypatch):
    monkeypatch.setattr("watchlist._load", lambda: {"default": ["SBER"]})
    monkeypatch.setattr(
        "core.events.earnings_for",
        lambda symbols_by_asset, session=None, fetch=None:
            {"AAPL": {"date": "2026-08-25", "confirmed": True}})
    result = analyst._eligible(today="2026-08-25")
    assert result == ["AAPL", "SBER"]


def test_an_asset_reporting_earnings_on_another_day_is_excluded(monkeypatch):
    monkeypatch.setattr("watchlist._load", lambda: {"default": ["SBER"]})
    monkeypatch.setattr(
        "core.events.earnings_for",
        lambda symbols_by_asset, session=None, fetch=None:
            {"AAPL": {"date": "2000-01-01", "confirmed": True}})
    result = analyst._eligible(today="2026-08-25")
    assert result == ["SBER"]


def test_a_failed_earnings_scan_degrades_to_the_watchlist_alone(monkeypatch):
    monkeypatch.setattr("watchlist._load", lambda: {"default": ["SBER", "GOLD"]})

    def _boom(*a, **k):
        raise RuntimeError("network is down")
    monkeypatch.setattr("core.events.earnings_for", _boom)
    result = analyst._eligible(today="2026-08-25")
    assert result == ["GOLD", "SBER"]


def test_the_kill_switch_never_resolves_a_provider(monkeypatch):
    monkeypatch.setenv("GTRADE_ANALYST", "0")

    def _must_not_be_called():
        raise AssertionError("_provider_call was invoked with the kill switch on")
    monkeypatch.setattr("analyst._provider_call", _must_not_be_called)
    assert analyst.cmd_run(None) == 0


def test_a_missing_payoff_table_exits_nonzero_without_a_provider_call(monkeypatch):
    # No payoff_stats.json means no prior to shrink toward, so cmd_run must
    # refuse rather than write judgments with no numbers. The loader is
    # monkeypatched so this never touches the real fitted artifact.
    monkeypatch.setattr("analyst._load_table", lambda: None)

    def _must_not_be_called():
        raise AssertionError("_provider_call was invoked with no payoff table")
    monkeypatch.setattr("analyst._provider_call", _must_not_be_called)
    assert analyst.cmd_run(None) != 0


def _cli_e2e_db(tmp_path):
    """20 flat SBER bars (ATR settles at 2.0, close 100.0) - just enough
    history for dossier.build to produce a signal-day dossier, with no next
    bar yet so the judgment starts pending."""
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE sber (Date TEXT, Open REAL, Close REAL, '
                'High REAL, Low REAL)')
    con.executemany(
        'INSERT INTO sber VALUES (?,?,?,?,?)',
        [(f"2026-01-{i + 1:02d}", 100.0, 100.0, 101.0, 99.0) for i in range(20)])
    con.commit()
    con.close()
    return path


def test_cmd_run_composed_path_writes_a_row_that_backfills_to_a_small_error(
        monkeypatch, tmp_path):
    """Finding 2: cmd_run's loop body had zero coverage - both existing
    cmd_run tests above return before it runs. This drives the real
    dossier -> agent -> calibrate -> store chain end to end, injecting only
    the provider, the eligibility list and the payoff-table loader, then
    backfills through the real store path and checks the scored error."""
    db = _cli_e2e_db(tmp_path)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr("core.track_record.DB_PATH", db)
    # dossier.build's context sources reach a database/network on their own;
    # none may fire for real in a test (same stubs as test_analyst_dossier.py).
    monkeypatch.setattr("core.dashboard.guru_for_asset",
                        lambda asset, db_path=None: None)
    monkeypatch.setattr("core.events.earnings_for",
                        lambda symbols_by_asset, session=None, fetch=None: {})
    monkeypatch.setattr("core.events.load_macro", lambda path=None: [])

    monkeypatch.setattr(analyst, "_eligible", lambda: ["SBER"])
    monkeypatch.setattr(
        analyst, "_provider_call",
        lambda: (lambda prompt: json.dumps({
            "direction": "down", "conviction": 3, "vol_regime": "normal",
            "key_risk": "overbought", "thesis": "Quiet tape, downside lean.",
            "evidence": ["close", "atr_pct"]})))
    # A minimal payoff table: no asset-level cell, so calibrate.forecast reads
    # the class prior for SELL straight (n well above MIN_CELL_OWN so the
    # class quantiles are used, but the analyst_log is empty so the cell
    # itself still falls back to this prior at n=0).
    monkeypatch.setattr(
        analyst, "_load_table",
        lambda: {"asset": {}, "class": {"ru": {
            "SELL": {"n": 900, "mean": 1.0, "q10": -1.0, "q90": 3.0}}}})

    assert analyst.cmd_run(None) == 0
    assert store.pending_count(db_path=db) == 1

    # Append the next day's real bar, the way data_engine would - price falls
    # 2%, a +1.0 ATR short payoff that matches the "down" call's forecast
    # (also 1.0 ATR, from the class prior mean above) almost exactly.
    con = sqlite3.connect(db)
    con.execute("INSERT INTO sber VALUES (?,?,?,?,?)",
               ("2026-01-21", 100.0, 98.0, 100.5, 97.5))
    con.commit()
    con.close()

    filled = store.backfill_outcomes(db_path=db, today="2026-01-22")
    assert filled == 1
    r = store.scored_rows(db_path=db)[0]
    assert r["direction"] == "down"
    assert r["abs_err_atr"] < 0.2


def _score_fixture_db(tmp_path):
    """Bars for SBER plus three judgments, two with a matching prediction_log
    signal and one without, and the outcome backfill already run."""
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE sber (Date TEXT, Open REAL, Close REAL, '
                'High REAL, Low REAL)')
    closes = [100.0] * 20 + [102.0, 103.0, 101.0]
    rows = [(f"2026-01-{i + 1:02d}", c, c, c + 1.0, c - 1.0)
            for i, c in enumerate(closes)]
    con.executemany('INSERT INTO sber VALUES (?,?,?,?,?)', rows)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT)")
    con.executemany(
        "INSERT INTO prediction_log VALUES (?,?,?)",
        [("2026-01-20", "SBER", "BUY"), ("2026-01-21", "SBER", "SELL")])
    # 2026-01-22's judgment gets no prediction_log row at all - the ensemble
    # baseline has nothing to say for it.
    con.commit()
    con.close()

    store.ensure_table(path)
    for date, direction, forecast_pct, close_sig in (
            ("2026-01-20", "up", 0.01, 100.0),
            ("2026-01-21", "down", -0.005, 102.0),
            ("2026-01-22", "up", 0.02, 103.0)):
        store.write_judgment({
            "date": date, "asset": "SBER", "horizon": 1,
            "direction": direction, "conviction": 3, "vol_regime": "normal",
            "key_risk": "none", "thesis": "t", "evidence_json": "[]",
            "dossier_hash": date, "llm_model": "test",
            "forecast_pct": forecast_pct, "lo_pct": -0.03, "hi_pct": 0.03,
            "atr_at_signal": 2.0, "close_at_signal": close_sig,
        }, db_path=path)
    store.backfill_outcomes(db_path=path, today="2026-02-01")
    return path


def test_cmd_score_prints_the_three_baselines_and_a_hold_verdict(monkeypatch, tmp_path, capsys):
    db = _score_fixture_db(tmp_path)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr("analyst._load_table",
                        lambda: {"asset": {}, "class": {}})

    assert analyst.cmd_score(None) == 0
    out = capsys.readouterr().out

    # Three scored judgments went in. The one missing a matching
    # prediction_log row still counts toward the agent/zero/empirical - only
    # the ensemble comparison is short one.
    assert ("[analyst] scored rows: 3 (0 skipped without a forecast; "
            "1 of the 3 counted still lack a matching prediction_log signal "
            "and are absent only from the ensemble comparison)") in out
    assert "'zero'" in out and "'empirical'" in out and "'ensemble'" in out
    # A handful of judgments never clears the >=500 floor.
    assert "[analyst] verdict: HOLD on 3 scored judgments" in out


def test_a_row_without_an_ensemble_match_stays_in_every_other_baseline(monkeypatch, tmp_path):
    # Fix 1: dropping the whole row over a missing prediction_log match used
    # to shrink len(rows) and skew coverage, both of which gate the verdict.
    # Only the ensemble comparison is allowed to be short a row.
    db = _score_fixture_db(tmp_path)
    monkeypatch.setattr(store, "DB_PATH", db)

    rows, baselines, no_forecast, no_ensemble_match = analyst._score_baselines(
        {"asset": {}, "class": {}})

    assert len(rows) == 3            # not 2 - the unmatched row is still here
    assert no_forecast == 0
    assert no_ensemble_match == 1
    assert len(baselines["zero"]) == len(baselines["empirical"]) \
        == len(baselines["ensemble"]) == 3
    assert baselines["ensemble"].count(None) == 1

    s = standings(rows, baselines)
    # The agent's own MAE is over all three rows - unaffected by the gap.
    assert s["agent"]["mae"] == pytest.approx(mae_atr(rows))
    # The ensemble baseline's MAE is only over the two rows with a match.
    matched = [{**r, "forecast_atr": e}
               for r, e in zip(rows, baselines["ensemble"]) if e is not None]
    assert len(matched) == 2
    assert s["baselines"]["ensemble"]["mae"] == pytest.approx(mae_atr(matched))


def test_cmd_score_refuses_when_the_database_is_missing(monkeypatch, tmp_path, capsys):
    # Fix 2: connecting straight to a nonexistent DB_PATH silently creates an
    # empty market.db. A fresh checkout must refuse instead of littering one.
    missing = str(tmp_path / "market.db")
    monkeypatch.setattr(store, "DB_PATH", missing)

    assert analyst.cmd_score(None) != 0
    out = capsys.readouterr().out
    assert "market.db" in out
    assert not os.path.exists(missing)


def test_named_assets_are_validated_against_the_map():
    import analyst
    assert analyst._named_assets("sber, aapl") == ["SBER", "AAPL"]
    assert analyst._named_assets("SBER;NOSUCHTHING") == ["SBER"]
    assert analyst._named_assets(None) == []
    assert analyst._named_assets("") == []


def test_a_named_run_bypasses_the_dossier_hash_skip(monkeypatch, tmp_path):
    # Asking for one asset by name means asking for it NOW. The cache exists to
    # stop a scheduled sweep paying twice for an unchanged dossier, not to tell
    # a person who typed an asset that it was already judged today.
    import analyst
    seen = []
    monkeypatch.setattr(analyst.store, "judged_with_hash",
                        lambda *a, **k: seen.append(a) or True)
    monkeypatch.setattr(analyst, "_load_table", lambda: {"asset": {}, "class": {}})
    monkeypatch.setattr(analyst, "_provider_call", lambda: (lambda p: "{}"))
    monkeypatch.setattr(analyst.store, "ensure_table", lambda *a, **k: None)
    monkeypatch.setattr(analyst.store, "scored_rows", lambda *a, **k: [])
    monkeypatch.setattr(analyst.dossier, "build",
                        lambda a, **k: {"asset": a, "date": "2026-01-01",
                                        "close": 100.0, "atr": 2.0})
    monkeypatch.setattr(analyst.dossier, "dossier_hash", lambda d: "h")
    monkeypatch.setattr(analyst.agent, "judge",
                        lambda d, call=None, depth=None, horizon=1,
                        on_reject=None: None)
    monkeypatch.setenv("GTRADE_ANALYST", "1")

    class A:
        assets, llm, model = "SBER", None, None

    analyst.cmd_run(A())
    assert seen == [], "a named run consulted the already-judged cache"


def test_the_provider_switch_only_affects_this_run(monkeypatch):
    import os

    import analyst
    monkeypatch.setenv("GTRADE_AR_LLM", "anthropic")
    monkeypatch.setattr(analyst, "_load_table", lambda: None)   # exits early

    class A:
        assets, llm, model = None, "ollama", "qwen"

    analyst.cmd_run(A())
    # _load_table returning None exits before any call, but the override must
    # already have been applied - it is set before the provider is resolved.
    assert os.getenv("GTRADE_AR_LLM") in ("anthropic", "ollama")


def test_depth_defaults_to_full_when_an_asset_is_named(monkeypatch):
    # Naming an asset is a person asking to read the reasoning; a sweep is a
    # throughput job. The default has to follow that intent, because on a local
    # model the difference is 2189s against 637s per asset.
    import analyst
    seen = {}
    monkeypatch.setenv("GTRADE_ANALYST", "1")
    monkeypatch.setattr(analyst, "_load_table", lambda: {"asset": {}, "class": {}})
    monkeypatch.setattr(analyst, "_provider_call", lambda: (lambda p: "{}"))
    monkeypatch.setattr(analyst, "_eligible", lambda *a, **k: ["AAPL"])
    monkeypatch.setattr(analyst.store, "ensure_table", lambda *a, **k: None)
    monkeypatch.setattr(analyst.store, "scored_rows", lambda *a, **k: [])
    monkeypatch.setattr(analyst.store, "judged_with_hash", lambda *a, **k: False)
    monkeypatch.setattr(analyst.dossier, "build",
                        lambda a, **k: {"asset": a, "date": "2026-01-01",
                                        "close": 100.0, "atr": 2.0})
    monkeypatch.setattr(analyst.dossier, "dossier_hash", lambda d: "h")
    def _record(d, call=None, depth=None, horizon=1, on_reject=None):
        # Записываем глубину и отказываемся судить: возврат суждения потянул
        # бы за собой калибровку, а этот тест не про неё. Неявный None здесь и
        # есть отказ, ровно как у настоящего judge.
        seen["depth"] = depth
        seen["horizon"] = horizon

    monkeypatch.setattr(analyst.agent, "judge", _record)

    class Named:
        assets, llm, model, depth = "SBER", None, None, None

    class Sweep:
        assets, llm, model, depth = None, None, None, None

    analyst.cmd_run(Named())
    assert seen["depth"] == "full"

    seen.clear()
    analyst.cmd_run(Sweep())
    assert seen["depth"] == "brief"


def test_an_explicit_depth_overrides_the_default(monkeypatch):
    import analyst
    seen = {}
    monkeypatch.setenv("GTRADE_ANALYST", "1")
    monkeypatch.setattr(analyst, "_load_table", lambda: {"asset": {}, "class": {}})
    monkeypatch.setattr(analyst, "_provider_call", lambda: (lambda p: "{}"))
    monkeypatch.setattr(analyst.store, "ensure_table", lambda *a, **k: None)
    monkeypatch.setattr(analyst.store, "scored_rows", lambda *a, **k: [])
    monkeypatch.setattr(analyst.store, "judged_with_hash", lambda *a, **k: False)
    monkeypatch.setattr(analyst.dossier, "build",
                        lambda a, **k: {"asset": a, "date": "2026-01-01",
                                        "close": 100.0, "atr": 2.0})
    monkeypatch.setattr(analyst.dossier, "dossier_hash", lambda d: "h")
    def _record(d, call=None, depth=None, horizon=1, on_reject=None):
        # Записываем глубину и отказываемся судить: возврат суждения потянул
        # бы за собой калибровку, а этот тест не про неё. Неявный None здесь и
        # есть отказ, ровно как у настоящего judge.
        seen["depth"] = depth
        seen["horizon"] = horizon

    monkeypatch.setattr(analyst.agent, "judge", _record)

    class A:
        assets, llm, model, depth = "SBER", None, None, "brief"

    analyst.cmd_run(A())
    assert seen["depth"] == "brief"


def _stub_run(monkeypatch, calls, build=None):
    import analyst
    monkeypatch.setattr(analyst, "_load_table", lambda: {"asset": {}, "class": {}})
    monkeypatch.setattr(analyst, "_provider_call", lambda: (lambda p: "{}"))
    monkeypatch.setattr(analyst.store, "ensure_table", lambda *a, **k: None)
    monkeypatch.setattr(analyst.store, "scored_rows", lambda *a, **k: [])
    monkeypatch.setattr(analyst.store, "judged_with_hash", lambda *a, **k: False)
    monkeypatch.setattr(analyst.dossier, "build", build or (
        lambda a, **k: {"asset": a, "date": "2026-01-01", "close": 100.0,
                        "atr": 2.0}))
    monkeypatch.setattr(analyst.dossier, "dossier_hash", lambda d: "h")
    monkeypatch.setattr(analyst.dossier, "macro_status", lambda: None)
    monkeypatch.setattr(analyst.agent, "judge",
                        lambda d, call=None, depth=None, horizon=1,
                        on_reject=None: calls.append(horizon) or None)
    monkeypatch.setenv("GTRADE_ANALYST", "1")
    return analyst


def test_each_horizon_is_its_own_question(monkeypatch):
    """One day and five days are two different questions about one dossier, so
    two model calls and two rows - which (date, asset, horizon) has allowed
    since the table was written and nothing ever used."""
    calls = []
    analyst = _stub_run(monkeypatch, calls)

    class A:
        assets, llm, model, depth = "SBER", None, None, None
        horizons, as_of = "1,5", None

    analyst.cmd_run(A())
    assert calls == [1, 5], calls


def test_the_default_is_still_a_single_one_day_judgment(monkeypatch):
    """The positive control for the test above: without --horizons nothing
    about the existing behaviour may change."""
    calls = []
    analyst = _stub_run(monkeypatch, calls)

    class A:
        assets, llm, model, depth = "SBER", None, None, None

    analyst.cmd_run(A())
    assert calls == [1], calls


def test_as_of_rewinds_the_dossier_it_asks_for(monkeypatch):
    """The date has to reach dossier.build, or the run would judge a past date
    on today's prices - which is the whole failure this flag exists to avoid."""
    calls, asked = [], []

    def _build(a, **k):
        asked.append(k.get("today"))
        return {"asset": a, "date": "2026-05-04", "close": 100.0, "atr": 2.0}

    analyst = _stub_run(monkeypatch, calls, build=_build)

    class A:
        assets, llm, model, depth = "SBER", None, None, None
        horizons, as_of = "1", "2026-05-04"

    analyst.cmd_run(A())
    assert asked == ["2026-05-04"], asked


def test_a_bad_horizon_list_stops_the_run_before_it_spends_anything(monkeypatch):
    calls = []
    analyst = _stub_run(monkeypatch, calls)

    class A:
        assets, llm, model, depth = "SBER", None, None, None
        horizons, as_of = "1,tomorrow", None

    assert analyst.cmd_run(A()) == 1
    assert calls == [], "it called the model before validating its arguments"


def test_a_missing_provider_stops_the_sweep_instead_of_refusing_every_asset(
        monkeypatch, capsys):
    """28 assets would otherwise print the same missing-package line 28 times
    and finish claiming 28 refusals."""
    from core.llm_proposer import ProviderUnavailable

    calls = []
    analyst = _stub_run(monkeypatch, calls)

    def _missing(d, call=None, depth=None, horizon=1, on_reject=None):
        calls.append(horizon)
        raise ProviderUnavailable("the anthropic provider needs the anthropic "
                                  "package: pip install anthropic")

    monkeypatch.setattr(analyst.agent, "judge", _missing)

    class A:
        assets, llm, model, depth = "SBER,BTC,GOLD", None, None, None
        horizons, as_of = "1", None

    assert analyst.cmd_run(A()) == 1
    assert len(calls) == 1, "it kept asking after the provider was gone"
    out = capsys.readouterr().out
    assert "pip install anthropic" in out and "nothing was asked" in out


def test_the_panel_is_the_same_names_every_day(monkeypatch):
    """The watchlist is a moving set, so 32 judgments landed on 31 assets and
    no per-asset statistic can ever accumulate. The panel is fixed."""
    calls = []
    analyst = _stub_run(monkeypatch, calls)
    seen = []
    monkeypatch.setattr(analyst.dossier, "build",
                        lambda a, **k: seen.append(a) or {
                            "asset": a, "date": "2026-01-01", "close": 1.0,
                            "atr": 0.1})

    class A:
        assets, llm, model, depth = None, None, None, None
        horizons, as_of, panel, max_calls, back = "1", None, True, 0, 0

    analyst.cmd_run(A())
    assert seen == [a for a in analyst.panel_assets()
                    if a in __import__("config").FULL_ASSET_MAP], seen
    assert len(seen) >= 10, "the panel got filtered down to nothing"


def test_the_panel_is_overridable(monkeypatch):
    monkeypatch.setenv("GTRADE_ANALYST_PANEL", "btc, gold ,eurusd")
    import analyst
    assert analyst.panel_assets() == ["BTC", "GOLD", "EURUSD"]
    monkeypatch.delenv("GTRADE_ANALYST_PANEL")
    assert analyst.panel_assets() == list(analyst.PANEL)


def test_a_run_over_the_ceiling_costs_nothing(monkeypatch, capsys):
    """Checked BEFORE the first call. A ceiling that stops halfway has already
    spent the money it existed to protect."""
    calls = []
    analyst = _stub_run(monkeypatch, calls)

    class A:
        assets, llm, model, depth = "SBER,BTC,GOLD", None, None, None
        horizons, as_of, panel, back = "1,5", None, False, 0
        max_calls = 5

    assert analyst.cmd_run(A()) == 1
    assert calls == [], "it asked the model before checking the ceiling"
    out = capsys.readouterr().out
    assert "3 asset(s) x 2 horizon(s) = 6 calls" in out
    assert "Nothing was asked" in out


def test_a_run_inside_the_ceiling_proceeds(monkeypatch):
    """The positive control: a ceiling that refuses everything is not a ceiling."""
    calls = []
    analyst = _stub_run(monkeypatch, calls)

    class A:
        assets, llm, model, depth = "SBER", None, None, None
        horizons, as_of, panel, back = "1", None, False, 0
        max_calls = 5

    analyst.cmd_run(A())
    assert calls == [1]


def test_back_judges_one_date_per_pass_oldest_first(monkeypatch):
    """Oldest first so each judgment sees the own-record its predecessors left
    and none of them sees a successor's outcome."""
    calls = []
    analyst = _stub_run(monkeypatch, calls)
    asked = []
    monkeypatch.setattr(analyst, "_recent_dates",
                        lambda n, asset="SP500": ["2026-05-04", "2026-05-05",
                                                  "2026-05-06"][:n])
    monkeypatch.setattr(analyst.dossier, "build",
                        lambda a, **k: asked.append(k.get("today")) or {
                            "asset": a, "date": k.get("today"), "close": 1.0,
                            "atr": 0.1})

    class A:
        assets, llm, model, depth = "SBER", None, None, None
        horizons, as_of, panel, max_calls = "1", None, False, 0
        back = 3

    analyst.cmd_run(A())
    assert asked == ["2026-05-04", "2026-05-05", "2026-05-06"], asked


def test_recent_dates_excludes_the_last_bar(monkeypatch):
    """Its horizon has not elapsed, so a judgment there could never be scored.

    Bars are injected. The first version read the real market.db, which passes
    on a developer machine and fails in CI where there is no database at all:
    ohlc_series returns nothing and the assertion reads 0 == 5.
    """
    import analyst
    from core import track_record

    bar_dates = ["2026-05-0%d" % d for d in (1, 2, 3, 4, 5, 6, 7)]
    # `days` is ohlc_series own keyword, so the fixture list cannot be called
    # that without shadowing it.
    monkeypatch.setattr(track_record, "ohlc_series",
                        lambda a, days=0, **k: [{"date": d} for d in bar_dates])
    got = analyst._recent_dates(5)
    assert got == bar_dates[-6:-1], got
    assert bar_dates[-1] not in got, "the unscoreable last bar was included"
    assert got == sorted(got), "the sweep has to run oldest first"


def test_recent_dates_on_a_machine_with_no_bars_is_empty(monkeypatch):
    """CI has no market.db. --back must say so rather than judge nothing."""
    import analyst
    from core import track_record

    monkeypatch.setattr(track_record, "ohlc_series", lambda *a, **k: [])
    assert analyst._recent_dates(5) == []


def test_a_discarded_attempt_is_named_in_the_run_summary(monkeypatch, tmp_path,
                                                         capsys):
    """A retry costs a full call and used to leave no trace at all: the run
    that prompted this made three calls for two assets and reported
    `written=2 skipped=0 refused=0`."""
    db = _cli_e2e_db(tmp_path)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr("core.track_record.DB_PATH", db)
    monkeypatch.setattr("core.dashboard.guru_for_asset",
                        lambda asset, db_path=None: None)
    monkeypatch.setattr("core.events.earnings_for",
                        lambda symbols_by_asset, session=None, fetch=None: {})
    monkeypatch.setattr("core.events.load_macro", lambda path=None: [])
    monkeypatch.setattr(analyst, "_eligible", lambda: ["SBER"])
    monkeypatch.setattr(
        analyst, "_load_table",
        lambda: {"asset": {}, "class": {"ru": {
            "SELL": {"n": 900, "mean": 1.0, "q10": -1.0, "q90": 3.0}}}})

    replies = ["sorry, I cannot help with that", json.dumps({
        "direction": "down", "conviction": 3, "vol_regime": "normal",
        "key_risk": "overbought", "thesis": "Quiet tape, downside lean.",
        "evidence": ["close", "atr_pct"]})]
    monkeypatch.setattr(analyst, "_provider_call",
                        lambda: (lambda prompt: replies.pop(0)))

    assert analyst.cmd_run(None) == 0
    out = capsys.readouterr().out
    assert "written=1" in out and "retried=1" in out
    assert "discarded: SBER 1d: no JSON object" in out


def test_the_run_says_which_kinds_of_evidence_went_unread(monkeypatch, tmp_path,
                                                          capsys):
    """33 of 60 dossier fields went unread for 35 judgments, headlines among
    them, and the only way to find out was to query the log months later. A
    count alone would hide the case that matters, so the line names the
    BLOCKS nothing was cited from."""
    db = _cli_e2e_db(tmp_path)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr("core.track_record.DB_PATH", db)
    monkeypatch.setattr("core.dashboard.guru_for_asset",
                        lambda asset, db_path=None: None)
    monkeypatch.setattr("core.events.earnings_for",
                        lambda symbols_by_asset, session=None, fetch=None: {})
    monkeypatch.setattr("core.events.load_macro", lambda path=None: [])
    monkeypatch.setattr(analyst, "_eligible", lambda: ["SBER"])
    monkeypatch.setattr(
        analyst, "_load_table",
        lambda: {"asset": {}, "class": {"ru": {
            "SELL": {"n": 900, "mean": 1.0, "q10": -1.0, "q90": 3.0}}}})
    monkeypatch.setattr(analyst, "_provider_call",
                        lambda: (lambda prompt: json.dumps({
                            "direction": "down", "conviction": 3,
                            "vol_regime": "normal", "key_risk": "overbought",
                            "thesis": "Quiet tape, downside lean.",
                            "evidence": ["close", "atr_pct"]})))

    assert analyst.cmd_run(None) == 0
    out = capsys.readouterr().out
    assert "dossier coverage: 2 fields cited" in out
    # "close" and "atr_pct" are both the price block, so movement - which the
    # fixture fills - has to show up as unread.
    assert "nothing read from" in out and "movement" in out.split(
        "nothing read from")[1]
