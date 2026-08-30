"""core.analyst.dossier: what the analyst is allowed to see, and what it is not."""

import json
import sqlite3

import pytest

from core.analyst import dossier


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE sber (Date TEXT, Open REAL, Close REAL, '
                'High REAL, Low REAL)')
    con.executemany('INSERT INTO sber VALUES (?,?,?,?,?)',
                    [(f"2026-01-{i + 1:02d}", 100.0, 100.0 + i * 0.1,
                      101.0 + i * 0.1, 99.0 + i * 0.1) for i in range(28)])
    con.commit()
    con.close()
    return path


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """The context sources each reach a database or the network on their own.
    No test may trigger any of them unmocked, so every one is stubbed here
    regardless of which test is running."""
    monkeypatch.setattr("core.dashboard.guru_for_asset", lambda asset, db_path=None: None)
    monkeypatch.setattr("core.events.earnings_for",
                        lambda symbols_by_asset, session=None, fetch=None: {})
    monkeypatch.setattr("core.events.load_macro", lambda path=None: [])
    # The profile and the headlines reach yfinance and the news feeds, so they
    # are stubbed at the dossier's own boundary: patching yfinance itself would
    # leave the news path live, and patching news_analyzer would leave yfinance
    # live. One seam per source, both closed.
    from core.analyst import dossier as _d
    # dict(PROFILE_BLANK), not a restatement of it: the stub must inherit the
    # real block's shape or it goes stale the next time a field is added, which
    # is exactly what happened when the valuation fields arrived.
    monkeypatch.setattr(_d, "_profile", lambda asset: dict(_d.PROFILE_BLANK))
    monkeypatch.setattr(_d, "_headlines", lambda asset, limit=6: {"headlines": []})


def test_the_dossier_carries_price_scale_and_regime(db):
    d = dossier.build("SBER", db_path=db)
    assert d["asset"] == "SBER"
    assert d["close"] == pytest.approx(102.7)
    assert d["atr"] is not None
    assert d["atr_pct"] == pytest.approx(d["atr"] / d["close"])


def test_the_dossier_never_carries_the_ensembles_opinion(db):
    # THE constraint of this whole feature. The analyst is a second opinion
    # only for as long as it cannot see the first one. Without this test
    # someone adds `probability` to the dossier for completeness in three
    # months and the agent silently stops being independent while continuing
    # to look like it.
    d = dossier.build("SBER", db_path=db)
    flat = json.dumps(d).lower()
    for banned in dossier.FORBIDDEN_KEYS:
        assert banned not in flat, f"{banned} leaked into the dossier"


def test_forbidden_keys_names_every_channel_to_the_ensemble():
    assert dossier.FORBIDDEN_KEYS >= {"probability", "cb_prob", "lstm_prob",
                                      "meta_prob", "signal", "timing_action",
                                      "shadow_action", "sig_shown",
                                      "correct", "actual_next_ret",
                                      "model_version"}


def test_the_dossier_shape_is_declared_and_any_new_field_must_be_too(db):
    # The scan above is vocabulary-bound: a probability stored as
    # "confidence" would pass it. This test closes the other half. The
    # dossier has a fixed shape, so a field nobody declared is a field
    # nobody reviewed, and reviewing it is the moment to ask whether it
    # carries the ensemble's opinion. Adding a field here is meant to be a
    # deliberate act, not a silent one.
    assert set(dossier.build("SBER", db_path=db)) == {
        # price and scale
        "asset", "date", "close", "atr", "atr_pct", "bars_available",
        # movement
        "ret_1", "ret_5", "ret_20", "ret_60", "streak_days",
        # where the price sits, in the asset's own units of movement
        "high_20", "low_20", "atr_to_high_20", "atr_to_low_20", "drawdown_60",
        # volatility against this asset's own norm, not an absolute threshold
        "vol_20", "vol_20_vs_60",
        # fundamentals and the calendar
        "guru_verdict", "guru_pct", "next_earnings", "macro_events",
        # the analyst's OWN history on this asset. Not the ensemble's: these
        # come from analyst_log, so they say what this agent previously called
        # and how that turned out, which is the one track record it is entitled
        # to see.
        "past_calls", "past_hit_rate", "past_last_call", "past_last_outcome",
        # flow: how much actually traded, and how the day opened
        "volume_vs_20", "turnover", "gap_open", "range_atr",
        # the market the asset moved in, so a fall can be told apart from a
        # fall that everything shared
        "benchmark", "benchmark_ret_1", "benchmark_ret_20",
        "corr_to_benchmark_60", "vix_level", "vix_chg_20",
        # what the instrument IS, as opposed to what anyone thinks of it
        "sector", "industry", "market_cap", "float_shares", "short_ratio",
        "beta", "ex_dividend_date",
        # valuation, from Yahoo where it resolves and from Smart-Lab where it
        # does not. Not an opinion about the asset: a bank at a P/E of 3.5 with
        # a 14 percent yield is a fact, and the block was blank for every
        # Moscow-listed name until this was wired to the table the guru report
        # already downloads.
        "pe", "roe", "debt_ebitda", "div_yield",
        # raw headlines, without the sentiment score computed on them
        "headlines",
    }


def test_a_moscow_name_gets_its_fundamentals_from_smartlab(monkeypatch):
    """Yahoo cannot resolve a bare MOEX ticker, so can_have_earnings skips it
    and the whole block used to come back blank. The positive control is the
    empty-table half: without it this passes on a function that hardcodes."""
    dossier._SMARTLAB_CACHE.clear()
    monkeypatch.setattr(dossier, "_smartlab_map", lambda: {
        "SBER": {"pe": 3.5, "roe": 0.227, "debt": 0.0, "div": 14.0}})
    assert dossier._smartlab_fundamentals("SBER") == {
        "pe": 3.5, "roe": 0.227, "debt_ebitda": None, "div_yield": 14.0}
    # the remap guru_report applies for the two renamed tickers
    monkeypatch.setattr(dossier, "_smartlab_map", lambda: {
        "YDEX": {"pe": 12.0, "roe": 0.1, "debt": 1.5, "div": 0.0}})
    assert dossier._smartlab_fundamentals("YNDX")["pe"] == 12.0
    # 99.0 is the fetcher's "not reported" filler, never a real P/E
    monkeypatch.setattr(dossier, "_smartlab_map", lambda: {
        "GAZP": {"pe": 99.0, "roe": 0.0, "debt": 0.0, "div": 0.0}})
    assert dossier._smartlab_fundamentals("GAZP")["pe"] is None
    monkeypatch.setattr(dossier, "_smartlab_map", dict)
    assert dossier._smartlab_fundamentals("SBER") == dossier._FUNDAMENTALS_BLANK


def test_a_dead_network_is_visible_block_by_block():
    """Every network field is wrapped in _safe, so a run with no connection
    produces judgments and no complaint. filled_blocks is what the CLI counts
    to say so out loud."""
    full = {"asset": "AAPL", "headlines": ["x"], "guru_verdict": "BUY",
            "pe": 3.5, "next_earnings": "2026-09-01"}
    assert dossier.filled_blocks(full) == {k: 1 for k in dossier.NETWORK_BLOCKS}
    assert dossier.filled_blocks({"asset": "AAPL"}) == {
        k: 0 for k in dossier.NETWORK_BLOCKS}
    # a Moscow name carrying only its Smart-Lab valuation is a FILLED block,
    # not a thin one: sector and market cap were never available for it
    thin = {"asset": "SBER", "headlines": ["x"], "guru_verdict": "BUY",
            "sector": None, "market_cap": None, "pe": 3.5}
    assert dossier.filled_blocks(thin)["fundamentals"] == 1


def test_what_an_asset_cannot_have_is_not_a_missing_source():
    """The first version of this summary told a person to check the network
    because SBER had no earnings date and gold no P/E. Neither has a source at
    all, so both must read as not-applicable rather than as a failed fetch.
    The AAPL line is the positive control: for a name that CAN have both, an
    absence is still a real 0."""
    assert dossier.filled_blocks({"asset": "SBER"})["earnings"] is None
    assert dossier.filled_blocks({"asset": "SBER"})["fundamentals"] == 0
    gold = dossier.filled_blocks({"asset": "GOLD"})
    assert gold["earnings"] is None and gold["fundamentals"] is None
    assert gold["headlines"] == 0 and gold["guru"] == 0
    aapl = dossier.filled_blocks({"asset": "AAPL"})
    assert aapl["earnings"] == 0 and aapl["fundamentals"] == 0


def test_macro_names_the_file_rather_than_blaming_the_network(monkeypatch,
                                                              tmp_path):
    """macro_events is read from a local calendar, so an empty one says nothing
    about the connection. It shipped only as an example, which is why it was
    empty for all 853 assets."""
    from core import events

    monkeypatch.setattr(events, "MACRO_PATH",
                        str(tmp_path / "macro_calendar.json"))
    assert "does not exist" in dossier.macro_status()

    cal = tmp_path / "macro_calendar.json"
    cal.write_text("[]", encoding="utf-8")
    assert "no usable events" in dossier.macro_status()

    # load_macro is stubbed to [] for every test in this file, so the "a real
    # calendar is quiet" case has to say so explicitly. Parsing the file is
    # core.events' own test, not this one's.
    monkeypatch.setattr(events, "load_macro",
                        lambda path=None: [{"name": "FOMC"}])
    assert dossier.macro_status() is None


def test_the_hash_is_stable_for_equal_content(db):
    a, b = dossier.build("SBER", db_path=db), dossier.build("SBER", db_path=db)
    assert dossier.dossier_hash(a) == dossier.dossier_hash(b)


def test_the_hash_changes_when_the_content_does(db):
    a = dossier.build("SBER", db_path=db)
    b = dict(a, close=a["close"] + 1.0)
    assert dossier.dossier_hash(a) != dossier.dossier_hash(b)


def test_an_asset_with_no_bars_returns_an_empty_dossier(db):
    d = dossier.build("NOSUCH", db_path=db)
    assert d["close"] is None and d["atr"] is None


def test_context_fields_are_mapped_from_the_real_source_shapes(db, monkeypatch):
    # guru_for_asset returns "verdict"/"pct", not "council_verdict"/"council_pct";
    # load_macro returns "name", not "title". This test pins those field names.
    monkeypatch.setattr("core.dashboard.guru_for_asset",
                        lambda asset, db_path=None: {"verdict": "BUY", "pct": 75.0})
    monkeypatch.setattr(
        "core.events.earnings_for",
        lambda symbols_by_asset, session=None, fetch=None:
            {"SBER": {"date": "2026-09-01", "confirmed": True}})
    monkeypatch.setattr(
        "core.events.load_macro",
        lambda path=None: [{"kind": "macro", "asset": None, "date": "2026-09-05",
                            "name": "CPI", "importance": "high", "confirmed": True}])
    d = dossier.build("SBER", db_path=db)
    assert d["guru_verdict"] == "BUY"
    assert d["guru_pct"] == 75.0
    assert d["next_earnings"] == {"date": "2026-09-01", "confirmed": True}
    assert d["macro_events"] == ["CPI"]


def test_a_dead_context_source_does_not_stop_the_dossier(db, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network is down")
    monkeypatch.setattr("core.dashboard.guru_for_asset", _boom)
    monkeypatch.setattr("core.events.earnings_for", _boom)
    monkeypatch.setattr("core.events.load_macro", _boom)
    d = dossier.build("SBER", db_path=db)
    assert d["guru_verdict"] is None
    assert d["guru_pct"] is None
    assert d["next_earnings"] is None
    assert d["macro_events"] == []
