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
    }


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
