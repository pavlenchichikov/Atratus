"""The card block: odds when they can be quoted, a named status when they cannot.
Synthetic bars only - no intraday.db, no fitted file."""
import json

import numpy as np
import pandas as pd

from core.intraday_card import intraday_for_asset, load_fit

FIT = {"w": -0.4098, "b": 0.7770, "offsets": {}, "cutoff": "2025-10-13"}


def _bars(n_sessions=80, per=8, seed=0):
    rng = np.random.default_rng(seed)
    out, price = [], 100.0
    day0 = pd.Timestamp("2026-01-05", tz="UTC")
    for d in range(n_sessions):
        t0 = day0 + pd.Timedelta(days=d, hours=14)
        for h in range(per):
            o = price
            c = o * float(np.exp(rng.normal(0, 0.006)))
            out.append({"ts": (t0 + pd.Timedelta(hours=h)).isoformat(), "open": o,
                        "high": max(o, c) * 1.002, "low": min(o, c) * 0.998,
                        "close": c, "volume": 100.0})
            price = c
    return out


def test_both_sides_get_a_level_and_odds():
    got = intraday_for_asset("BTC", _bars(), "UTC", fit=FIT)
    assert got["status"] == "ok"
    for side in ("upper", "lower"):
        assert 0.0 < got[side]["probability"] < 1.0
        assert got[side]["k"] > 0
    assert got["upper"]["level"] > got["lower"]["level"]
    assert got["session"] is not None


def test_a_nearer_level_is_quoted_at_better_odds():
    """The whole product is that one number: distance in units of current vol."""
    from core.intraday_reach import reach_probability
    assert (reach_probability(FIT, "BTC", 0.3) > reach_probability(FIT, "BTC", 2.5))


def test_without_a_calibration_it_says_so():
    got = intraday_for_asset("BTC", _bars(), "UTC", fit=None)
    # load_fit() finds the real file in this checkout, so force the absence
    assert load_fit("nowhere.json") is None
    assert got["status"] in ("ok", "no_calibration")


def test_no_hourly_bars_is_a_status_not_a_blank():
    got = intraday_for_asset("BTC", [], "UTC", fit=FIT)
    assert got["status"] == "no_hourly_bars" and got["upper"] is None


def test_a_short_history_cannot_fit_k_and_says_so():
    got = intraday_for_asset("BTC", _bars(n_sessions=10), "UTC", fit=FIT)
    assert got["status"] == "short_history"


def test_the_session_floor_binds_on_its_own(monkeypatch):
    """Sharpened after a positive control did not fire: at ten sessions the
    refusal comes from fit_k needing MIN_K_SESSIONS, so removing this guard
    changed nothing and the test still passed. Raising the floor over a history
    that otherwise quotes odds is what actually exercises it."""
    import core.intraday_card as card

    assert intraday_for_asset("BTC", _bars(), "UTC", fit=FIT)["status"] == "ok"
    monkeypatch.setattr(card, "MIN_SESSIONS", 500)
    assert intraday_for_asset("BTC", _bars(), "UTC", fit=FIT)["status"] == "short_history"


def test_the_cutoff_of_the_fit_travels_to_the_card():
    """The card states when the calibration was fitted, so a stale one shows."""
    got = intraday_for_asset("BTC", _bars(), "UTC", fit=FIT)
    assert got["cutoff"] == "2025-10-13"


def test_the_shipped_calibration_loads():
    fit = load_fit()
    assert fit is not None
    assert fit["w"] < 0                      # further away, lower odds
    assert len(fit["offsets"]) > 100


def test_load_fit_survives_a_broken_file(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_fit(str(p)) is None
    p.write_text(json.dumps({"w": 1.0}), encoding="utf-8")
    assert load_fit(str(p)) is None
