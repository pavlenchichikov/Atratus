"""Fitting the two numbers levels have always guessed.

core/levels.py shipped `K_ENTRY = 0.5` and `K_STOP = 2.0` as bare constants:
never fitted, never measured. This is the harness that fits them on replay and
refuses to adopt unless a held-out split says so, built on the same ES, split
and Wilcoxon gate the adopted timing policy already uses.
"""

import json

import numpy as np
import pytest

import train_levels as tl
from core import levels as levels_mod


def _series(closes, spread=1.0, atr=2.0, forex=False):
    """A flat-ish asset whose bars straddle the close by `spread`."""
    c = np.asarray(closes, dtype=float)
    return {"open": c.copy(), "close": c.copy(),
            "high": c + spread, "low": c - spread,
            "atr": np.full(len(c), atr, dtype=float),
            "is_forex": forex}


def _flip_at_end(n, keep=2):
    """Sides that hold long and then go flat, so every trade actually closes.
    A trade still running is dropped from the sample by design, so a series
    that never flips and never stops out would score nothing at all."""
    sides = np.ones(n, dtype=int)
    sides[-keep:] = 0
    return sides


def test_a_wider_stop_survives_noise_a_tight_one_does_not():
    """The point of fitting: on an asset that shakes before it moves, a two-ATR
    stop and a half-ATR stop are not the same trade."""
    closes = [100, 99, 98.6, 99.5, 101, 102, 103, 104, 105, 106]
    s = _series(closes, spread=1.2, atr=2.0)
    sides = _flip_at_end(len(closes))
    tight = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 0.5}, sides=sides)
    wide = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 3.0}, sides=sides)
    assert tight["n"] > 0 and wide["n"] > 0
    assert wide["mean_ret"] > tight["mean_ret"]


def test_a_flat_asset_pays_the_costs_and_nothing_else():
    s = _series([100.0] * 8, spread=0.2, atr=1.0)
    sides = _flip_at_end(8)
    out = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 2.0}, sides=sides)
    assert out["n"] > 0
    assert out["mean_ret"] < 0, "a trade that goes nowhere still pays two legs"


def test_bars_with_no_atr_are_skipped_not_scored_as_zero():
    """A zero ATR would put every level on the close, which is not a trade."""
    s = _series([100, 101, 102, 103], atr=0.0)
    out = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS), sides=np.ones(4, dtype=int))
    assert out == {"mean_ret": 0.0, "n": 0, "wins": 0}


def test_a_flat_side_issues_nothing():
    s = _series([100, 101, 102, 103])
    out = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS), sides=np.zeros(4, dtype=int))
    assert out["n"] == 0


def test_forex_is_charged_its_own_cheaper_legs():
    closes = [100, 100.4, 100.8, 101.2, 101.6, 102.0]
    sides = _flip_at_end(len(closes))
    normal = tl.eval_levels(_series(closes, forex=False), dict(tl.DEFAULT_PARAMS),
                            sides=sides)
    forex = tl.eval_levels(_series(closes, forex=True), dict(tl.DEFAULT_PARAMS),
                           sides=sides)
    assert forex["mean_ret"] > normal["mean_ret"]


# --- the gate ---------------------------------------------------------------

def _by_asset(n, closes):
    return {"A%d" % i: _series(closes) for i in range(n)}


def test_the_gate_holds_when_the_effect_is_below_the_floor(monkeypatch):
    """A statistically clean but economically meaningless gain must not adopt."""
    monkeypatch.setattr(tl, "sides_for", lambda s: np.ones(len(s["close"]), dtype=int))
    monkeypatch.setattr(tl, "eval_levels",
                        lambda s, p, sides=None: {"mean_ret": 0.00001 if p["k_stop"] != 2.0 else 0.0,
                                                  "n": 5, "wins": 3})
    gate = tl.gate_policy(_by_asset(12, [100, 101]), {"k_entry": 0.5, "k_stop": 3.0})
    assert gate["p"] < 0.05
    assert gate["verdict"] == "HOLD"
    assert gate["mean_d"] < gate["floor"]


def test_the_gate_adopts_a_real_gain(monkeypatch):
    monkeypatch.setattr(tl, "sides_for", lambda s: np.ones(len(s["close"]), dtype=int))
    monkeypatch.setattr(tl, "eval_levels",
                        lambda s, p, sides=None: {"mean_ret": 0.004 if p["k_stop"] != 2.0 else 0.0,
                                                  "n": 5, "wins": 3})
    gate = tl.gate_policy(_by_asset(12, [100, 101]), {"k_entry": 0.5, "k_stop": 3.0})
    assert gate["verdict"] == "ADOPT"


def test_too_few_assets_is_never_an_adoption(monkeypatch):
    monkeypatch.setattr(tl, "sides_for", lambda s: np.ones(len(s["close"]), dtype=int))
    monkeypatch.setattr(tl, "eval_levels",
                        lambda s, p, sides=None: {"mean_ret": 0.05 if p["k_stop"] != 2.0 else 0.0,
                                                  "n": 5, "wins": 3})
    gate = tl.gate_policy(_by_asset(4, [100, 101]), {"k_entry": 0.5, "k_stop": 3.0})
    assert gate["n"] == 4 and gate["verdict"] == "HOLD"


def test_the_floor_is_in_the_units_this_gate_measures():
    """The 2026-08-18 failure was a floor of 0.5 in Score units checked against a
    value in AUC units. Mean net return per signal is a fraction, so a floor
    anywhere near 0.5 would reject everything forever."""
    assert 0.0 < tl.ADOPT_FLOOR < 0.01


def test_a_held_verdict_writes_nothing(tmp_path):
    path = str(tmp_path / "levels_policy.json")
    assert tl.save_policy({"k_entry": 0.4, "k_stop": 3.0},
                          {"verdict": "HOLD"}, path=path) is None
    assert not (tmp_path / "levels_policy.json").exists()


def test_an_adopted_verdict_writes_the_params_and_its_evidence(tmp_path):
    path = str(tmp_path / "levels_policy.json")
    gate = {"verdict": "ADOPT", "p": 0.01, "mean_d": 0.002, "n": 12}
    tl.save_policy({"k_entry": 0.4, "k_stop": 3.0}, gate, path=path)
    body = json.loads(open(path, encoding="utf-8").read())
    # every gene is stored, not only the ones the caller happened to pass:
    # a policy file that omits a delta would load as a different policy.
    assert body["params"]["k_entry"] == 0.4
    assert body["params"]["k_stop"] == 3.0
    assert set(body["params"]) == {n for n, _l, _h, _i in tl.PARAM_SPECS}
    assert body["gate"]["p"] == 0.01
    assert body["baseline"] == tl.baseline_params()


# --- what serving does with it ----------------------------------------------

BARS = [{"date": "2026-01-%02d" % (i + 1), "high": 101.0 + i, "low": 99.0 + i,
         "close": 100.0 + i} for i in range(30)]


def test_serving_falls_back_to_the_shipped_constants(tmp_path):
    assert levels_mod.load_policy(str(tmp_path / "absent.json")) is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert levels_mod.load_policy(str(tmp_path / "bad.json")) is None


def test_a_policy_with_a_nonsense_multiplier_is_refused(tmp_path):
    p = tmp_path / "levels_policy.json"
    p.write_text(json.dumps({"params": {"k_entry": 0.0, "k_stop": 2.0}}),
                 encoding="utf-8")
    assert levels_mod.load_policy(str(p)) is None


def test_serving_uses_the_fitted_multipliers_when_they_exist(monkeypatch):
    monkeypatch.setattr(levels_mod, "load_policy",
                        lambda path=None: {"k_entry": 1.0, "k_stop": 4.0})
    wide = levels_mod.levels(BARS, "BUY")
    monkeypatch.setattr(levels_mod, "load_policy", lambda path=None: None)
    shipped = levels_mod.levels(BARS, "BUY")
    assert wide["status"] == shipped["status"] == "ok"
    assert wide["entry_high"] > shipped["entry_high"]
    assert wide["stop"] < shipped["stop"]


def test_an_explicit_multiplier_still_wins_over_the_policy(monkeypatch):
    """Callers that pass a number are running an experiment; the adopted policy
    must not silently override them."""
    monkeypatch.setattr(levels_mod, "load_policy",
                        lambda path=None: {"k_entry": 1.0, "k_stop": 4.0})
    row = levels_mod.levels(BARS, "BUY", k_stop=2.0)
    assert row["stop"] == pytest.approx(row["close"] - 2.0 * row["atr"])


# --- the regime-conditioned form --------------------------------------------

def test_the_flat_policy_is_the_conditioned_one_with_its_deltas_at_zero():
    """Which is what makes the flat fit an honest baseline for this one."""
    flat = levels_mod.effective_multipliers({"k_entry": 0.5, "k_stop": 2.0})
    assert flat == (0.5, 2.0)
    with_zero_deltas = levels_mod.effective_multipliers(
        dict(levels_mod.POLICY_DEFAULTS), taleb_hi=True, risky=True)
    assert with_zero_deltas == (levels_mod.K_ENTRY, levels_mod.K_STOP)


def test_both_regimes_apply_at_once_and_add():
    p = {"k_entry": 0.5, "k_stop": 2.0, "d_stop_hi_taleb": 1.0,
         "d_stop_risky": 0.5, "d_entry_hi_taleb": 0.1, "d_entry_risky": 0.2}
    assert levels_mod.effective_multipliers(p, taleb_hi=True) == (0.6, 3.0)
    assert levels_mod.effective_multipliers(p, risky=True) == (0.7, 2.5)
    k_entry, k_stop = levels_mod.effective_multipliers(p, taleb_hi=True, risky=True)
    assert (round(k_entry, 6), round(k_stop, 6)) == (0.8, 3.5)


def test_a_delta_can_never_drive_a_multiplier_to_zero_or_below():
    """A non-positive multiplier is not a tighter level, it is a stop on the
    wrong side of the close."""
    p = {"k_entry": 0.2, "k_stop": 0.6, "d_stop_hi_taleb": -5.0,
         "d_entry_hi_taleb": -5.0}
    k_entry, k_stop = levels_mod.effective_multipliers(p, taleb_hi=True)
    assert k_entry > 0 and k_stop > 0


def test_the_replay_reacts_to_a_regime_delta():
    closes = [100, 99, 98.6, 99.5, 101, 102, 103, 104, 105, 106]
    s = _series(closes, spread=1.2, atr=2.0)
    s["taleb_hi"] = np.ones(len(closes), dtype=bool)
    sides = _flip_at_end(len(closes))
    tight = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 3.0,
                               "d_stop_hi_taleb": -2.5}, sides=sides)
    wide = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 3.0,
                              "d_stop_hi_taleb": 0.0}, sides=sides)
    assert tight["mean_ret"] != wide["mean_ret"], "the delta did not reach the trade"


def test_the_gate_measures_against_the_live_policy_not_the_constants(monkeypatch):
    """A conditioned fit judged against the shipped constants would take credit
    for whatever the flat fit already earned."""
    monkeypatch.setattr(levels_mod, "load_policy",
                        lambda path=None: dict(levels_mod.POLICY_DEFAULTS,
                                               k_stop=3.0))
    assert tl.baseline_params()["k_stop"] == 3.0
    monkeypatch.setattr(levels_mod, "load_policy", lambda path=None: None)
    assert tl.baseline_params()["k_stop"] == levels_mod.K_STOP
