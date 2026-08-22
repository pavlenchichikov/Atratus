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

# The ATR the replay uses is the real 14-period one computed from the bars, the
# same one serving draws levels with, so a series has to carry enough warm-up
# for it to exist at all.
WARMUP = 20


def _series(closes, spread=1.0, forex=False):
    """Bars straddling the close by `spread`, with a flat warm-up prepended."""
    c = np.asarray([closes[0]] * WARMUP + list(closes), dtype=float)
    return {"open": c.copy(), "close": c.copy(),
            "high": c + spread, "low": c - spread,
            "is_forex": forex}


def _sides(closes, keep=2):
    """Long through the whole series, flat at the end so every trade closes."""
    sides = np.ones(WARMUP + len(closes), dtype=int)
    sides[-keep:] = 0
    return sides


def test_a_wider_stop_survives_noise_a_tight_one_does_not():
    """The point of fitting: on an asset that shakes before it moves, a two-ATR
    stop and a half-ATR stop are not the same trade."""
    closes = [100, 99, 98.6, 99.5, 101, 102, 103, 104, 105, 106]
    s = _series(closes, spread=1.2)
    sides = _sides(closes)
    tight = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 0.5}, sides=sides,
                           objective="rate")
    wide = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 3.0}, sides=sides,
                          objective="rate")
    assert tight["n"] > 0 and wide["n"] > 0
    assert wide["mean_ret"] > tight["mean_ret"]


def test_a_flat_asset_pays_the_costs_and_nothing_else():
    s = _series([100.0] * 8, spread=0.2)
    sides = _sides([100.0] * 8)
    out = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 2.0}, sides=sides, objective="rate")
    assert out["n"] > 0
    assert out["mean_ret"] < 0, "a trade that goes nowhere still pays two legs"


def test_bars_with_no_atr_are_skipped_not_scored_as_zero():
    """A zero ATR would put every level on the close, which is not a trade."""
    s = _series([100, 101, 102, 103], spread=0.0)   # no range at all, so no ATR
    s["high"] = s["close"].copy()
    s["low"] = s["close"].copy()
    s["open"] = s["close"].copy()
    out = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS),
                         sides=np.ones(WARMUP + 4, dtype=int), objective="rate")
    assert out["n"] == 0 and out["ret_per_bar"] == 0.0 and out["total_ret"] == 0.0


def test_a_flat_side_issues_nothing():
    s = _series([100, 101, 102, 103])
    out = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS),
                         sides=np.zeros(WARMUP + 4, dtype=int), objective="rate")
    assert out["n"] == 0


def test_forex_is_charged_its_own_cheaper_legs():
    closes = [100, 100.4, 100.8, 101.2, 101.6, 102.0]
    sides = _sides(closes)
    normal = tl.eval_levels(_series(closes, forex=False), dict(tl.DEFAULT_PARAMS),
                            sides=sides, objective="rate")
    forex = tl.eval_levels(_series(closes, forex=True), dict(tl.DEFAULT_PARAMS),
                           sides=sides, objective="rate")
    assert forex["mean_ret"] > normal["mean_ret"]


# --- the gate ---------------------------------------------------------------

def _by_asset(n, closes):
    return {"A%d" % i: _series(closes) for i in range(n)}


def test_the_gate_holds_when_the_effect_is_below_the_floor(monkeypatch):
    """A statistically clean but economically meaningless gain must not adopt."""
    monkeypatch.setattr(tl, "OBJECTIVE", "rate")
    monkeypatch.setattr(tl, "walk_for",
                        lambda s: (np.ones(len(s["close"]), dtype=int),
                                   ["HOLD"] * len(s["close"])))
    monkeypatch.setattr(tl, "eval_levels",
                        lambda s, p, sides=None, actions=None, objective=None: {
                            "score": 0.00001 if p["k_stop"] != 2.0 else 0.0,
                            "n": 5, "wins": 3})
    gate = tl.gate_policy(_by_asset(12, [100, 101]), {"k_entry": 0.5, "k_stop": 3.0})
    assert gate["p"] < 0.05
    assert gate["verdict"] == "HOLD"
    assert gate["mean_d"] < gate["floor"]


def test_the_gate_adopts_a_real_gain(monkeypatch):
    monkeypatch.setattr(tl, "OBJECTIVE", "rate")
    monkeypatch.setattr(tl, "walk_for",
                        lambda s: (np.ones(len(s["close"]), dtype=int),
                                   ["HOLD"] * len(s["close"])))
    monkeypatch.setattr(tl, "eval_levels",
                        lambda s, p, sides=None, actions=None, objective=None: {
                            "score": 0.004 if p["k_stop"] != 2.0 else 0.0,
                            "n": 5, "wins": 3})
    gate = tl.gate_policy(_by_asset(12, [100, 101]), {"k_entry": 0.5, "k_stop": 3.0})
    assert gate["verdict"] == "ADOPT"


def test_too_few_assets_is_never_an_adoption(monkeypatch):
    monkeypatch.setattr(tl, "OBJECTIVE", "rate")
    monkeypatch.setattr(tl, "walk_for",
                        lambda s: (np.ones(len(s["close"]), dtype=int),
                                   ["HOLD"] * len(s["close"])))
    monkeypatch.setattr(tl, "eval_levels",
                        lambda s, p, sides=None, actions=None, objective=None: {
                            "score": 0.05 if p["k_stop"] != 2.0 else 0.0,
                            "n": 5, "wins": 3})
    gate = tl.gate_policy(_by_asset(4, [100, 101]), {"k_entry": 0.5, "k_stop": 3.0})
    assert gate["n"] == 4 and gate["verdict"] == "HOLD"


def test_the_floor_is_in_the_units_this_gate_measures():
    """The 2026-08-18 failure was a floor of 0.5 in Score units checked against a
    value in AUC units. Net return per bar is a small fraction, so a floor
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
    s = _series(closes, spread=1.2)
    s["taleb_hi"] = np.ones(WARMUP + len(closes), dtype=bool)
    sides = _sides(closes)
    tight = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 3.0,
                               "d_stop_hi_taleb": -2.5}, sides=sides, objective="rate")
    wide = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 3.0,
                              "d_stop_hi_taleb": 0.0}, sides=sides, objective="rate")
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


def test_the_reward_no_longer_pays_for_scratching_every_trade():
    """The fix, with the exploit it fixes as its own control.

    A stop tight enough to scratch every trade RAISES the average of a trade,
    because each loss is tiny, while leaving the strategy with nothing. Driven
    at the per-trade mean, the ES pinned k_stop to the bottom of its range and
    turned every regime delta negative. Return per bar has to prefer the sane
    stop on the same data where the mean prefers the scratch.
    """
    closes = [100, 99.2, 99.6, 100.4, 101.2, 102.0, 102.8, 103.6, 104.4, 105.2,
              106.0, 106.8]
    s = _series(closes, spread=0.9)
    sides = _sides(closes)
    scratch = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 0.5}, sides=sides, objective="rate")
    sane = tl.eval_levels(s, {"k_entry": 0.5, "k_stop": 2.5}, sides=sides, objective="rate")
    assert scratch["n"] > 0 and sane["n"] > 0
    assert sane["ret_per_bar"] > scratch["ret_per_bar"], (
        "the reward still pays for cutting every trade to nothing")


def test_the_rate_is_per_bar_so_assets_of_different_length_compare():
    """fitness() reduces ACROSS assets, and a raw sum would make a long history
    outrank a good policy."""
    closes = [100, 99.2, 99.6, 100.4, 101.2, 102.0, 102.8, 103.6]
    short = _series(closes, spread=0.9)
    long_ = _series(closes * 3, spread=0.9)
    a = tl.eval_levels(short, dict(tl.DEFAULT_PARAMS), sides=_sides(closes),
                       objective="rate")
    b = tl.eval_levels(long_, dict(tl.DEFAULT_PARAMS), sides=_sides(closes * 3),
                       objective="rate")
    assert b["total_ret"] != a["total_ret"]
    assert abs(b["ret_per_bar"] - a["ret_per_bar"]) < abs(b["total_ret"] - a["total_ret"])


# --- the equity objective ---------------------------------------------------
#
# The reason it exists: under `rate` the trades overlap, so the number is not
# an equity curve, and nothing in it charges for drawdown. That is what left
# k_stop with no interior optimum - wider was better all the way to the top of
# its range. `equity` issues one trade per ENTER, so the trades do not overlap
# and score_strategy's drawdown and Sharpe terms mean something.

def _actions(n, enter_at):
    a = ["HOLD"] * n
    for i in enter_at:
        a[i] = "ENTER"
    return a


def test_equity_issues_one_trade_per_enter_not_one_per_bar():
    closes = [100, 101, 102, 101, 103, 104, 105, 106]
    s = _series(closes, spread=0.9)
    sides = _sides(closes)
    n = len(sides)
    rate = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS), sides=sides,
                          objective="rate")
    eq = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS), sides=sides,
                        actions=_actions(n, [WARMUP]), objective="equity")
    assert rate["n"] > 5, "the rate reading issues on every bar carrying a side"
    assert eq["n"] == 1, "the equity reading issues once, on the ENTER bar"


def test_equity_issues_nothing_when_the_policy_never_enters():
    closes = [100, 101, 102, 103]
    s = _series(closes)
    sides = _sides(closes)
    out = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS), sides=sides,
                         actions=["HOLD"] * len(sides), objective="equity")
    assert out["n"] == 0


def test_issue_bars_refuses_the_equity_reading_without_actions():
    """Sides alone cannot tell an ENTER from a HOLD, so they cannot answer it."""
    with pytest.raises(ValueError):
        tl._issue_bars(np.ones(5, dtype=int), None, "equity")


def test_an_unknown_objective_is_refused_not_guessed():
    s = _series([100, 101, 102])
    with pytest.raises(ValueError):
        tl.eval_levels(s, dict(tl.DEFAULT_PARAMS),
                       sides=_sides([100, 101, 102]), objective="sharpe")


def test_the_same_total_return_scores_worse_when_the_path_is_worse():
    """The whole point of the change, isolated from any price data.

    Two accounts end the slice on the same money. One got there in a straight
    line, the other by giving most of it back first. Under `rate` these are the
    identical number; score_strategy separates them, which is what gives a
    wider stop something to pay.
    """
    smooth = [0.05] * 12
    lumpy = [0.60] + [-0.05] * 11 + [0.05 * 12 - 0.60 + 0.05 * 11]
    assert abs(sum(smooth) - sum(lumpy)) < 1e-9, "same total, different path"
    p_s, n_s, w_s, dd_s, sh_s = tl._equity_stats(np.asarray(smooth), smooth)
    p_l, n_l, w_l, dd_l, sh_l = tl._equity_stats(np.asarray(lumpy), lumpy)
    assert dd_l > dd_s, "the lumpy path draws down and the smooth one does not"
    assert sh_l < sh_s
    from core.backtesting import score_strategy
    assert (score_strategy(p_l, dd_l, w_l, n_l, sh_l, min_trades=5)
            < score_strategy(p_s, dd_s, w_s, n_s, sh_s, min_trades=5))


def test_an_unfilled_limit_is_not_counted_as_a_losing_trade():
    """A level that never traded contributes an exact zero, so it belongs off
    the win rate rather than dragging it down."""
    closes = [100, 105, 110, 115]          # gaps away, the zone is never touched
    s = _series(closes, spread=0.05)
    sides = _sides(closes)
    out = tl.eval_levels(s, {"k_entry": 0.01, "k_stop": 2.0}, sides=sides,
                         actions=_actions(len(sides), [WARMUP]),
                         objective="equity")
    assert out["n_entered"] == 0


def test_a_barely_traded_asset_scores_unreliable_not_lucky():
    from core.backtesting import UNRELIABLE_SCORE
    closes = [100, 101, 102, 103, 104, 105]
    s = _series(closes, spread=0.9)
    sides = _sides(closes)
    out = tl.eval_levels(s, dict(tl.DEFAULT_PARAMS), sides=sides,
                         actions=_actions(len(sides), [WARMUP]),
                         objective="equity")
    assert out["score"] == UNRELIABLE_SCORE, "one trade cannot carry a verdict"


def test_the_gate_drops_a_pair_that_scored_unreliable(monkeypatch):
    from core.backtesting import UNRELIABLE_SCORE
    monkeypatch.setattr(tl, "OBJECTIVE", "equity")
    monkeypatch.setattr(tl, "walk_for",
                        lambda s: (np.ones(len(s["close"]), dtype=int),
                                   ["ENTER"] + ["HOLD"] * (len(s["close"]) - 1)))
    thin = {"A0", "A3"}

    def _ev(s, p, sides=None, actions=None, objective=None):
        score = (UNRELIABLE_SCORE if s.get("thin")
                 else (1.0 if p["k_stop"] != 2.0 else 0.0))
        return {"score": score, "n": 5, "wins": 3}

    monkeypatch.setattr(tl, "eval_levels", _ev)
    by_asset = _by_asset(12, [100, 101])
    for a in thin:
        by_asset[a]["thin"] = True
    gate = tl.gate_policy(by_asset, {"k_entry": 0.5, "k_stop": 3.0})
    assert gate["n"] == 10, "the two unreliable pairs are dropped, not averaged"
    assert gate["n_unscorable"] == 2
    assert set(gate["unscorable"]) == thin


def test_each_objective_has_a_floor_in_its_own_units():
    """The 2026-08-18 failure was a floor of 0.5 in Score units checked against
    a value in AUC units. Net return per bar is a small fraction; score points
    are not. One floor for both would be that mistake again."""
    assert 0.0 < tl.adopt_floor("rate") < 0.01
    assert tl.adopt_floor("equity") >= 0.1
    assert tl.adopt_floor("rate") != tl.adopt_floor("equity")
