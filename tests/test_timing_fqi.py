"""The FQI timing challenger: state, transitions, Q, and the policy over it."""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import timing_fqi as fq
from core import timing_policy as tp


def _series(n=300, seed=0):
    """A synthetic asset series in the shape build_asset_series returns."""
    rng = np.random.default_rng(seed)
    probs = np.clip(0.5 + rng.normal(0, 0.06, n), 0.01, 0.99)
    next_ret = rng.normal(0, 0.01, n)
    close = 100.0 * np.cumprod(1.0 + next_ret)
    return {"probs": probs, "next_ret": next_ret,
            "atr": np.abs(rng.normal(1.0, 0.2, n)),
            "taleb_hi": rng.random(n) > 0.8,
            "close": close, "buy_thr": 0.55, "sell_thr": 0.45,
            "risky": False, "is_forex": False}


def test_the_state_row_is_the_state_vector_the_spec_names():
    feat = fq.series_features(_series())
    row = fq.state_row(feat, 250, pos=1, days_held=4, pnl_atr=0.8,
                       cool_left=0, streak=3, action=1)
    assert len(row) == len(fq.FEATURE_NAMES)
    assert fq.FEATURE_NAMES[-1] == "action"
    assert row[-1] == 1
    assert not any(np.isnan(v) for v in row)


def test_every_per_bar_feature_is_finite_from_the_first_bar():
    """A NaN in the first 200 bars would poison a CatBoost fit silently, and
    the warm-up is exactly where a rolling mean and a rolling rank are weakest."""
    feat = fq.series_features(_series())
    for name in ("prob", "prob_d1", "margin", "taleb_hi", "trend_up", "atr_pct"):
        assert np.isfinite(feat[name]).all(), name


def test_the_volatility_percentile_ranks_within_its_own_window():
    """Positive control: a bar carrying the largest ATR of its window must rank
    at the top, and one carrying the smallest at the bottom. Without this the
    feature could be a constant and nothing else in the suite would notice."""
    s = _series()
    s["atr"] = np.ones(len(s["probs"]))
    s["atr"][100] = 99.0          # the biggest in its window
    s["atr"][200] = 0.001         # the smallest in its window
    feat = fq.series_features(s)
    assert feat["atr_pct"][100] == 1.0
    assert feat["atr_pct"][200] < 0.1


def test_the_regime_flag_follows_the_long_mean():
    s = _series()
    s["close"] = np.arange(1.0, len(s["probs"]) + 1.0)      # monotone up
    feat = fq.series_features(s)
    assert feat["trend_up"][-1] == 1.0
    s["close"] = s["close"][::-1].copy()                    # monotone down
    feat = fq.series_features(s)
    assert feat["trend_up"][-1] == 0.0


def _feat_at(prob, taleb=0.0, atr=1.0):
    """A one-bar feature dict, enough for advance()."""
    return {"prob": np.array([prob]), "prob_d1": np.array([0.0]),
            "margin": np.array([prob - 0.55 if prob >= 0.5 else 0.45 - prob]),
            "taleb_hi": np.array([taleb]), "trend_up": np.array([1.0]),
            "atr_pct": np.array([0.5]), "atr": np.array([atr]),
            "risky": 0.0, "is_forex": 0.0, "buy_thr": 0.55, "sell_thr": 0.45,
            "n": 1}


def test_no_signal_forces_staying_out_whatever_the_model_wants():
    """Spec 4.2: the policy can never open a position against, or without, the
    ensemble signal. That is not a preference the model can outvote."""
    st = dict(tp.FRESH_STATE)
    assert fq.forced_action(raw=0, pos=0, cool_left=0) == fq.ACT_NO
    new, label, reason = fq.advance(st, _feat_at(0.50), 0, fq.ACT_YES)
    assert label == "STAY_OUT" and new["pos"] == 0 and reason == "forced"


def test_a_flipped_signal_forces_an_exit_and_never_a_reversal():
    st = dict(tp.FRESH_STATE, pos=1, days_held=5)
    new, label, reason = fq.advance(st, _feat_at(0.20), 0, fq.ACT_YES)
    assert label == "EXIT" and new["pos"] == 0 and reason == "forced"


def test_a_cooldown_bar_forces_staying_out():
    st = dict(tp.FRESH_STATE, cooldown_left=2)
    new, label, _r = fq.advance(st, _feat_at(0.90), 0, fq.ACT_YES)
    assert label == "STAY_OUT" and new["pos"] == 0
    assert new["cooldown_left"] == 1        # the clock still runs down


def test_where_the_spec_leaves_a_choice_the_action_decides_it():
    flat = dict(tp.FRESH_STATE)
    assert fq.forced_action(raw=1, pos=0, cool_left=0) is None
    entered, label, reason = fq.advance(flat, _feat_at(0.90), 0, fq.ACT_YES)
    assert label == "ENTER" and entered["pos"] == 1 and reason == "model"
    stayed, label, _r = fq.advance(flat, _feat_at(0.90), 0, fq.ACT_NO)
    assert label == "STAY_OUT" and stayed["pos"] == 0

    held = dict(tp.FRESH_STATE, pos=1, days_held=3)
    kept, label, _r = fq.advance(held, _feat_at(0.90), 0, fq.ACT_YES)
    assert label == "HOLD" and kept["pos"] == 1 and kept["days_held"] == 4
    left, label, _r = fq.advance(held, _feat_at(0.90), 0, fq.ACT_NO)
    assert label == "EXIT" and left["pos"] == 0


def test_a_rollout_returns_one_transition_per_bar_with_both_next_actions():
    import random
    s = _series(120)
    out = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                     random.Random(0), epsilon=0.0)
    n = len(s["probs"])
    assert out["rows"].shape == (n, len(fq.FEATURE_NAMES))
    assert out["next_rows"].shape == (n, 2, len(fq.FEATURE_NAMES))
    assert out["rewards"].shape == (n,)
    assert out["terminal"][-1] and not out["terminal"][0]


def test_holding_a_winning_bar_pays_and_a_losing_bar_costs():
    """The reward must be the position's own return, signed by the side."""
    import random
    s = _series(10)
    s["probs"] = np.full(10, 0.99)         # a permanent long signal
    s["next_ret"] = np.full(10, 0.02)
    out = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                     random.Random(0), epsilon=0.0)
    assert out["rewards"][1] > 0           # bar 0 enters, bar 1 is in position
    s["next_ret"] = np.full(10, -0.02)
    out = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                     random.Random(0), epsilon=0.0)
    assert out["rewards"][1] < 0


def test_a_position_change_is_charged_and_holding_is_not():
    """Positive control on the costs: with a cost of 10 percent per side change
    the entry bar must be worse than the same bar with no costs, and a bar that
    only holds must be identical."""
    import random
    s = _series(10)
    s["probs"] = np.full(10, 0.99)
    s["next_ret"] = np.zeros(10)
    free = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                      random.Random(0), epsilon=0.0, costs=(0.0, 0.0))
    dear = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                      random.Random(0), epsilon=0.0, costs=(0.05, 0.05))
    assert dear["rewards"][0] < free["rewards"][0]
    assert dear["rewards"][3] == free["rewards"][3]


def test_exploration_changes_what_is_tried_and_zero_epsilon_does_not():
    import random
    s = _series(200)
    base = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                      random.Random(0), epsilon=0.0)
    same = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                      random.Random(7), epsilon=0.0)
    noisy = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                       random.Random(0), epsilon=0.5)
    assert base["labels"] == same["labels"]        # no noise means no rng use
    assert noisy["labels"] != base["labels"]


def test_q_learns_which_action_pays_in_an_environment_where_it_is_obvious():
    """The positive control for the whole fit. In this environment holding
    always pays and standing aside never does, so a Q that cannot rank those
    two apart is not fitting anything and every later verdict is noise."""
    import random
    s = _series(400, seed=3)
    s["probs"] = np.full(400, 0.99)        # a permanent long signal
    s["next_ret"] = np.full(400, 0.01)     # that always pays
    batch = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                       random.Random(1), epsilon=0.3)
    models = fq.fit_q([batch], iters=3, seed=0)
    assert len(models) == 3
    feat = fq.series_features(s)
    held = fq.state_row(feat, 300, pos=1, days_held=5, pnl_atr=0.5,
                        cool_left=0, streak=5, action=fq.ACT_YES)
    quit_ = fq.state_row(feat, 300, pos=1, days_held=5, pnl_atr=0.5,
                         cool_left=0, streak=5, action=fq.ACT_NO)
    q = models[-1]
    assert fq.q_value(q, held) > fq.q_value(q, quit_)


def test_a_shorter_fit_is_a_prefix_of_a_longer_one_at_the_same_seed():
    """Selection on VAL compares iteration counts, so iteration k must mean the
    same thing whether the loop was told to stop at k or later."""
    import random
    s = _series(200, seed=4)
    batch = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                       random.Random(1), epsilon=0.2)
    short = fq.fit_q([batch], iters=2, seed=11)
    long_ = fq.fit_q([batch], iters=4, seed=11)
    row = batch["rows"][50]
    assert abs(fq.q_value(short[1], row) - fq.q_value(long_[1], row)) < 1e-9


class _ConstQ:
    """A stand-in Q that always prefers the action given to it."""

    def __init__(self, prefer):
        self.prefer = prefer

    def predict(self, rows):
        rows = np.asarray(rows, dtype=float)
        want = rows[:, fq.FEATURE_NAMES.index("action")]
        return np.where(want == self.prefer, 1.0, 0.0)


def test_a_q_that_always_acts_enters_on_every_live_signal():
    s = _series(150, seed=5)
    sides, actions, _reasons = fq.FqiPolicy(_ConstQ(fq.ACT_YES)).apply_series(s)
    assert set(actions) <= {"ENTER", "HOLD", "STAY_OUT", "EXIT"}
    assert (sides != 0).any()


def test_a_q_that_never_acts_never_takes_a_position():
    """The other end of the control: an always-refuse Q must be flat forever,
    which is the one behaviour that cannot be produced by a bug in the state."""
    s = _series(150, seed=5)
    sides, actions, _r = fq.FqiPolicy(_ConstQ(fq.ACT_NO)).apply_series(s)
    assert (sides == 0).all()
    assert "ENTER" not in actions


def test_the_evaluator_routes_a_series_policy_through_apply_series():
    import train_timing as tt
    s = _series(150, seed=6)
    s["next_ret"] = np.abs(s["next_ret"])          # make holding pay
    out = tt.eval_policy(s, fq.FqiPolicy(_ConstQ(fq.ACT_YES)))
    assert "score" in out and out["n_trades"] >= 0
    # and the rules path is untouched
    rules = tt.eval_policy(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)))
    assert "score" in rules


class _Blocks:
    """A policy that holds `on` bars then stands aside `off` bars.

    Not a constant side: score_strategy returns its -999 sentinel below five
    trades, so a policy that never closes has no score to compare and every
    delta collapses to zero whatever the reference is.
    """

    def __init__(self, on=15, off=5, side=1):
        self.on, self.off, self.side = on, off, side
        self.params = {}

    def apply_series(self, series):
        n = len(series["probs"])
        cycle = self.on + self.off
        sides = np.array([self.side if (i % cycle) < self.on else 0
                          for i in range(n)], dtype=int)
        return sides, ["HOLD"] * n, ["ok"] * n


def _paying_assets(count, seed0):
    out = {}
    for k in range(count):
        s = _series(200, seed=seed0 + k)
        s["next_ret"] = np.abs(s["next_ret"])      # long always pays
        out["A%d" % k] = s
    return out


def test_the_gate_compares_against_the_reference_it_is_given():
    """Stage A already beat the baseline. A challenger measured against the
    baseline would inherit that win and prove nothing about replacing it."""
    import train_timing as tt
    by_asset = _paying_assets(10, 0)
    more = _Blocks(on=15, off=5)          # more time in a market that pays
    less = _Blocks(on=5, off=15)
    vs_flat = tt.gate_policy(by_asset, more, reference=less)
    vs_long = tt.gate_policy(by_asset, more, reference=more)
    assert vs_flat["mean_d"] > vs_long["mean_d"]
    assert vs_long["verdict"] == "HOLD"            # nothing beats itself


def test_benjamini_hochberg_is_applied_across_the_candidates():
    """Six iteration counts are six chances to look good once."""
    import train_timing as tt
    rows = tt.bh_rows([{"p": 0.01}, {"p": 0.02}, {"p": 0.03},
                       {"p": 0.04}, {"p": 0.20}, {"p": 0.90}])
    assert rows[0]["p_bh"] >= rows[0]["p"]
    assert [r["p_bh"] for r in rows] == sorted(r["p_bh"] for r in rows)


def test_the_report_measures_its_own_objective_against_the_gate():
    """The 2026-08-18 lesson, mechanised: a fit is only trustworthy when the
    thing it maximises moves with the thing that judges it."""
    import train_timing as tt
    by_asset = _paying_assets(12, 20)
    out = tt.objective_vs_gate(by_asset, _Blocks(on=15, off=5),
                               _Blocks(on=5, off=15))
    assert out["n"] == 12
    assert -1.0 <= out["rho"] <= 1.0
    # holding a series that only pays raises BOTH quantities on every asset
    assert out["sign_agree"] == 12


def test_stage_b_runs_end_to_end_and_reports_the_proxy_check():
    """A smoke over synthetic assets: it must produce a verdict, one row per
    candidate, and the objective-versus-gate reading beside them."""
    import train_timing as tt
    by_asset = {"A%d" % k: _series(260, seed=40 + k) for k in range(10)}
    out = tt.stage_b(by_asset, iters=2, gamma=0.9, epsilon=0.2, seed=0,
                     reference=tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)))
    assert out["verdict"] in ("ADOPT", "HOLD")
    assert len(out["rows"]) == 2
    assert set(out["proxy"]) == {"rho", "p", "sign_agree", "n"}
    assert out["reference"] == "stage_a"


def test_stage_b_never_adopts_on_a_draw():
    """Lose or draw and the rules stay. A challenger identical to the incumbent
    must not be able to replace it."""
    import train_timing as tt
    incumbent = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    by_asset = {"A%d" % k: _series(200, seed=60 + k) for k in range(9)}
    out = tt.stage_b(by_asset, iters=1, gamma=0.9, epsilon=0.0, seed=0,
                     reference=incumbent,
                     challenger_factory=lambda _m: [("copy", incumbent)])
    assert out["verdict"] == "HOLD"


class _StubQ:
    """A Q that prefers holding whenever the probability is past the threshold."""

    def predict(self, rows):
        rows = np.asarray(rows, dtype=float)
        prob = rows[0][fq.FEATURE_NAMES.index("prob")]
        yes = 1.0 if prob > 0.5 else -1.0
        return np.array([-yes, yes])


class TestServingStageB:
    """The serve path added when Stage B adopted on 2026-08-22.

    The property that matters is not that it runs: it is that the decision
    served from today's scalars is the decision the gate measured on the
    series. Anything else means the adopted number describes a policy that is
    not the one running.
    """

    def test_the_flag_is_off_unless_it_says_b(self, monkeypatch):
        monkeypatch.delenv("GTRADE_TIMING_STAGE", raising=False)
        assert fq.stage_b_on() is False
        for value, expected in (("b", True), ("B", True), (" b ", True),
                                ("a", False), ("1", False), ("", False)):
            monkeypatch.setenv("GTRADE_TIMING_STAGE", value)
            assert fq.stage_b_on() is expected, value

    def test_an_absent_or_broken_model_falls_back_to_none(self, tmp_path):
        assert fq.load_served_policy(str(tmp_path / "nope.cbm")) is None
        junk = tmp_path / "junk.cbm"
        junk.write_text("not a model", encoding="utf-8")
        assert fq.load_served_policy(str(junk)) is None

    def test_serve_step_reproduces_the_walk_the_gate_measured(self):
        """Bar by bar, from scalars only, against the series the fit uses."""
        series = _series(320, seed=5)
        pol = fq.FqiPolicy(_StubQ())
        feat = fq.series_features(series)
        st_fit = dict(tp.FRESH_STATE)
        st_srv = dict(tp.FRESH_STATE)
        for i in range(1, feat["n"]):
            action = pol.act(feat, i, st_fit)
            st_fit, label_fit, reason_fit = fq.advance(st_fit, feat, i, action)
            label_srv, reason_srv, st_srv = fq.serve_step(
                pol, series["probs"][i], series["probs"][i - 1],
                series["buy_thr"], series["sell_thr"],
                series["close"][:i + 1], series["atr"][:i + 1],
                bool(series["taleb_hi"][i]), series["risky"],
                series["is_forex"], st_srv)
            assert label_srv == label_fit, "bar %d" % i
            assert reason_srv == reason_fit, "bar %d" % i

    def test_a_missing_previous_probability_does_not_crash(self):
        series = _series(250, seed=1)
        pol = fq.FqiPolicy(_StubQ())
        label, reason, _st = fq.serve_step(
            pol, series["probs"][-1], None, series["buy_thr"],
            series["sell_thr"], series["close"], series["atr"],
            False, False, False, dict(tp.FRESH_STATE))
        assert label in ("ENTER", "STAY_OUT", "HOLD", "EXIT")
        assert reason in ("model", "forced")

    def test_two_bars_is_the_floor(self):
        pol = fq.FqiPolicy(_StubQ())
        import pytest
        with pytest.raises(ValueError):
            fq.serve_step(pol, 0.6, 0.5, 0.55, 0.45, [100.0], [1.0],
                          False, False, False, dict(tp.FRESH_STATE))


class TestReplayReadsDecisionsAgainstFacts:
    """The hit-or-miss reading, so a timing layer can be judged the way the
    signal is: not by a score, by what the next bar then did."""

    def _series_with(self, sides_wanted):
        """A series whose next_ret is +1% wherever the caller wants a hit."""
        n = len(sides_wanted)
        return {"probs": np.full(n, 0.60), "buy_thr": 0.55, "sell_thr": 0.45,
                "next_ret": np.asarray(sides_wanted, dtype=float) * 0.01,
                "atr": np.full(n, 0.01), "taleb_hi": np.zeros(n, dtype=bool),
                "close": np.full(n, 100.0), "risky": False, "is_forex": False}

    def test_a_long_that_rises_is_a_hit_and_one_that_falls_is_not(self):
        import train_timing as tt
        s = self._series_with([1, 1, -1, -1])       # up, up, down, down
        st = tt.hit_stats(s, [1, 1, 1, 1])
        assert st["held"] == 4
        assert st["held_hits"] == 2
        assert st["held_accuracy"] == 0.5

    def test_staying_out_of_a_trade_that_would_have_lost_is_a_hit(self):
        import train_timing as tt
        s = self._series_with([-1, -1])             # the bar falls
        st = tt.hit_stats(s, [0, 0])                # policy stayed flat
        assert st["held"] == 0
        assert st["flat_with_signal"] == 2, "the raw signal wanted long"
        assert st["avoided_losses"] == 2
        assert st["accuracy"] == 1.0

    def test_a_bar_nobody_wanted_is_not_counted_either_way(self):
        import train_timing as tt
        n = 3
        s = {"probs": np.full(n, 0.50), "buy_thr": 0.55, "sell_thr": 0.45,
             "next_ret": np.full(n, 0.01), "atr": np.full(n, 0.01),
             "taleb_hi": np.zeros(n, dtype=bool), "close": np.full(n, 100.0),
             "risky": False, "is_forex": False}
        st = tt.hit_stats(s, [0, 0, 0])
        assert st["decided"] == 0
        assert st["bars"] == 3, "the bars exist, they were just not a decision"

    def test_an_unresolved_bar_is_left_out(self):
        import train_timing as tt
        s = self._series_with([1, 1])
        s["next_ret"] = np.array([0.01, np.nan])
        st = tt.hit_stats(s, [1, 1])
        assert st["bars"] == 1 and st["held"] == 1 and st["held_hits"] == 1


def test_the_ladder_reports_every_rung_it_climbs(capsys):
    """A pooled fit over every asset is an hour with no output, which is
    indistinguishable from a hang. One line per rung is what tells the two
    apart, so the count of them is the thing worth asserting."""
    import random
    s = _series(120, seed=5)
    batch = fq.rollout(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)),
                       random.Random(1), epsilon=0.3)
    fq.fit_q([batch], iters=3, seed=0)
    out = capsys.readouterr().out
    assert out.count("[fqi]   iter") == 3, out
    assert "iter 3/3" in out and "transitions" in out
