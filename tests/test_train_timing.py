"""Tests for train_timing.py (dataset-free parts use synthetic series)."""
import os

import numpy as np
import pytest

import train_timing as tt
from core import backtesting as bt
from core import timing_policy as tp


def _series(n=120, seed=3):
    rng = np.random.default_rng(seed)
    probs = np.clip(0.5 + rng.normal(0, 0.08, n), 0.05, 0.95)
    next_ret = rng.normal(0.0005, 0.01, n)
    next_ret[-1] = np.nan
    return {
        "probs": probs, "next_ret": next_ret,
        "atr": np.full(n, 0.015), "taleb_hi": np.zeros(n, dtype=bool),
        "buy_thr": 0.55, "sell_thr": 0.45, "risky": False,
        "dates": np.arange(n),
    }


class TestEvalPolicy:
    def test_baseline_equals_default_policy(self):
        s = _series()
        base = tt.eval_baseline(s)
        pol = tt.eval_policy(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)))
        assert pol["score"] == pytest.approx(base["score"])
        assert pol["n_trades"] == base["n_trades"]

    def test_stricter_entry_trades_less(self):
        s = _series()
        strict = tt.eval_policy(
            s, tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.08}))
        base = tt.eval_baseline(s)
        assert strict["n_trades"] <= base["n_trades"]

    def test_forex_costs_used(self):
        s = _series()
        s["risky"] = True
        s["is_forex"] = True
        r = tt.eval_baseline(s)
        s2 = _series()
        s2["is_forex"] = False
        r2 = tt.eval_baseline(s2)
        # same series, cheaper forex legs -> profit no worse
        assert r["profit"] >= r2["profit"]


class TestFitness:
    def test_median_minus_iqr(self):
        scores = [1.0, 2.0, 3.0, 4.0, 100.0]
        med = np.median(scores)
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        assert tt.fitness(scores) == pytest.approx(med - 0.25 * iqr)

    def test_empty_is_minus_inf(self):
        assert tt.fitness([]) == float("-inf")


class TestSplitFitGate:
    def _series_by_asset(self, k=8, n=240):
        out = {}
        for i in range(k):
            s = _series(n=n, seed=i)
            out[f"A{i}"] = s
        return out

    def test_split_is_time_ordered(self):
        s = _series(n=100)
        tr, va, te = tt.split_series(s)
        assert len(tr["probs"]) == 60 and len(va["probs"]) == 20
        assert len(te["probs"]) == 20
        assert tr["buy_thr"] == s["buy_thr"]

    def test_fit_returns_valid_params(self):
        data = self._series_by_asset()
        params = tt.fit_policy(
            {a: tt.split_series(s)[0] for a, s in data.items()},
            budget=30, seed=7)
        for name, lo, hi, is_int in tp.PARAM_SPECS:
            assert lo <= params[name] <= hi
            if is_int:
                assert float(params[name]).is_integer()

    def test_gate_hold_on_noise(self):
        data = self._series_by_asset()
        test_slices = {a: tt.split_series(s)[2] for a, s in data.items()}
        verdict = tt.gate_policy(test_slices, dict(tp.DEFAULT_PARAMS))
        # default params ARE the baseline -> all deltas 0 -> never ADOPT
        assert verdict["verdict"] == "HOLD"

    def test_save_policy_only_on_adopt(self, tmp_path):
        p = str(tmp_path / "timing_policy.json")
        tt.save_policy(dict(tp.DEFAULT_PARAMS),
                       {"verdict": "HOLD", "per_asset": {}}, path=p)
        assert not os.path.exists(p)
        tt.save_policy(dict(tp.DEFAULT_PARAMS),
                       {"verdict": "ADOPT", "per_asset": {}}, path=p)
        assert os.path.exists(p)


def _quiet_series(n, seed, spread):
    """A short, thin slice - the shape a recently listed asset has."""
    rng = np.random.default_rng(seed)
    probs = np.clip(0.5 + rng.normal(0, spread, n), 0.05, 0.95)
    nr = rng.normal(0.0015, 0.01, n)
    nr[-1] = np.nan
    return {"probs": probs, "next_ret": nr, "atr": np.full(n, 0.015),
            "taleb_hi": np.zeros(n, dtype=bool), "buy_thr": 0.55,
            "sell_thr": 0.45, "risky": False, "dates": np.arange(n)}


class TestUnscorableAssetsAreNotAveraged:
    """score_strategy returns -999 when an arm trades fewer than min_trades.

    That marker is missing data, not a score. Averaged into mean_d - which is
    what the ADOPT floor is compared against - one such asset out of twenty
    moves the effect size by tens of points while the rank test, which cannot
    see magnitude, keeps reporting the same p. Short-history assets are what
    make this reachable, and the asset list grew by 116 of them on 2026-08-21.
    """

    def _healthy(self, k=20, n=500):
        return {"A%d" % i: _series(n, seed=i) for i in range(k)}

    def test_one_thin_asset_cannot_decide_the_verdict(self):
        cand = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.06})
        healthy = self._healthy()
        clean = tt.gate_policy(healthy, cand)

        thin = dict(healthy)
        thin["NEW_SHORT"] = _quiet_series(70, seed=7, spread=0.05)
        dirty = tt.gate_policy(thin, cand)

        # the asset really is unscorable for one arm and not the other
        assert (tt.eval_policy(thin["NEW_SHORT"], cand)["score"]
                == pytest.approx(bt.UNRELIABLE_SCORE))
        assert (tt.eval_baseline(thin["NEW_SHORT"])["score"]
                != pytest.approx(bt.UNRELIABLE_SCORE))

        assert dirty["mean_d"] == pytest.approx(clean["mean_d"])
        assert dirty["verdict"] == clean["verdict"]
        assert dirty["n"] == clean["n"], "an unscorable asset must not count"
        assert dirty["n_unscorable"] == 1
        assert "NEW_SHORT" not in dirty["per_asset"]

    def test_both_arms_unscorable_is_dropped_not_counted_as_a_tie(self):
        cand = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.06})
        healthy = self._healthy()
        clean = tt.gate_policy(healthy, cand)

        both = dict(healthy)
        both["FLAT"] = _quiet_series(60, seed=99, spread=0.002)
        out = tt.gate_policy(both, cand)
        assert out["n_unscorable"] == 1
        assert out["mean_d"] == pytest.approx(clean["mean_d"])

    def test_control_no_unscorable_asset_changes_nothing(self):
        """The gate must be unchanged for every measurement already recorded."""
        cand = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.02})
        healthy = self._healthy()
        out = tt.gate_policy(healthy, cand)
        deltas = [tt.eval_policy(s, cand)["score"] - tt.eval_baseline(s)["score"]
                  for s in healthy.values()]
        assert all(d != pytest.approx(bt.UNRELIABLE_SCORE) for d in deltas)
        assert out["n"] == len(healthy)
        assert out["n_unscorable"] == 0
        assert out["mean_d"] == pytest.approx(float(np.mean(deltas)))


class TestRefusesToGateNothing:
    """A fit over no scorable asset must say so, not answer.

    Before this, train_timing and train_sizing ran a full ES over an empty set,
    reported fitness -inf, and printed "verdict: HOLD p=1.0000 mean_d=+0.00
    n=0" - a verdict shaped exactly like a measurement. Stage B did not get
    that far: np.concatenate raised ValueError on the empty batch list. Both
    are reachable by pointing --assets at names whose champions do not exist,
    which is what the whole 116-asset backlog looks like right now.
    """

    def test_too_few_assets_is_refused(self, capsys):
        assert tt.require_scorable({}, "timing") is False
        out = capsys.readouterr().out
        assert "only 0 asset" in out
        assert "model_health.py --missing" in out, "say how to fix it"

    def test_one_short_of_the_gate_minimum_is_refused(self):
        series = {"A%d" % i: _series(120, seed=i)
                  for i in range(tt.GATE_MIN_ASSETS - 1)}
        assert tt.require_scorable(series, "timing") is False

    def test_exactly_the_minimum_is_allowed(self):
        series = {"A%d" % i: _series(120, seed=i)
                  for i in range(tt.GATE_MIN_ASSETS)}
        assert tt.require_scorable(series, "timing") is True
