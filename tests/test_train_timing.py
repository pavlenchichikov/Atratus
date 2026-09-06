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

    Pinned to the score objective, because that is the only place the sentinel
    exists: the default `net` objective measures a rate, a thin asset has a
    perfectly defined one, so there is nothing to drop. Both halves are worth
    keeping - the score path is still reachable and still carries the hazard.
    """

    @pytest.fixture(autouse=True)
    def _score_objective(self, monkeypatch):
        monkeypatch.setattr(tt, "OBJECTIVE", "score")

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


class TestNetObjectiveHasNoSentinel:
    """Under the default objective a thin asset is measured, not discarded.

    The positive control for the class above: the same asset that is unscorable
    in score units carries a finite rate, so it enters the mean instead of being
    dropped, and no -999 can reach mean_d in the first place.
    """

    def test_thin_asset_is_scorable_as_a_rate(self):
        cand = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.06})
        thin = _quiet_series(70, seed=7, spread=0.05)
        assert (tt.eval_policy(thin, cand)["score"]
                == pytest.approx(bt.UNRELIABLE_SCORE))
        assert np.isfinite(tt.eval_policy(thin, cand)["rate"])

        healthy = {"A%d" % i: _series(500, seed=i) for i in range(20)}
        both = dict(healthy, NEW_SHORT=thin)
        out = tt.gate_policy(both, cand)
        assert out["objective"] == "net"
        assert out["n_unscorable"] == 0
        assert out["n"] == len(both)
        assert "NEW_SHORT" in out["per_asset"]

    def test_floor_is_in_the_units_the_gate_measures(self):
        assert tt.adopt_floor("net") == tt.ADOPT_FLOOR_NET
        assert tt.adopt_floor("score") == tt.ADOPT_FLOOR_SCORE
        assert tt.adopt_floor("net") < tt.adopt_floor("score")


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


def test_stage_b_reports_both_of_its_silent_passes(capsys, monkeypatch):
    """The ladder reports itself and takes five minutes; these two loops replay
    every candidate over every asset and took the ninety after it without a
    single line, which cannot be told apart from a hang.

    One line per candidate per pass, so six rungs give six and six. The counts
    are the assertion: a print inside gate_challenger only, or only on val,
    would leave half the wait dark again."""
    import train_timing as tt
    from core import timing_fqi as fq
    from core import timing_policy as tp

    def _fqi_series(seed):
        # _series above predates the FQI path and carries no `close`, which
        # series_features needs for its trend feature.
        s = _series(n=300, seed=seed)
        s["close"] = 100.0 * np.cumprod(1.0 + np.nan_to_num(s["next_ret"]))
        s["is_forex"] = False
        return s

    by_asset = {"A": _fqi_series(1), "B": _fqi_series(2)}
    rules = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))

    # Two rungs, not six: the point is the reporting, and fitting a real ladder
    # here would make this a slow test for no extra coverage.
    def two_rungs(models):
        return [("q_iter_1", fq.FqiPolicy(models[0])),
                ("q_iter_2", fq.FqiPolicy(models[-1]))]

    monkeypatch.setattr(fq, "fit_q", lambda batches, **k: [
        _ConstQ(0.0), _ConstQ(1.0)])
    tt.stage_b(by_asset, iters=2, seed=0, challenger_factory=two_rungs,
               reference=rules)
    out = capsys.readouterr().out
    # "/2" pins it to the per-candidate lines: "[stage-b] val picks ..." also
    # starts with "[stage-b] val " and would inflate the count to three.
    assert out.count("[stage-b] val  1/2") == 1, out
    assert out.count("[stage-b] val  2/2") == 1, out
    assert out.count("[stage-b] test 1/2") == 1, out
    assert out.count("[stage-b] test 2/2") == 1, out
    assert "val picks" in out and "[stage-b] gate " in out


class _ConstQ:
    """A stand-in CatBoost model: predict() returns one constant per row."""

    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        import numpy as _np

        return _np.full(len(rows), self.value, dtype=float)


class TestTheSearchRestartsInsteadOfStalling:
    """A collapsed ES must be reopened, not resampled 340 more times.

    Measured 2026-09-06: CmaEmitter's rank-mu update ranks noise on a flat
    landscape, so sigma falls from 0.25 of each span to its 0.01 floor by
    evaluation 60. The 400-evaluation fit of 2026-09-05 consequently returned
    its SIXTH candidate - a draw made before the first adaptation, identical to
    the one a 12-asset run of the same seed produced.
    """

    def test_a_collapsed_sigma_is_reopened_around_the_best(self):
        import random

        from core.ar_rl import CmaEmitter
        es = CmaEmitter(rng=random.Random(0), dims=tp.PARAM_SPECS)
        es.seed_from(tt._P(dict(tp.DEFAULT_PARAMS)))
        spans = [hi - lo for _n, lo, hi, _i in tp.PARAM_SPECS]

        es.sigma = [0.01 * s for s in spans]      # the collapsed state
        best = {**tp.DEFAULT_PARAMS, "min_hold_days": 4}
        tt._reopen(es, best)

        assert es.sigma == [tt.RESTART_SIGMA_FRAC * s for s in spans]
        assert es.evals == []
        assert dict(zip([n for n, _l, _h, _i in tp.PARAM_SPECS], es.mean)
                    )["min_hold_days"] == 4

    def test_a_stalled_search_actually_restarts(self, monkeypatch):
        """Fitness that never improves must trigger restarts, not silence."""
        calls = []
        monkeypatch.setattr(tt, "_reopen",
                            lambda es, around: calls.append(dict(around)))
        monkeypatch.setattr(tt, "eval_policy", lambda s, p: {"rate": 0.0, "score": 0.0})
        monkeypatch.setattr(tt, "objective_of", lambda stats, objective=None: 0.0)

        tt.fit_policy({"A": {}}, budget=25, patience=10, val_by_asset={"A": {}})
        assert len(calls) == 2, "25 evaluations at patience 10 is two restarts"

    def test_patience_zero_reproduces_the_old_single_shot_search(self, monkeypatch):
        """The control: every measurement made before 2026-09-06 must still be
        reachable, so a run that never restarts must never call _reopen."""
        calls = []
        monkeypatch.setattr(tt, "_reopen",
                            lambda es, around: calls.append(around))
        monkeypatch.setattr(tt, "eval_policy", lambda s, p: {"rate": 0.0, "score": 0.0})
        monkeypatch.setattr(tt, "objective_of", lambda stats, objective=None: 0.0)

        tt.fit_policy({"A": {}}, budget=25, patience=0, val_by_asset={"A": {}})
        assert calls == []
