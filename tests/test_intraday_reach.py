"""The reach probability is a calibration, not a model: one number in, odds out.
These pin the properties a level sheet depends on."""
import numpy as np
import pytest

from core.intraday_reach import (
    MIN_ASSET_SESSIONS,
    brier,
    fit_reach,
    reach_probability,
    reliability,
)


def _asset(n=400, base_dist=1.0, seed=0, rate=None):
    """Distances and outcomes where nearer levels are reached more often."""
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0.2, 3.0, n) * base_dist
    p = 1.0 / (1.0 + np.exp(2.0 * (dist - 1.5)))
    if rate is not None:
        p = np.clip(p + (rate - p.mean()), 0.01, 0.99)
    return dist, (rng.random(n) < p).astype(int)


def test_nearer_levels_get_higher_odds():
    model = fit_reach({"A": _asset()})
    near = reach_probability(model, "A", 0.3)
    far = reach_probability(model, "A", 2.8)
    assert 0.0 < far < near < 1.0


def test_probabilities_stay_inside_the_unit_interval():
    model = fit_reach({"A": _asset()})
    p = reach_probability(model, "A", np.array([-50.0, 0.0, 50.0]))
    assert np.all((p > 0.0) & (p < 1.0))


def test_an_asset_that_reaches_more_often_is_shifted_up():
    """The same distance means different odds on a name that gaps and one that
    crawls, which is what the per-asset intercept is for."""
    model = fit_reach({"CALM": _asset(seed=1, rate=0.25),
                       "JUMPY": _asset(seed=2, rate=0.75)})
    assert model["offsets"]["JUMPY"] > model["offsets"]["CALM"]
    assert (reach_probability(model, "JUMPY", 1.0)
            > reach_probability(model, "CALM", 1.0))


def test_a_thin_asset_keeps_the_pooled_curve():
    """An intercept fitted on a handful of sessions is noise with a name."""
    model = fit_reach({"BIG": _asset(seed=3),
                       "THIN": _asset(n=MIN_ASSET_SESSIONS - 1, seed=4)})
    assert "THIN" not in model["offsets"]
    assert reach_probability(model, "THIN", 1.0) == pytest.approx(
        reach_probability(model, "UNKNOWN_ASSET", 1.0))


def test_fitting_needs_both_outcomes():
    dist = np.linspace(0.2, 3.0, 100)
    with pytest.raises(ValueError):
        fit_reach({"A": (dist, np.zeros(100, dtype=int))})


def test_brier_rewards_being_right_and_sure():
    y = np.array([1, 1, 0, 0])
    assert brier(y, np.array([0.9, 0.9, 0.1, 0.1])) < brier(y, np.array([0.5] * 4))
    assert brier(y, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0


def test_reliability_reports_predicted_against_realised():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    table = reliability(y, p)
    low = next(row for row in table if row[0] == 0.0)
    high = next(row for row in table if row[0] == 0.8)
    assert low[2] == 3 and low[3] == pytest.approx(0.1) and low[4] == pytest.approx(1 / 3)
    assert high[2] == 3 and high[4] == pytest.approx(1.0)


def test_a_calibrated_fit_beats_a_flat_guess_on_its_own_data():
    dist, y = _asset(n=2000, seed=5)
    model = fit_reach({"A": (dist, y)})
    p = reach_probability(model, "A", dist)
    assert brier(y, p) < brier(y, np.full(len(y), y.mean()))
