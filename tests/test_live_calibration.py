"""The live calibration layer, and why it never installed.

Isotonic can only fit a non-decreasing map. The live stream is anti-calibrated,
so the best non-decreasing fit of it is a flat line, the spread guard correctly
refused to ship that, and the layer therefore did nothing for the whole live
window. These tests pin the two things that changed: a family that CAN express
a decreasing map, and a guard about whether the output still reaches a decision
threshold rather than about spread in the abstract.
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import recalibrate_live as rl
from core.calibration import PlattCalibrator, fit_platt, log_loss


def test_the_default_platt_map_is_the_identity():
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(PlattCalibrator().predict(p), p, atol=1e-6)


def test_a_negative_slope_turns_the_map_upside_down():
    p = np.array([0.2, 0.8])
    out = PlattCalibrator(a=-1.0).predict(p)
    assert out[0] > out[1]


def test_fit_platt_reports_a_negative_slope_on_an_inverted_stream():
    """The positive control for the whole change: on data where the confident
    predictions are the wrong ones, the fit has to be able to SAY so. Isotonic
    cannot, which is why nothing was ever installed."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.05, 0.95, 2000)
    # went up with probability 1 - p: perfectly inverted confidence
    ups = (rng.uniform(size=2000) < (1.0 - probs)).astype(int)
    model = fit_platt(probs, ups)
    assert model is not None and model.a < 0


def test_fit_platt_reports_a_positive_slope_on_an_honest_stream():
    """The other end: well-calibrated data must not come back inverted."""
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.05, 0.95, 2000)
    ups = (rng.uniform(size=2000) < probs).astype(int)
    model = fit_platt(probs, ups)
    assert model is not None and model.a > 0


def test_log_loss_cannot_be_improved_by_flattening_to_the_base_rate():
    """Why the choice is made on log loss and not accuracy: a constant map has
    to be scoreable and must not look good."""
    y = [1, 0, 1, 0]
    honest = log_loss([0.9, 0.1, 0.9, 0.1], y)
    flat = log_loss([0.5, 0.5, 0.5, 0.5], y)
    assert honest < flat


def test_a_map_that_reaches_no_threshold_is_refused():
    """A layer that puts every asset in a narrow band around the base rate
    never crosses a decision threshold, the whole book prints WAIT, and that is
    a kill switch wearing the word calibration. The old spread test of 0.02 let
    a band 0.055 wide through."""
    probs = np.linspace(0.05, 0.95, 500)
    # a uniform 0.05-0.95 spread leaves 11 percent inside the neutral band
    assert rl.crossing_share(rl._Identity(), probs) > 0.85
    flat = PlattCalibrator(a=-0.037, b=0.0)      # the real fitted slope
    assert rl.crossing_share(flat, probs) < rl.MIN_CROSSING


def test_the_split_is_on_the_date_and_refuses_when_it_cannot_be():
    """Rows from one day are correlated across assets, so a random split would
    put the same day on both sides and flatter every candidate equally."""
    probs = [0.6] * 600
    ups = [1, 0] * 300
    dates = ["2026-08-%02d" % (1 + i % 10) for i in range(600)]
    fit, held, cut = rl.split_in_time(probs, ups, dates)
    assert len(fit) + len(held) == 600 and cut == "2026-08-07"
    assert rl.split_in_time(probs, ups, ["2026-08-01"] * 600) is None


def test_nothing_is_chosen_when_nothing_beats_leaving_it_alone():
    """A layer level with identity is a layer nobody should be maintaining."""
    rng = np.random.default_rng(2)
    probs = rng.uniform(0.4, 0.6, 800)
    ups = (rng.uniform(size=800) < probs).astype(int)
    rows = list(zip(probs, ups))
    _report, best, _base = rl.choose(rows[:400], rows[400:])
    assert best is None or best["loss"] < _base
