"""Tests for core.calibration - isotonic probability calibration."""

import numpy as np

from core.calibration import (
    apply_calibrator,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
)


def test_identity_when_none():
    probs = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(apply_calibrator(None, probs), probs)


def test_returns_none_on_single_class():
    probs = np.linspace(0.3, 0.7, 100)
    targets = np.ones(100, dtype=int)
    assert fit_calibrator(probs, targets) is None


def test_returns_none_on_too_few_samples():
    assert fit_calibrator([0.4, 0.6], [0, 1]) is None


def test_calibration_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    # Overconfident raw scores: true freq grows with score but compressed.
    probs = rng.uniform(0, 1, 500)
    targets = (rng.uniform(0, 1, 500) < probs).astype(int)
    calib = fit_calibrator(probs, targets)
    assert calib is not None
    grid = np.linspace(0, 1, 50)
    out = apply_calibrator(calib, grid)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Isotonic output must be non-decreasing.
    assert np.all(np.diff(out) >= -1e-9)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 300)
    targets = (rng.uniform(0, 1, 300) < probs).astype(int)
    calib = fit_calibrator(probs, targets)
    save_calibrator(calib, str(tmp_path), "foo")
    loaded = load_calibrator(str(tmp_path), "foo")
    assert loaded is not None
    np.testing.assert_allclose(
        apply_calibrator(calib, probs), apply_calibrator(loaded, probs)
    )


def test_save_none_is_noop(tmp_path):
    assert save_calibrator(None, str(tmp_path), "foo") is None
    assert load_calibrator(str(tmp_path), "foo") is None
