"""Tests for core.scaling - train/serve scaler parity."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from core.scaling import load_or_fit_scaler, save_scaler, scaler_path


def test_fits_fresh_when_no_saved(tmp_path):
    x = np.random.randn(100, 5)
    scaler, source = load_or_fit_scaler(str(tmp_path), "foo", x)
    assert source == "fit"
    assert scaler.n_features_in_ == 5


def test_loads_saved_scaler(tmp_path):
    x_train = np.random.randn(200, 5) * 3 + 10
    saved = StandardScaler().fit(x_train)
    save_scaler(saved, str(tmp_path), "foo")
    assert scaler_path(str(tmp_path), "foo").endswith("foo_scaler.pkl")

    # A different inference window must still be transformed with the SAVED stats,
    # not re-fitted on the window (that is the train/serve skew we are preventing).
    x_infer = np.random.randn(30, 5) * 0.1 - 5
    scaler, source = load_or_fit_scaler(str(tmp_path), "foo", x_infer)
    assert source == "saved"
    np.testing.assert_allclose(scaler.mean_, saved.mean_)
    np.testing.assert_allclose(
        scaler.transform(x_infer), saved.transform(x_infer)
    )


def test_rejects_dimension_mismatch(tmp_path):
    saved = StandardScaler().fit(np.random.randn(100, 5))
    save_scaler(saved, str(tmp_path), "foo")
    # Inference now has 7 features: saved scaler is unusable, must refit.
    scaler, source = load_or_fit_scaler(str(tmp_path), "foo", np.random.randn(30, 7))
    assert source == "fit"
    assert scaler.n_features_in_ == 7
