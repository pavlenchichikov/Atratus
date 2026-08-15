"""Scaler shared between training and inference.

Training saves the train-fold StandardScaler next to the champion, and inference
must reuse the same one - otherwise there is train/serve skew. If none is saved
(old models), we fit on the current window.
"""

import os

import joblib
from sklearn.preprocessing import StandardScaler


def scaler_path(model_dir: str, table: str) -> str:
    """Path of the fitted StandardScaler persisted next to the champion."""
    suffix = "_scaler.pkl"
    return os.path.join(model_dir, f"{table}{suffix}")


def save_scaler(scaler, model_dir: str, table: str) -> str:
    """Persist a fitted scaler next to the champion model. Returns the path."""
    path = scaler_path(model_dir, table)
    joblib.dump(scaler, path)
    return path


def load_scaler(model_dir: str, table: str, n_cols=None):
    """Load a saved scaler, or None. Unlike load_or_fit_scaler this NEVER falls
    back to fitting: for the net scaler a silent refit on the serving window is
    exactly the train/serve skew this file exists to prevent, and the caller must
    be able to tell that the artifact is missing."""
    path = scaler_path(model_dir, table)
    if not os.path.exists(path):
        return None
    try:
        scaler = joblib.load(path)
    except Exception:
        return None
    saved_n = getattr(scaler, "n_features_in_", None)
    if n_cols is not None and saved_n is not None and saved_n != n_cols:
        return None
    return scaler


def load_or_fit_scaler(model_dir: str, table: str, x_fit):
    """Return (scaler, source).

    source is "saved" when a matching saved scaler was loaded, "fit" when a fresh
    scaler was fitted on x_fit. A saved scaler whose feature count does not match
    x_fit is rejected (returns a freshly fitted one) so dimension drift never
    produces a silent transform error downstream.
    """
    path = scaler_path(model_dir, table)
    n_cols = x_fit.shape[1] if hasattr(x_fit, "shape") and len(x_fit.shape) == 2 else None
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            saved_n = getattr(scaler, "n_features_in_", None)
            if saved_n is None or n_cols is None or saved_n == n_cols:
                return scaler, "saved"
        except Exception:
            pass
    scaler = StandardScaler()
    scaler.fit(x_fit)
    return scaler, "fit"
