"""Serve-time scoring must use the saved train-fold scaler and calibrator, not
refit on the inference window (train/serve skew). The scoring block now lives in
core/scoring.py and is shared by predict.py and alert_bot.py, so the guarantees
are asserted there; predict.py must delegate to it rather than re-implement it.
"""

import os

import numpy as np

from core.scaling import load_or_fit_scaler, save_scaler
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(__file__)
PREDICT_SRC = os.path.join(_HERE, "..", "predict.py")
SCORING_SRC = os.path.join(_HERE, "..", "core", "scoring.py")
ALERT_SRC = os.path.join(_HERE, "..", "alert_bot.py")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_scoring_uses_shared_scaler_loader():
    assert "load_or_fit_scaler" in _read(SCORING_SRC)


def test_scoring_does_not_refit_scaler_on_window():
    src = _read(SCORING_SRC)
    assert "scaler.fit_transform(" not in src


def test_scoring_applies_calibrator():
    assert "apply_calibrator" in _read(SCORING_SRC)


def test_predict_delegates_to_shared_scoring():
    src = _read(PREDICT_SRC)
    assert "from core.scoring import score_asset" in src
    assert "score_asset(" in src
    # The duplicated scoring block is gone from predict.py.
    assert "scaler.fit_transform(" not in src
    assert "build_stacking_features" not in src


def test_alert_bot_delegates_to_shared_scoring():
    src = _read(ALERT_SRC)
    assert "from core.scoring import score_asset" in src
    assert "score_asset(" in src
    # The bot's old private 2-model scoring (CB+LSTM gating) is gone.
    assert "ensemble_with_gating" not in src
    assert "load_or_fit_scaler" not in src


def test_parity_saved_scaler_used_at_serve(tmp_path):
    """End-to-end: a scaler fit on 'train' is reused verbatim at 'serve' time."""
    train = np.random.randn(300, 6) * 4 + 2
    serve_window = np.random.randn(50, 6) * 4 + 2
    save_scaler(StandardScaler().fit(train), str(tmp_path), "xau")

    scaler, source = load_or_fit_scaler(str(tmp_path), "xau", serve_window)
    assert source == "saved"
    # Transform must match the saved scaler exactly (no re-fit on serve window).
    expected = StandardScaler().fit(train).transform(serve_window)
    np.testing.assert_allclose(scaler.transform(serve_window), expected)
