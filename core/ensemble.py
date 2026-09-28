"""Ensemble: gating, stacking, weight selection."""

import numpy as np


def ensemble_with_gating(
    cb_prob: np.ndarray,
    lstm_prob: np.ndarray,
    trend_strength: np.ndarray,
    gate: float = 0.01,
    w_lstm_override: float | None = None,
) -> np.ndarray:
    """Soft-gated ensemble of CatBoost and LSTM predictions.

    Uses trend_strength to dynamically weight LSTM vs CatBoost.
    Higher trend - more LSTM weight. Disagreement between models
    pulls the prediction toward 0.5 (reduces confidence).

    Args:
        cb_prob: CatBoost predicted probabilities
        lstm_prob: LSTM predicted probabilities
        trend_strength: abs(close/sma50 - 1)
        gate: trend threshold for gating activation
        w_lstm_override: fixed LSTM weight (skips dynamic gating)

    Returns:
        Blended probability array
    """
    if w_lstm_override is not None:
        w_lstm = np.full_like(cb_prob, w_lstm_override, dtype=float)
    else:
        # Soft sigmoid gating (smooth transition instead of binary)
        w_lstm = 1.0 / (1.0 + np.exp(-10.0 * (trend_strength - gate)))
        w_lstm = 0.4 + 0.3 * w_lstm  # range: 0.4..0.7
    w_cb = 1.0 - w_lstm
    raw = w_lstm * lstm_prob + w_cb * cb_prob

    # Disagreement penalty: if models disagree, pull toward 0.5
    disagree = np.abs(lstm_prob - cb_prob)
    confidence = 1.0 - 0.3 * disagree
    return 0.5 + (raw - 0.5) * confidence


def tune_ensemble_weights(
    cb_val: np.ndarray,
    lstm_val: np.ndarray,
    y_val: np.ndarray,
    trend_val: np.ndarray,
    gate: float,
    lstm_acc: float | None = None,
) -> tuple[float, float]:
    """Grid-search optimal LSTM weight on validation set.

    Returns (best_weight, best_accuracy).
    """
    if lstm_acc is not None and lstm_acc < 0.52:
        max_lstm_w = max(0.15, (lstm_acc - 0.45) * 5.0)
        max_lstm_w = min(max_lstm_w, 0.40)
        min_lstm_w = 0.10
    else:
        max_lstm_w = 0.80
        min_lstm_w = 0.20

    best_w, best_acc = None, -1.0
    for w_int in range(int(min_lstm_w * 100), int(max_lstm_w * 100) + 1, 5):
        w = w_int / 100.0
        prob = ensemble_with_gating(cb_val, lstm_val, trend_val, gate, w_lstm_override=w)
        acc = float(((prob >= 0.5).astype(int) == y_val).mean())
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w, best_acc


def build_stacking_features(
    cb_p: np.ndarray,
    lstm_p: np.ndarray,
    tf_p: np.ndarray,
    tcn_p: np.ndarray,
    trend: np.ndarray,
) -> np.ndarray:
    """Build meta-features for stacking classifier from 4 model outputs.

    Returns array with columns: [cb, lstm, tf, tcn, disagreement, mean_prob, trend]
    """
    stack = np.column_stack([cb_p, lstm_p, tf_p, tcn_p])
    disagree = np.std(stack, axis=1, keepdims=True)
    mean_prob = np.mean(stack, axis=1, keepdims=True)
    trend_col = np.array(trend).reshape(-1, 1)
    return np.hstack([stack, disagree, mean_prob, trend_col])


def fixed_mix(cb, lstm, tf, tcn):
    """Half CatBoost, half the mean of the nets that produced a number.

    Nothing here is fitted. On 20 assets / 150 folds (2026-09-28) every
    combiner fitted on the validation slice (the stacker, Platt, a stronger
    penalty) lost accuracy to unfitted mixes: the validation window's up-rate
    differs from the test window's by 2.5 pts on average, and a fitted
    intercept follows it. This mix beat the stacker by +1.1 pts (16 of 20) with
    the best AUC. A net that returned NaN or nothing is left out rather than
    poisoning the mix; with no net at all it is CatBoost alone.
    Works on scalars and on equal-length arrays.
    """
    nets = [n for n in (lstm, tf, tcn) if n is not None]
    if np.ndim(cb) == 0:
        vals = [float(n) for n in nets if np.isfinite(n)]
        return float(cb) if not vals else 0.5 * float(cb) + 0.5 * float(np.mean(vals))
    cb = np.asarray(cb, dtype=float)
    if not nets:
        return cb
    stack = np.vstack([np.asarray(n, dtype=float) for n in nets])
    with np.errstate(invalid="ignore"):
        net_mean = np.nanmean(stack, axis=0)
    return np.where(np.isnan(net_mean), cb, 0.5 * cb + 0.5 * net_mean)


def combine_fold(combiner, val_members, val_trend, val_target, test_members, test_trend):
    """(val_prob, test_prob, meta_clf) for one training fold.

    `*_members` are [cb, lstm, tf, tcn] arrays. "fixed" fits nothing and
    returns meta_clf None; "stack" is the historical per-fold logistic
    stacker fitted on the validation slice.
    """
    if combiner == "fixed":
        return fixed_mix(*val_members), fixed_mix(*test_members), None
    from sklearn.linear_model import LogisticRegression

    X_val = build_stacking_features(*val_members, val_trend)
    X_test = build_stacking_features(*test_members, test_trend)
    meta = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs").fit(X_val, val_target)
    return meta.predict_proba(X_val)[:, 1], meta.predict_proba(X_test)[:, 1], meta
