"""Probability that a session reaches its level, from the distance to it.

The measurement this implements (2026-09-12, 492 assets, held-out year):

    distance only, no fitting   AUC 0.6824
    pooled CatBoost on 21 cols  AUC 0.7390
    shuffled-label pedestal     AUC 0.54 to 0.60
    coin flip                   AUC 0.5

The model's margin over the distance rule is +0.022 by median, which sits inside
the band a shuffled-label fit reaches by pooling assets with different base
rates. So the rule is what gets built and the model does not: one number, no
training, and nothing a pedestal can inflate.

What is fitted here is only the CALIBRATION - turning "1.4 hourly sigmas away"
into "reached on 38% of such days" - because a level sheet has to state odds,
not a ranking. The curve is pooled and each asset gets an intercept of its own,
since the same distance means different odds on a name that gaps and one that
crawls. Everything is fitted on train sessions and applied to later ones.
"""
from itertools import pairwise

import numpy as np

MIN_ASSET_SESSIONS = 60          # below this an asset keeps the pooled curve
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_reach(train_by_asset):
    """Calibration from {asset: (distance array, label array)} on TRAIN only.

    Returns {"w", "b", "offsets"}: one pooled slope and intercept over the
    distance, plus a per-asset intercept shift. The shift is a moment match -
    it moves the asset's mean predicted probability onto its own realised rate -
    which needs one number per asset instead of a second regression, and cannot
    overfit a shape it does not have.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.concatenate([np.asarray(d, dtype=float) for d, _ in train_by_asset.values()])
    y = np.concatenate([np.asarray(t, dtype=int) for _, t in train_by_asset.values()])
    if len(np.unique(y)) < 2:
        raise ValueError("fit_reach needs both outcomes in train")
    lr = LogisticRegression(C=1.0, max_iter=200)
    lr.fit(x.reshape(-1, 1), y)
    w, b = float(lr.coef_[0][0]), float(lr.intercept_[0])

    offsets = {}
    for asset, (dist, lab) in train_by_asset.items():
        dist = np.asarray(dist, dtype=float)
        lab = np.asarray(lab, dtype=int)
        if len(lab) < MIN_ASSET_SESSIONS:
            continue
        base = float(lab.mean())
        if base <= 0.0 or base >= 1.0:
            continue
        pooled = float(_sigmoid(w * dist + b).mean())
        offsets[asset] = float(_logit(base) - _logit(pooled))
    return {"w": w, "b": b, "offsets": offsets}


def reach_probability(model, asset, distance):
    """P(the session reaches the level) for one bar or an array of them.

    `distance` is signed toward the level in units of the current hourly
    volatility: positive means the level is still ahead, which is how
    core.intraday_labels builds f_lvl_dist.
    """
    z = model["w"] * np.asarray(distance, dtype=float) + model["b"]
    z = z + model["offsets"].get(asset, 0.0)
    # Clamped, and not for tidiness: a few sigmas out the sigmoid saturates to
    # exactly 1.0 in float64, and a sheet quoting "100%" claims a certainty the
    # curve cannot have - it has only run off the end of the range it was fitted
    # on. The bound is the same one _logit works in.
    out = np.clip(_sigmoid(z), _EPS, 1 - _EPS)
    return float(out) if np.ndim(out) == 0 else out


def brier(y, p):
    """Mean squared error of a probability. Lower is better; 0.25 is a coin."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability(y, p, buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    """[(lo, hi, n, mean predicted, realised frequency)] per probability band.

    A probability that ranks well can still lie about the odds, and a level
    sheet quotes the odds. This is the table that says whether it does.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = []
    for lo, hi in pairwise(buckets):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        out.append((float(lo), float(min(hi, 1.0)), int(m.sum()),
                    float(p[m].mean()), float(y[m].mean())))
    return out
