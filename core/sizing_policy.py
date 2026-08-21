"""How big a position is, once something else has decided there is one.

The side comes from the ensemble and the timing policy; this only scales it.
Five parameters over features the timing state already computes, so nothing
new is calculated and the rule stays readable: more size when the signal is
further past its own threshold, less when volatility or tail risk is high.

DEFAULT_PARAMS is the identity - every multiplier 1.0 - so the fit starts from
the incumbent the gate measures it against, exactly as the timing fit does.
"""
import numpy as np

from core import timing_fqi as fq

SIZE_LO, SIZE_HI = 0.25, 2.0

PARAM_SPECS = (
    ("base", 0.5, 1.5, False),
    ("k_margin", 0.0, 4.0, False),
    ("k_vol", 0.0, 1.0, False),
    ("k_taleb", 0.0, 1.0, False),
    ("band", 0.02, 0.30, False),
)

DEFAULT_PARAMS = {"base": 1.0, "k_margin": 0.0, "k_vol": 0.0,
                  "k_taleb": 0.0, "band": 0.10}


class SizingPolicy:
    def __init__(self, params=None):
        self.params = dict(DEFAULT_PARAMS)
        self.params.update(params or {})

    def sizes_for(self, series):
        """One multiplier per bar, in [SIZE_LO, SIZE_HI]."""
        p = self.params
        feat = fq.series_features(series)
        # Distance past the asset's own threshold, divided by the band it sits
        # in, so one coefficient means the same thing on every asset.
        margin = np.clip(feat["margin"] / max(1e-6, float(p["band"])), 0.0, 1.0)
        size = (float(p["base"])
                + float(p["k_margin"]) * margin
                - float(p["k_vol"]) * feat["atr_pct"]
                - float(p["k_taleb"]) * feat["taleb_hi"])
        return np.clip(size, SIZE_LO, SIZE_HI)


def match_exposure(sizes, sides):
    """Rescale sizes so the average notional held equals the incumbent's.

    Without this the gate measures LEVERAGE, not sizing. Measured 2026-08-21 on
    ten assets: a CONSTANT size of 1.5, which carries no information at all,
    scored +61.0 against the unit-size arm at p 0.0010, and the fitted rule at
    an average size of 1.63 scored +148.5. score_strategy compounds profit with
    the position while Sharpe is scale-invariant and drawdown is weighted 0.5,
    so any candidate can buy a verdict by simply holding more.

    Matched exposure leaves exactly one thing a candidate can vary: WHERE the
    size goes, not how much of it there is.
    """
    held = np.asarray(sides) != 0
    if not held.any():
        return sizes
    mean_held = float(np.mean(np.abs(np.asarray(sizes, dtype=float)[held])))
    if mean_held <= 0.0:
        return sizes
    return np.asarray(sizes, dtype=float) / mean_held
