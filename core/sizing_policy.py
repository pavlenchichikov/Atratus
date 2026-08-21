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
