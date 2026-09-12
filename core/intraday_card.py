"""Today's intraday levels and the odds of touching them, for the asset card.

What this shows is the only part of the intraday programme that survived its
gate. Measured 2026-09-12 over a held-out year, 492 assets, 177765 sessions:

    Brier   this rule 0.2286   20-session frequency 0.2576   flat rate 0.2491
    better than the frequency baseline on 486 of 492 assets
    reliability 0.127/0.119  0.324/0.303  0.508/0.513  0.638/0.688

The odds come from ONE number - how far the level is in units of the current
hourly volatility - through the calibration in intraday_reach.json. No model:
a pooled CatBoost over 21 columns beat that rule by +0.022 of AUC by median,
which is inside the band a shuffled-label fit reaches by pooling assets with
different base rates, so the rule is what ships.

Two limits the card has to state rather than imply:
  - the calibration was fitted and validated AT THE FIRST BAR of a session, so
    that is when these odds are true. Quoting them mid-session is a
    generalisation nobody has measured.
  - k is fitted so the level is reached about half the time, so most sessions
    sit near a coin flip: 116307 of the 177765 held-out rows landed in the
    0.4-0.6 band. The odds separate at the edges, not every day.

Direction inside the session is NOT offered. It measured 0.5201 against a
0.5154 baseline, and the hourly timing policy did not clear its floor.
"""
import json
import os

import numpy as np

from core.intraday import session_zone, sessionize
from core.intraday_features import build_frame
from core.intraday_labels import MIN_K_SESSIONS, fit_k
from core.intraday_reach import reach_probability

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIT_PATH = os.path.join(BASE, "intraday_reach.json")
MIN_SESSIONS = 40          # below this there is nothing to fit k on


def load_fit(path=None):
    """The calibration, or None when it has never been fitted here."""
    try:
        with open(path or FIT_PATH, encoding="utf-8") as fh:
            blob = json.load(fh)
        return {"w": float(blob["w"]), "b": float(blob["b"]),
                "offsets": {k: float(v) for k, v in (blob.get("offsets") or {}).items()},
                "cutoff": blob.get("cutoff")}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _levels_for(frame, asset, side):
    """(level, k, distance in sigma_h) for one side of today's session, or None.

    k is fitted on the asset's OWN history, every session before today, exactly
    as core.intraday_labels does it for the measurement. Using a global k would
    quote a level this asset does not reach half the time.
    """
    sessions = sorted(frame["session"].unique())
    today = sessions[-1]
    past = frame[frame["session"] < today]
    if len(set(past["session"])) < MIN_SESSIONS:
        return None
    first = past[past["bar_idx"] == 0].set_index("session")
    rest = (past[past["bar_idx"] > 0].groupby("session")
            .agg(rest_hi=("high", "max"), rest_lo=("low", "min")))
    ses = first.join(rest, how="inner")
    ses = ses[ses["sigma_d"].notna() & ses["sigma_h"].notna()]
    if len(ses) < MIN_K_SESSIONS:
        return None
    o = ses["sess_open"].to_numpy(dtype=float)
    sd = ses["sigma_d"].to_numpy(dtype=float)
    first_ext = (ses["high"] if side == 1 else ses["low"]).to_numpy(dtype=float)
    rest_ext = (ses["rest_hi"] if side == 1 else ses["rest_lo"]).to_numpy(dtype=float)
    k = fit_k(o, sd, first_ext, rest_ext, side, np.ones(len(ses), dtype=bool))
    if k is None:
        return None

    bar = frame[(frame["session"] == today) & (frame["bar_idx"] == 0)]
    if bar.empty:
        return None
    row = bar.iloc[0]
    if not np.isfinite(row["sigma_d"]) or not np.isfinite(row["sigma_h"]):
        return None
    level = float(row["sess_open"]) * (1.0 + side * k * float(row["sigma_d"]))
    # Signed toward the level, the way f_lvl_dist is built for the fit.
    dist = side * float(np.log(level / float(row["close"]))) / float(row["sigma_h"])
    return {"level": level, "k": float(k), "dist": dist,
            "reached": bool(row["high"] >= level) if side == 1
            else bool(row["low"] <= level)}


def intraday_for_asset(asset, bars, tz_name, fit=None):
    """{status, session, upper, lower} for the card, never an exception.

    Every refusal is a status, like core.levels: a card with a silent gap is
    worse than one that says why the number is missing.
    """
    out = {"status": "ok", "session": None, "upper": None, "lower": None,
           "cutoff": None}
    fit = load_fit() if fit is None else fit
    if fit is None:
        out["status"] = "no_calibration"
        return out
    out["cutoff"] = fit.get("cutoff")
    if not bars or not tz_name:
        out["status"] = "no_hourly_bars"
        return out
    df = sessionize(bars, session_zone(asset, tz_name))
    if not len(df):
        out["status"] = "no_sessions"
        return out
    frame = build_frame(df)
    out["session"] = max(frame["session"].unique())
    sides = {}
    for side, name in ((1, "upper"), (-1, "lower")):
        got = _levels_for(frame, asset, side)
        if got is None:
            continue
        got["probability"] = float(reach_probability(fit, asset, got["dist"]))
        sides[name] = got
    if not sides:
        out["status"] = "short_history"
        return out
    out.update(sides)
    return out
