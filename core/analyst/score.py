"""Scoring for the analyst log: coverage, error, and a control that can fail.

Nothing here believes a number until it has been shown that the number could
have come out wrong. shuffle_control detaches each forecast from the outcome it
was made for; a real edge collapses and a scorer that was only measuring the
asset's volatility does not.
"""

import random

SHUFFLE_MARGIN = 1.2
# A shuffled forecaster must be at least 20 percent worse for the unshuffled
# advantage to count as real. Below that the two are indistinguishable given
# the sample sizes this log will realistically reach.


def coverage(rows):
    """How often the realized move landed inside the stated interval."""
    marked = [r for r in rows if r.get("inside_interval") is not None]
    inside = sum(1 for r in marked if r["inside_interval"])
    n = len(marked)
    return {"n": n, "inside": inside, "rate": (inside / n) if n else None}


def mae_atr(rows, forecast_key="forecast_atr"):
    """Mean absolute error in ATR units, or None when nothing is scorable."""
    errs = [abs(r[forecast_key] - r["realized_atr_units"]) for r in rows
            if r.get(forecast_key) is not None
            and r.get("realized_atr_units") is not None]
    return (sum(errs) / len(errs)) if errs else None


def shuffle_control(rows, seed=0, forecast_key="forecast_atr"):
    """Rescore with forecasts permuted across rows.

    `survives_shuffle` True is a FAILURE of the measurement, not a success of
    the forecaster: it means the score did not depend on which outcome each
    forecast was paired with.
    """
    base = mae_atr(rows, forecast_key)
    usable = [r for r in rows if r.get(forecast_key) is not None
              and r.get("realized_atr_units") is not None]
    forecasts = [r[forecast_key] for r in usable]
    random.Random(seed).shuffle(forecasts)
    shuffled = [{**r, forecast_key: f} for r, f in zip(usable, forecasts)]
    smae = mae_atr(shuffled, forecast_key)

    if base is None or smae is None:
        survives = True            # no data: never claim the control passed
    elif base > 0:
        survives = smae < base * SHUFFLE_MARGIN
    else:
        # base == 0 is a forecaster that was exactly right every time. Only a
        # shuffle that is ALSO exactly right leaves the score unchanged, so
        # anything above zero is the control doing its job.
        survives = smae <= 0
    return {"mae": base, "mae_shuffled": smae, "n": len(usable),
            "survives_shuffle": survives}


def standings(rows, baselines):
    """The agent against every baseline, plus the disagreement subset.

    `baselines` maps a name to a list of forecasts in ATR units, aligned
    index-wise with `rows`. The alignment is the caller's job because only the
    caller knows how each baseline was constructed.
    """
    agent_mae = mae_atr(rows)
    out_baselines = {}
    for name, values in baselines.items():
        scored = [{**r, "forecast_atr": v} for r, v in zip(rows, values)]
        out_baselines[name] = {"mae": mae_atr(scored)}

    beats = [name for name, b in out_baselines.items()
             if agent_mae is not None and b["mae"] is not None
             and agent_mae < b["mae"]]

    dis_idx = [i for i, r in enumerate(rows)
               if r.get("agent_direction") and r.get("ensemble_direction")
               and r["agent_direction"] != r["ensemble_direction"]]
    dis_rows = [rows[i] for i in dis_idx]
    dis = {"n": len(dis_rows), "agent_mae": mae_atr(dis_rows)}
    for name, values in baselines.items():
        scored = [{**rows[i], "forecast_atr": values[i]} for i in dis_idx]
        dis[f"{name}_mae"] = mae_atr(scored)

    return {"agent": {"mae": agent_mae, "beats": beats},
            "baselines": out_baselines,
            "coverage": coverage(rows),
            "disagreement": dis,
            "control": shuffle_control(rows)}
