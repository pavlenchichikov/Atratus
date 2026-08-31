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


def _directional(rows):
    """(conviction, hit) for every up/down call whose outcome is known.

    `flat` is excluded on purpose: a flat call is a claim about the SIZE of the
    move, and scoring it as a direction would need a band this function has no
    business choosing. coverage() already measures flat calls properly.
    """
    out = []
    for r in rows:
        d, realized = r.get("direction"), r.get("realized_ret")
        if d not in ("up", "down") or realized is None:
            continue
        out.append((r.get("conviction"), int((d == "up") == (realized > 0))))
    return out


def conviction_calibration(rows):
    """Does a 4-of-5 call land more often than a 2-of-5 one?

    Nothing in this scorer ever asked. The agent states a conviction on every
    judgment, the card prints it, and until now no measurement told anyone
    whether the number carried information, was noise, or ran backwards - the
    last of which this project has already found once, in the ensemble's own
    confidence.

    `informative` is deliberately three-valued. On the sample this log will
    reach for months the honest answer is "cannot tell", and reporting a
    monotone-looking table as an effect is how a 26-row rho of -0.125 at
    p=0.543 turns into a belief.
    """
    d = _directional(rows)
    by = {}
    for conv, hit in d:
        cell = by.setdefault(conv, [0, 0])
        cell[0] += hit
        cell[1] += 1
    table = {c: {"hits": h, "n": n, "rate": h / n}
             for c, (h, n) in sorted(by.items()) if n}
    out = {"n": len(d), "by_conviction": table, "rho": None, "p": None,
           "informative": "unknown"}
    if len(d) >= 3 and len({c for c, _h in d}) >= 2:
        from scipy.stats import spearmanr

        rho, p = spearmanr([c for c, _h in d], [h for _c, h in d])
        out["rho"], out["p"] = round(float(rho), 3), round(float(p), 3)
        if p < 0.05:
            out["informative"] = "yes" if rho > 0 else "INVERTED"
    return out


def payoff_agreement(rows):
    """How often the stated direction and its own expected payoff agree.

    They come from two unconnected mechanisms: the direction is the model's,
    the forecast is the empirical payoff of that side out of the shrunk cell
    table. Ten of the first 33 judgments carried a NEGATIVE expected payoff
    behind a directional call, which the card printed verbatim ("up, expected
    payoff -0.32%").

    The split is the point. If the agreeing subset scores better, agreement is
    a free filter that costs no new model call; if it does not, the two
    mechanisms are independent noise and the card should stop implying
    otherwise.
    """
    marked = [r for r in rows if r.get("forecast_atr") is not None]
    agree = [r for r in marked if r["forecast_atr"] > 0]
    disagree = [r for r in marked if r["forecast_atr"] <= 0]
    return {"n": len(marked), "agree": len(agree),
            "agree_rate": (len(agree) / len(marked)) if marked else None,
            "agree_mae": mae_atr(agree), "disagree_mae": mae_atr(disagree)}


def field_usage(rows, min_n=20):
    """Which dossier fields the model cites, and whether citing one pays.

    The dossier now carries 60 fields and the first 33 judgments named 24 of
    them, mostly plain returns. Some of the other 36 are worth their place and
    some are prompt weight; nothing measured which was which.

    Per field: how often it was cited, and the directional hit rate of the
    judgments that cited it against those that did not. `verdict` stays
    "thin" until BOTH sides reach min_n, because a field cited five times has
    a hit rate that means nothing and a table of those reads as knowledge.
    """
    import json as _json

    d = []
    for r in rows:
        if r.get("direction") not in ("up", "down") or r.get("realized_ret") is None:
            continue
        try:
            cited = set(_json.loads(r.get("evidence_json") or "[]"))
        except ValueError:
            cited = set()
        d.append((cited, int((r["direction"] == "up") == (r["realized_ret"] > 0))))

    fields = sorted({f for cited, _h in d for f in cited})
    out = {}
    for f in fields:
        with_f = [h for cited, h in d if f in cited]
        without = [h for cited, h in d if f not in cited]
        entry = {"cited": len(with_f),
                 "hit_with": (sum(with_f) / len(with_f)) if with_f else None,
                 "hit_without": (sum(without) / len(without)) if without else None,
                 "verdict": "thin"}
        if len(with_f) >= min_n and len(without) >= min_n:
            from scipy.stats import fisher_exact

            p = fisher_exact([[sum(with_f), len(with_f) - sum(with_f)],
                              [sum(without), len(without) - sum(without)]])[1]
            if p < 0.05:
                entry["verdict"] = ("helps" if entry["hit_with"] >
                                    entry["hit_without"] else "hurts")
            else:
                entry["verdict"] = "no effect"
            entry["p"] = round(float(p), 3)
        out[f] = entry
    return {"n": len(d), "fields": out,
            "measurable": sum(1 for e in out.values() if e["verdict"] != "thin")}


SHIP_FLOOR = 500
COVERAGE_BAND = (0.75, 0.85)


def verdict(st, n, floor=SHIP_FLOOR):
    """SHIP or HOLD, and every condition that decided it.

    Returns the conditions rather than just the word, because "HOLD" alone says
    nothing about WHICH of the five is missing, and on this log four of them
    pass while the sample sits at 32 of 500.

    Lives here rather than in analyst.py's cmd_score so the page and the command
    line cannot drift into two different definitions of the same verdict.
    """
    cov = (st.get("coverage") or {}).get("rate")
    beats = (st.get("agent") or {}).get("beats") or []
    checks = [
        ("control", st.get("control", {}).get("survives_shuffle") is False,
         "the shuffled forecaster must score WORSE, by a clear margin"),
        ("coverage", cov is not None and COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1],
         "%.0f to %.0f percent of moves inside the stated interval"
         % (COVERAGE_BAND[0] * 100, COVERAGE_BAND[1] * 100)),
        ("beats zero", "zero" in beats, "better than forecasting no move at all"),
        ("beats empirical", "empirical" in beats,
         "better than the cell's own historical payoff"),
        ("sample", n >= floor, "%d scored judgments" % floor),
    ]
    return {"verdict": "SHIP" if all(ok for _n, ok, _w in checks) else "HOLD",
            "checks": [{"name": n, "ok": bool(ok), "want": w}
                       for n, ok, w in checks],
            "n": n, "floor": floor}


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
            "conviction": conviction_calibration(rows),
            "payoff_agreement": payoff_agreement(rows),
            "control": shuffle_control(rows)}
