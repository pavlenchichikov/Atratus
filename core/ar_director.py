"""LLM campaign director: chooses WHAT the next search tries, never how it is judged.

The split this module exists to enforce. "What to try" is an open question with
no right answer, which is what a model is good for: which axis, which label,
how big a budget, whether to spend the LLM proposer arm. "Does this count" is
answered by the Benjamini-Hochberg gate in auto_research and by
ab_build.verdict, and it must be answered by constants chosen before the data
was seen. A director allowed to move the objective, the score basis or alpha
after reading a verdict would be running the search until a measurement passes
and journalling it as a decision.

So the returned settings are checked against a whitelist and can only contain
search levers. The two gate constants are reachable in exactly one way: the
model may return `new_campaign`, which is a request to START A NEW campaign
with a written reason. The caller treats that as a fresh campaign - new freeze,
cleared archive - and never as an edit to the running one. Archive clearing is
not optional there: fitness on the Score scale runs 1.5 to 8.9 and on the AUC
scale about 0.01, so a surviving Score elite would outrank every AUC elite
forever.

Any parse failure, unknown key or out-of-range value falls back to the caller's
campaign. A director that cannot be understood must not be able to widen its own
authority by returning nonsense.
"""

import json
import os

from core import llm_proposer

# Whitelists. A value outside them is a validation failure, not a clamp: a model
# that asked for something impossible has misunderstood the task, and quietly
# rounding its answer into range hides that.
AXES = ("qd", "features", "labeling", "pruning", "hyper", "nets",
        "thresholds", "regime", "weighting")
LABEL_MODES = ("direction", "triple_barrier")
PROPOSERS = ("evolutionary", "llm")
BASES = ("raw", "neural", "net_auc", "net_gain", "ens_auc")
OBJECTIVES = ("mean", "min", "median", "cvar", "sharpe", "trimmed_mean")

BUDGET_MIN, BUDGET_MAX = 5, 60
HORIZON_MIN, HORIZON_MAX = 1, 60

# The only keys a director may set on a running campaign.
SEARCH_KEYS = ("axes", "label_mode", "label_horizon", "budget", "proposer")


def director_on():
    return (os.getenv("GTRADE_AR_DIRECTOR") or "").strip() in ("1", "true", "True")


def compact_findings(findings, n=12):
    """The findings journal reduced to what a decision needs.

    The raw journal is 145 KB and mostly genome bodies. A local model given the
    whole thing spends its context on feature names and answers about those.
    """
    out = []
    for rec in list(findings or [])[-n:]:
        winners = rec.get("winners") or []
        values = [w.get("value") for w in winners if w.get("value") is not None]
        out.append({
            "ts": (rec.get("ts") or "")[:10],
            "mode": rec.get("mode"),
            "basis": rec.get("basis", "raw"),
            "axes": rec.get("axes") or ["qd"],
            "tried": len(winners),
            "adoptable": sum(1 for w in winners if w.get("adoptable")),
            "best": round(max(values), 4) if values else None,
        })
    return out


def _prompt(ctx):
    return (
        "You direct a machine-learning research agent that searches for better "
        "trading-model configurations. Choose only WHAT to try next.\n\n"
        "Current campaign (you may NOT change these; they decide whether a "
        "result counts and were fixed before any data was seen):\n"
        "  score_basis = %(basis)s\n  objective    = %(objective)s\n\n"
        "Recent runs, oldest first:\n%(findings)s\n\n"
        "Archive elites: %(archive_n)d. Cycles run in this campaign: %(cycles)d.\n\n"
        "Reply with ONE JSON object and nothing else:\n"
        '  {"axes": <one of %(axes)s>,\n'
        '   "label_mode": "direction" | "triple_barrier",\n'
        '   "label_horizon": %(hmin)d-%(hmax)d,\n'
        '   "budget": %(bmin)d-%(bmax)d,\n'
        '   "proposer": "evolutionary" | "llm",\n'
        '   "reason": "<one sentence>"}\n\n'
        "If and only if the campaign looks exhausted (several runs, nothing "
        "adoptable, best values flat), you may instead add:\n"
        '  "new_campaign": {"basis": <one of %(bases)s>, '
        '"objective": <one of %(objs)s>, "reason": "<why the current one is done>"}\n'
        "Starting a new campaign discards the search archive, so do not ask for "
        "one to chase a single disappointing run.\n"
        % {"basis": ctx["basis"], "objective": ctx["objective"],
           "findings": json.dumps(ctx["findings"], indent=1),
           "archive_n": ctx["archive_n"], "cycles": ctx["cycles"],
           "axes": list(AXES), "bases": list(BASES), "objs": list(OBJECTIVES),
           "hmin": HORIZON_MIN, "hmax": HORIZON_MAX,
           "bmin": BUDGET_MIN, "bmax": BUDGET_MAX})


def _int_in(value, lo, hi):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi else None


def validate(obj, campaign):
    """(settings, problems) for a director reply. Pure, so the rules are testable.

    settings is None whenever problems is non-empty: a partly-understood reply is
    a misunderstanding, and half-applying it would produce a run nobody chose.
    """
    if not isinstance(obj, dict):
        return None, ["reply was not a JSON object"]
    problems = []
    unknown = sorted(set(obj) - set(SEARCH_KEYS) - {"reason", "new_campaign"})
    if unknown:
        problems.append("keys outside the whitelist: %s" % ", ".join(unknown))

    axes = obj.get("axes", campaign.get("GTRADE_AR_AXES", "qd"))
    if axes not in AXES:
        problems.append("axes %r is not one of %s" % (axes, ", ".join(AXES)))
    label = obj.get("label_mode", campaign.get("GTRADE_LABEL_MODE", "direction"))
    if label not in LABEL_MODES:
        problems.append("label_mode %r is not one of %s"
                        % (label, ", ".join(LABEL_MODES)))
    proposer = obj.get("proposer", campaign.get("GTRADE_AR_PROPOSER", "evolutionary"))
    if proposer not in PROPOSERS:
        problems.append("proposer %r is not one of %s"
                        % (proposer, ", ".join(PROPOSERS)))
    horizon = _int_in(obj.get("label_horizon",
                              campaign.get("GTRADE_LABEL_HORIZON", 1)),
                      HORIZON_MIN, HORIZON_MAX)
    if horizon is None:
        problems.append("label_horizon outside %d-%d" % (HORIZON_MIN, HORIZON_MAX))
    budget = _int_in(obj.get("budget", campaign.get("AR_BUDGET", 15)),
                     BUDGET_MIN, BUDGET_MAX)
    if budget is None:
        problems.append("budget outside %d-%d" % (BUDGET_MIN, BUDGET_MAX))

    # The weighting axis measures nothing under a next-bar label: the label spans
    # one bar, the uniqueness weights come out all-ones, and every candidate
    # equals the base. Catching it here saves a whole run that could only report
    # zero.
    if axes == "weighting" and label == "direction":
        problems.append("the weighting axis is a no-op under a direction label: "
                        "every candidate would equal the base")

    fresh = obj.get("new_campaign")
    new_campaign = None
    if fresh is not None:
        if not isinstance(fresh, dict):
            problems.append("new_campaign was not an object")
        else:
            basis = fresh.get("basis", campaign.get("GTRADE_AR_SCORE_BASIS"))
            objective = fresh.get("objective", campaign.get("GTRADE_AR_OBJECTIVE"))
            reason = str(fresh.get("reason") or "").strip()
            if basis not in BASES:
                problems.append("new_campaign basis %r is not one of %s"
                                % (basis, ", ".join(BASES)))
            if objective not in OBJECTIVES:
                problems.append("new_campaign objective %r is not one of %s"
                                % (objective, ", ".join(OBJECTIVES)))
            # An unexplained campaign switch is the move this whole module is
            # built to make visible, so it is refused rather than logged.
            if len(reason) < 10:
                problems.append("new_campaign needs a written reason")
            if not problems:
                new_campaign = {"GTRADE_AR_SCORE_BASIS": basis,
                                "GTRADE_AR_OBJECTIVE": objective,
                                "reason": reason}
    if problems:
        return None, problems
    settings = {
        "GTRADE_AR_AXES": axes,
        "GTRADE_LABEL_MODE": label,
        "GTRADE_LABEL_HORIZON": str(horizon),
        "AR_BUDGET": str(budget),
        "GTRADE_AR_PROPOSER": proposer,
        "reason": str(obj.get("reason") or "").strip(),
    }
    if new_campaign:
        settings["new_campaign"] = new_campaign
    return settings, []


def propose(findings, campaign, archive_n=0, cycles=0):
    """Ask the director for the next search settings. None means keep the campaign.

    Never raises: an unavailable or unparseable model must degrade to the
    settings already in force, not stop an unattended run.
    """
    ctx = {"basis": campaign.get("GTRADE_AR_SCORE_BASIS", "raw"),
           "objective": campaign.get("GTRADE_AR_OBJECTIVE", "mean"),
           "findings": compact_findings(findings),
           "archive_n": archive_n, "cycles": cycles}
    try:
        reply = llm_proposer._backend("director")(_prompt(ctx))
    except Exception as exc:
        print("[director] unavailable (%s); keeping the current campaign." % exc)
        return None
    settings, problems = validate(llm_proposer._parse_obj(reply), campaign)
    if problems:
        print("[director] reply rejected, keeping the current campaign:")
        for p in problems:
            print("  - %s" % p)
        return None
    return settings
