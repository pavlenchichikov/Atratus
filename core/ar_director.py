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
MODES = ("search", "regate")
ILLUM = ("cb", "full")
SELECTIONS = ("full", "fast")

BUDGET_MIN, BUDGET_MAX = 5, 60
HORIZON_MIN, HORIZON_MAX = 1, 60

# A cycle may run several axes. auto_research already accepts a comma list and
# runs each axis independently against the SAME shared base, then corrects all
# their p-values together (Benjamini-Hochberg over the winners), so a mixed
# cycle is a wider net, not a compound genome: two axes in a list never combine
# their changes. The `qd` genome is what composes genes, which is why it cannot
# share a cycle - auto_research short-circuits to run_qd() and silently ignores
# every other name in the list.
MAX_AXES = 3
# The budget is spent PER AXIS, so a three-axis cycle at budget 40 is 120
# genomes and about eighteen hours. BUDGET_MAX is therefore the cap on the
# cycle's total, and a director that wants breadth pays for it in depth.

# Every remaining lever, as (director key, env var, low, high). All are numbers
# with a range; the enumerated ones are handled separately above. A range is a
# refusal boundary, not a clamp, for the reason at the top of the whitelists.
NUMERIC_LEVERS = (
    # QD shape. init seeds the archive, final decides how many elites reach the
    # expensive held-out gate, max_misses is how long the search keeps drawing
    # children the tried-registry has already seen before it gives up.
    ("qd_init", "GTRADE_AR_QD_INIT", 2, 30),
    ("qd_final", "GTRADE_AR_QD_FINAL", 1, 10),
    ("max_misses", "GTRADE_AR_QD_MAX_MISSES", 1, 50),
    # The triple-barrier shape. barrier_k is in units of the local volatility,
    # so 0.5 is a barrier half a sigma away (easy to touch, dense labels) and 3
    # is a rare one. vol_window is the EWM span that sigma is measured over.
    ("barrier_k", "GTRADE_LABEL_BARRIER_K", 0.25, 5.0),
    ("vol_window", "GTRADE_LABEL_VOL_WINDOW", 5, 100),
    # How many stored candidates a regate cycle re-tests.
    ("regate_k", "GTRADE_AR_REGATE_K", 3, 20),
    # Wall-clock stop for the cycle, 0 = none. Cuts the search between
    # candidates and still gates everything already found.
    ("hours", "GTRADE_AR_TIME_BUDGET_H", 0, 24),
    # Probability the QD loop asks the LLM for a child instead of mutating.
    ("qd_llm_p", "GTRADE_AR_QD_LLM_P", 0.0, 1.0),
)
_FLOAT_LEVERS = {"barrier_k", "hours", "qd_llm_p"}

# The two cheap gates in front of the expensive one. Measured 2026-08-13:
# screen 43s, tier 545s, held-out 1997s per candidate.
BOOL_LEVERS = (("screen", "GTRADE_AR_SCREEN"), ("tier", "GTRADE_AR_TIER"))

# The only keys a director may set on a running campaign.
SEARCH_KEYS = (("axes", "label_mode", "label_horizon", "budget", "proposer",
                "mode", "illum", "search_assets")
               + tuple(k for k, _e, _lo, _hi in NUMERIC_LEVERS)
               + tuple(k for k, _e in BOOL_LEVERS))


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
            # Named for what it IS. This counts SEARCH-GATE flags: a winner
            # measured on the search basis against a bare base, which says
            # "worth an A/B", not "better than production". Called "adoptable",
            # the model read it as a result and chased the one axis that kept
            # producing it - the labeling winner was flagged eight times, and
            # its A/B then lost on 10 of 14 assets.
            "gate_flagged": sum(1 for w in winners if w.get("adoptable")),
            "best": round(max(values), 4) if values else None,
        })
    return out


def axis_yield(findings, n=30):
    """Per-axis totals over the recent journal: cycles spent, winners, gate flags.

    compact_findings puts one record per cycle in front of the model and leaves
    the adding up to it, which a local model does not do: on 2026-08-20 the
    director spent five consecutive cycles on `hyper`, every one of them
    returning zero winners, because no line said "this axis has produced
    nothing five times running". Winners are attributed by their own `axis`
    field, so a mixed cycle credits the axis that actually produced them.
    """
    tot = {}

    def slot(ax):
        return tot.setdefault(ax, {"axis": ax, "cycles": 0, "winners": 0,
                                   "gate_flagged": 0})

    for rec in list(findings or [])[-n:]:
        axes = rec.get("axes") or ["qd"]
        for ax in axes:
            slot(ax)["cycles"] += 1
        for w in rec.get("winners") or []:
            ax = w.get("axis") or (axes[0] if len(axes) == 1 else None)
            if ax is None:
                continue
            slot(ax)["winners"] += 1
            if w.get("adoptable"):
                slot(ax)["gate_flagged"] += 1
    return sorted(tot.values(), key=lambda c: (-c["cycles"], c["axis"]))


def _prompt(ctx):
    return (
        "You direct a machine-learning research agent that searches for better "
        "trading-model configurations. Choose only WHAT to try next.\n\n"
        "Current campaign (you may NOT change these; they decide whether a "
        "result counts and were fixed before any data was seen):\n"
        "  search_basis   = %(basis)s   (what the SEARCH optimises)\n"
        "  decision_basis = %(decision)s   (what an ADOPTION is judged on)\n"
        "  objective      = %(objective)s\n\n"
        "There are two levels of evidence and they are not the same thing.\n\n"
        "1. Search runs. gate_flagged counts candidates the search gate thought "
        "were WORTH TESTING, measured on the search basis against a bare base. "
        "It is not a result, and the same axis being flagged again and again is "
        "one finding re-found, not accumulating proof.\n%(findings)s\n\n"
        "2. A/B outcomes. These were trained on a fresh held-out set and measured "
        "against what is ACTUALLY RUNNING, on the decision basis. would_promote "
        "and would_demote count the assets whose champion the candidate would "
        "replace or lose, which is the decision production makes. Only a PASSED "
        "verdict here means something improved.\n%(adoptions)s\n\n"
        "3. Per-axis totals over the same journal: how many cycles each axis "
        "was searched, how many winners it produced and how many of those the "
        "gate flagged. An axis with several cycles and no winners has already "
        "answered: under this basis the search finds nothing there, and "
        "spending another cycle on it buys another empty run.\n%(axis_yield)s\n\n"
        "If an axis keeps being gate_flagged while its A/B outcomes FAIL, that "
        "axis is exhausted whatever its gate values look like. Prefer an axis or "
        "a search mode that has not been ruled out that way.\n\n"
        "You may name up to %(maxax)d axes, comma separated, and a mixed cycle is "
        "the right move when you do not know which of them is alive. They are "
        "searched INDEPENDENTLY against the same base and their p-values are "
        "corrected together, so a mix is a wider net, never a combined change: "
        "nothing composes two axes except the qd genome, and qd must be alone. "
        "The budget is spent per axis, so budget x number of axes may not exceed "
        "%(bmax)d; breadth is paid for in depth.\n\n"
        "A cycle does not have to search. mode=regate spends it re-testing the "
        "genomes the search ALREADY flagged, under the current gate. That is how "
        "a finding earns its second independent clear on purpose instead of by "
        "chance, and a flagged finding that never replicates is worth nothing. "
        "Prefer it when gate_flagged has been running well ahead of replications.\n\n"
        "Archive elites: %(archive_n)d. Cycles run in this campaign: %(cycles)d.\n\n"
        "Reply with ONE JSON object and nothing else. Every key except reason is "
        "optional; omit what you are not deliberately choosing.\n"
        '  {"axes": <one of %(axes)s, or up to %(maxax)d of them comma separated>,\n'
        '   "label_mode": "direction" | "triple_barrier",\n'
        '   "label_horizon": %(hmin)d-%(hmax)d,\n'
        '   "budget": %(bmin)d-%(bmax)d,\n'
        '   "proposer": "evolutionary" | "llm",\n'
        '   "mode": "search" | "regate",       // regate: re-test stored winners\n'
        '   "regate_k": 3-20,                  // mode=regate only\n'
        '   "hours": 0-24,                     // wall-clock stop, 0 = none\n'
        '   "search_assets": "full" | "fast",  // fast = 5 assets, ~2x quicker, noisier\n'
        '   "screen": true|false,              // 43s CatBoost prefilter\n'
        '   "tier": true|false,                // 545s 4-asset prefilter\n'
        '   "illum": "cb" | "full",            // qd only: cb stubs the nets to a\n'
        "                                      //   constant, so the archive is a pure\n"
        "                                      //   CatBoost selection and a net basis\n"
        "                                      //   scores every genome the same. full\n"
        "                                      //   trains real nets, costs about 12x,\n"
        "                                      //   and is the ONLY way a search can\n"
        "                                      //   hunt neural levers.\n"
        '   "qd_init": 2-30, "qd_final": 1-10, "max_misses": 1-50, "qd_llm_p": 0.0-1.0,\n'
        "                                      // qd only: archive seed, how many elites\n"
        "                                      //   reach the expensive gate, how long to\n"
        "                                      //   keep drawing already-tried children,\n"
        "                                      //   how often to ask an LLM for one\n"
        '   "barrier_k": 0.25-5.0, "vol_window": 5-100,\n'
        "                                      // triple_barrier only: barrier width in\n"
        "                                      //   sigmas and the span sigma is measured\n"
        "                                      //   over. Narrow = denser, easier labels.\n"
        '   "reason": "<one sentence>"}\n\n'
        "A lever that lands on nothing is REFUSED, not ignored, so do not send qd "
        "levers for a non-qd cycle or barrier settings under a direction label.\n\n"
        "If and only if the campaign looks exhausted (several runs, nothing "
        "adoptable, best values flat), you may instead add:\n"
        '  "new_campaign": {"basis": <one of %(bases)s>, '
        '"objective": <one of %(objs)s>, "reason": "<why the current one is done>"}\n'
        "Starting a new campaign discards the search archive, so do not ask for "
        "one to chase a single disappointing run.\n"
        % {"basis": ctx["basis"], "objective": ctx["objective"],
           "decision": ctx["decision"],
           "findings": json.dumps(ctx["findings"], indent=1),
           "axis_yield": json.dumps(ctx["axis_yield"], indent=1),
           "adoptions": (json.dumps(ctx["adoptions"], indent=1)
                         if ctx["adoptions"] else
                         "  (no A/B has finished in this campaign yet)"),
           "archive_n": ctx["archive_n"], "cycles": ctx["cycles"],
           "axes": list(AXES), "bases": list(BASES), "objs": list(OBJECTIVES),
           "maxax": MAX_AXES,
           "hmin": HORIZON_MIN, "hmax": HORIZON_MAX,
           "bmin": BUDGET_MIN, "bmax": BUDGET_MAX})


def _axis_list(value):
    """(names, problems) for the `axes` field, which may be one name, a comma string
    or a list. Order is kept and duplicates collapse, so "hyper,hyper" is one axis
    and not a doubled bill."""
    if isinstance(value, (list, tuple)):
        raw = [str(v).strip() for v in value]
    else:
        raw = [n.strip() for n in str(value).split(",")]
    names, problems = [], []
    for n in raw:
        if not n:
            continue
        if n not in AXES:
            problems.append("axes %r is not one of %s" % (n, ", ".join(AXES)))
        elif n not in names:
            names.append(n)
    if not names and not problems:
        problems.append("axes was empty")
    if len(names) > MAX_AXES:
        problems.append("at most %d axes per cycle, got %d" % (MAX_AXES, len(names)))
    if "qd" in names and len(names) > 1:
        problems.append("qd cannot share a cycle with another axis: the runner "
                        "takes the qd path and ignores the rest")
    return names, problems


def _int_in(value, lo, hi):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi else None


def _num_in(value, lo, hi, as_float):
    try:
        n = float(value) if as_float else int(value)
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi else None


def _bool_env(value):
    """A director's true/false as the "1"/"0" the runner reads. None if unreadable."""
    if isinstance(value, bool):
        return "1" if value else "0"
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return "1"
    if v in ("0", "false", "no", "off"):
        return "0"
    return None


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

    names, axis_problems = _axis_list(obj.get("axes",
                                              campaign.get("GTRADE_AR_AXES", "qd")))
    problems.extend(axis_problems)
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
    elif budget * len(names) > BUDGET_MAX:
        problems.append("budget %d across %d axes is %d genomes, over the %d a cycle "
                        "may cost (the budget is spent per axis)"
                        % (budget, len(names), budget * len(names), BUDGET_MAX))

    mode = obj.get("mode", "search")
    if mode not in MODES:
        problems.append("mode %r is not one of %s" % (mode, ", ".join(MODES)))
    illum = obj.get("illum", campaign.get("GTRADE_AR_ILLUM", "cb"))
    if illum not in ILLUM:
        problems.append("illum %r is not one of %s" % (illum, ", ".join(ILLUM)))
    selection = obj.get("search_assets", "full")
    if selection not in SELECTIONS:
        problems.append("search_assets %r is not one of %s"
                        % (selection, ", ".join(SELECTIONS)))

    numbers = {}
    for key, env_key, lo, hi in NUMERIC_LEVERS:
        if key not in obj:
            continue
        n = _num_in(obj[key], lo, hi, key in _FLOAT_LEVERS)
        if n is None:
            problems.append("%s outside %s-%s" % (key, lo, hi))
        else:
            numbers[env_key] = str(n)
    bools = {}
    for key, env_key in BOOL_LEVERS:
        if key not in obj:
            continue
        b = _bool_env(obj[key])
        if b is None:
            problems.append("%s is not a true/false value" % key)
        else:
            bools[env_key] = b

    # The weighting axis measures nothing under a next-bar label: the label spans
    # one bar, the uniqueness weights come out all-ones, and every candidate
    # equals the base. Catching it here saves a whole run that could only report
    # zero. Refused rather than dropped from the list: a director that asked for
    # a no-op has misread the campaign, and quietly running the other two axes
    # would hide that behind a result.
    if "weighting" in names and label == "direction":
        problems.append("the weighting axis is a no-op under a direction label: "
                        "every candidate would equal the base")
    # Same rule for every other lever that lands on nothing. A setting nobody
    # reads is not harmless: it is journalled as part of the run, so a later
    # reader would credit a result to a barrier width that was never applied.
    if label != "triple_barrier" and ({"barrier_k", "vol_window"} & set(obj)):
        problems.append("barrier_k and vol_window only shape a triple_barrier "
                        "label; under %s nothing reads them" % label)
    qd_keys = {"qd_init", "qd_final", "max_misses", "qd_llm_p"} & set(obj)
    if qd_keys and names != ["qd"]:
        problems.append("%s %s only to the qd search; this cycle runs %s"
                        % (", ".join(sorted(qd_keys)),
                           "apply" if len(qd_keys) > 1 else "applies",
                           ",".join(names) or "nothing"))
    if mode == "regate" and (qd_keys or "axes" in obj or "proposer" in obj):
        problems.append("a regate cycle re-tests stored genomes and runs no "
                        "search, so axes, proposer and the qd levers do nothing")
    if mode != "regate" and "regate_k" in obj:
        problems.append("regate_k only applies to mode=regate")
    if illum == "full" and "qd" not in names:
        problems.append("illum only shapes the qd archive; this cycle runs %s"
                        % (",".join(names) or "nothing"))

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
        "GTRADE_AR_AXES": ",".join(names),
        "GTRADE_LABEL_MODE": label,
        "GTRADE_LABEL_HORIZON": str(horizon),
        "AR_BUDGET": str(budget),
        "GTRADE_AR_PROPOSER": proposer,
        # Written on every reply, not only when asked for: these are sticky env
        # vars on a long-lived loop, so leaving one out would silently carry the
        # previous cycle's choice into a cycle whose reason never mentions it.
        "GTRADE_AR_MODE": mode,
        "GTRADE_AR_ILLUM": illum,
        "GTRADE_AR_SELECTION": selection,
        "reason": str(obj.get("reason") or "").strip(),
    }
    settings.update(numbers)
    settings.update(bools)
    if new_campaign:
        settings["new_campaign"] = new_campaign
    return settings, []


def propose(findings, campaign, archive_n=0, cycles=0, adoptions=None):
    """Ask the director for the next search settings. None means keep the campaign.

    Never raises: an unavailable or unparseable model must degrade to the
    settings already in force, not stop an unattended run.
    """
    ctx = {"basis": campaign.get("GTRADE_AR_SCORE_BASIS", "raw"),
           "decision": (campaign.get("GTRADE_AR_DECISION_BASIS")
                        or campaign.get("GTRADE_AR_SCORE_BASIS", "raw")),
           "objective": campaign.get("GTRADE_AR_OBJECTIVE", "mean"),
           "findings": compact_findings(findings),
           "axis_yield": axis_yield(findings),
           "adoptions": adoptions or [],
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
