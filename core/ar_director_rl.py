"""RL campaign director: a bandit over named cycle recipes.

The LLM director in core/ar_director.py chooses the next cycle's settings from
a prompt. This one chooses them from outcomes. Both leave through the same
validate(), so neither has authority the other lacks, and neither can move the
score basis, the objective or the adoption rule.

An arm is a WHOLE RECIPE, not a lever. Levers interact: illum=full is
meaningless without qd, the budget is spent per axis, a barrier width is
meaningless under a direction label. A bandit over individual levers would
ignore all of that, and on the 63 findings records that exist it would act at
random for months.

stdlib + core.ar_rl + core.ar_memory only. Genome signatures need
auto_research, which imports THIS layer, so the caller injects them as a
callable - the rule core/ar_rl.py already follows.
"""
import os
import random

from core import ar_director, ar_memory, ar_rl

# name -> {"reply": what a director returns, "hours": prior cost}
# The hours are priors, replaced by the measured median once the history holds
# three finished cycles for that arm (see measured_hours).
RECIPES = {
    # Neither names illum: they FOLLOW the campaign's illumination, so a
    # campaign that illuminates on real nets is not quietly downgraded to the
    # CatBoost-only screen by the arm that happened to be drawn. What separates
    # them is the proposer and the budget, which is what their names say.
    "qd_cheap": {
        "hours": 4.0,
        "reply": {"axes": "qd", "proposer": "evolutionary", "budget": 20}},
    "qd_llm": {
        "hours": 5.0,
        "reply": {"axes": "qd", "proposer": "llm", "budget": 20,
                  "qd_llm_p": 0.5}},
    "qd_neural": {
        # illum=full trains real nets during illumination instead of stubbing
        # them to a constant. About 12x the cost per genome, and the only arm
        # that can hunt a neural lever at all, so the budget is cut to match.
        "hours": 12.0,
        "reply": {"axes": "qd", "proposer": "evolutionary", "budget": 8,
                  "illum": "full"}},
    "hyper_nets": {
        "hours": 5.0,
        "reply": {"axes": "hyper,nets", "proposer": "evolutionary",
                  "budget": 20}},
    "features_deep": {
        "hours": 7.0,
        "reply": {"axes": "features", "proposer": "llm", "budget": 40}},
    "labeling_tight": {
        "hours": 5.0,
        "reply": {"axes": "labeling", "proposer": "evolutionary", "budget": 20,
                  "label_mode": "triple_barrier", "label_horizon": 10,
                  "barrier_k": 0.75, "vol_window": 20}},
    "labeling_wide": {
        "hours": 5.0,
        "reply": {"axes": "labeling", "proposer": "evolutionary", "budget": 20,
                  "label_mode": "triple_barrier", "label_horizon": 30,
                  "barrier_k": 2.0, "vol_window": 40}},
    "thresholds_regime": {
        "hours": 4.0,
        "reply": {"axes": "thresholds,regime", "proposer": "evolutionary",
                  "budget": 20}},
    "weighting_tb": {
        # weighting is a no-op under a next-bar label, so the arm carries the
        # multi-bar label that makes it mean anything.
        "hours": 4.0,
        "reply": {"axes": "weighting", "proposer": "evolutionary", "budget": 20,
                  "label_mode": "triple_barrier", "label_horizon": 20}},
    "pruning": {
        "hours": 4.0,
        "reply": {"axes": "pruning", "proposer": "evolutionary", "budget": 20}},
    "regate": {
        # Re-tests stored winners instead of searching. The only arm that earns
        # a finding its SECOND independent clear on purpose.
        "hours": 2.0,
        "reply": {"mode": "regate", "regate_k": 10}},
    "fast_scout": {
        # Cheap information. Without an arm like this every exploratory draw
        # costs a full cycle.
        "hours": 1.5,
        "reply": {"axes": "hyper", "proposer": "evolutionary", "budget": 30,
                  "search_assets": "fast", "hours": 3}},
}

ARMS = tuple(RECIPES)


def recipe_reply(name):
    """The director reply for one arm, as a fresh dict (callers mutate it)."""
    return dict(RECIPES[name]["reply"])


# What validate() fills in for anything a recipe does not name. Pinned on
# purpose: these five are what makes settings_of stable, so arm_of can still
# recognise a cycle recorded weeks ago. The BASIS is deliberately absent - see
# _campaign().
FIXTURE = {"GTRADE_AR_AXES": "qd", "GTRADE_LABEL_MODE": "direction",
           "GTRADE_LABEL_HORIZON": "1", "AR_BUDGET": "15",
           "GTRADE_AR_PROPOSER": "evolutionary"}


def _campaign():
    """The campaign a recipe is validated against.

    FIXTURE plus the two levers the CAMPAIGN owns rather than the cycle.

    The basis is frozen for the whole campaign and decides whether a recipe is
    legal at all: illum=full pays about 12x per genome to train real nets during
    illumination, and on the raw Score basis nothing can read them.

    The illumination is derived from that basis when the campaign starts
    (run_gtrade.bat [0a]), and it has to be read live for the same reason.
    validate() writes GTRADE_AR_ILLUM on every reply so a stale value cannot
    leak, so a fixture that does not name it silently reset a campaign
    illuminating on real nets back to the CatBoost-only screen on every arm but
    one - which is what happened for the whole 2026-08-22 campaign.
    """
    camp = dict(FIXTURE)
    for env_key in ("GTRADE_AR_SCORE_BASIS", "GTRADE_AR_ILLUM"):
        live = (os.getenv(env_key) or "").strip().lower()
        if live:
            camp[env_key] = live
    return camp


def _check_recipes():
    """Every recipe must be a cycle the runner can execute.

    Checked at import so an illegal recipe fails the module load rather than
    the fourth hour of a search. validate() already refuses levers that land on
    nothing, so this is a free control over the whole set.

    Against FIXTURE, not _campaign(): a recipe that only some campaigns can run
    is legal, and failing the import on a raw campaign would take the whole
    loop down over one arm. legal_arms() is where a campaign narrows the set.
    """
    for name in ARMS:
        _settings, problems = ar_director.validate(recipe_reply(name), FIXTURE)
        if problems:
            raise ValueError("recipe %r is not a legal cycle: %s"
                             % (name, "; ".join(problems)))


_check_recipes()


# Reward components. The cap on R_FLAG is the load-bearing one: a search-gate
# flag says "worth testing", not "better than production", and an arm that
# only ever produces flags must not be able to outrank one that produces a
# passing A/B.
R_FLAG, R_PASS, R_REPLICATED = 0.2, 1.0, 2.0
MIN_HOURS = 0.25   # a cycle shorter than this is a crash or a cache hit
REWARD_CAP = 2.0       # a pass on a 1.6h cycle; above this an arm saturates
_RNG = random.Random()


def cycle_reward(found=False, flagged=False, passed=False, replicated=False,
                 hours=1.0):
    """Reward per hour for one finished cycle.

    Per hour, not per cycle: regate and fast_scout are cheaper than qd_neural
    by an order of magnitude, and without the division the bandit buys the
    expensive arm every time it wins at all.
    """
    if not found:
        return 0.0
    total = 0.0
    if flagged:
        total += R_FLAG          # once per cycle however many winners flagged
    if passed:
        total += R_PASS
    if replicated:
        total += R_REPLICATED
    return total / max(MIN_HOURS, float(hours))


def settings_of(name):
    """The validated env-key dict a recipe produces, or None if it is illegal.

    Goes through the same validator the loop uses, so what is stored in the
    history for an RL cycle is the same shape as for an LLM one.
    """
    settings, problems = ar_director.validate(recipe_reply(name), _campaign())
    if problems:
        return None
    settings.pop("reason", None)
    return settings


def legal_arms():
    """The arms this campaign can actually run, never empty.

    Choosing an arm the campaign refuses would run the cycle with no settings
    applied at all, and the bandit would then credit the result to a recipe
    that was never used. The one arm this narrows today is qd_neural under the
    raw basis; if a campaign somehow refused everything, the full set is
    returned rather than nothing, because a loop that cannot choose is worse
    than a loop that chooses badly.
    """
    return [a for a in ARMS if settings_of(a) is not None] or list(ARMS)


def arm_of(settings):
    """Which recipe a recorded settings dict came from, or None.

    Exact match on the env keys the recipe sets, so an LLM cycle that happens
    to coincide with a recipe is credited to that arm, which is correct: the
    reward belongs to the move, not to who thought of it.
    """
    if not isinstance(settings, dict):
        return None
    for name in ARMS:
        want = settings_of(name)
        if want and want == {k: settings.get(k) for k in want}:
            return name
    return None


def _cycle_findings(entry, findings, next_ts):
    """The findings records this cycle produced: written after it started and
    before the next recorded cycle did."""
    return [f for f in findings
            if entry["ts"] <= (f.get("ts") or "") < (next_ts or "9999")]


def settle(history, findings, outcomes, replicated_sigs, credited, sig_of):
    """Newly creditable {arm, reward, key} records. Pure.

    history is newest-first as auto_loop stores it. credited is the set of keys
    already paid, so calling this every tick is idempotent.
    """
    by_sig = {}          # genome signature -> the cycle entry that produced it
    empty = []           # cycles that produced no winner at all
    flagged_ts = set()   # cycles that produced at least one gate flag
    ordered = list(reversed(history))
    for i, entry in enumerate(ordered):
        if entry.get("action") != "search" or entry.get("rc") != 0:
            continue
        arm = arm_of(entry.get("settings"))
        if arm is None:
            continue                      # predates the settings field
        nxt = ordered[i + 1]["ts"] if i + 1 < len(ordered) else None
        winners = [w for f in _cycle_findings(entry, findings, nxt)
                   for w in (f.get("winners") or [])]
        if not winners:
            empty.append((entry, arm))
            continue
        for w in winners:
            if w.get("adoptable"):
                flagged_ts.add(entry["ts"])
            sig = sig_of(w.get("genome") or {})
            if sig:
                by_sig.setdefault(sig, (entry, arm))

    out = []
    for entry, arm in empty:
        key = "empty:%s" % entry["ts"]
        if key not in credited:
            out.append({"arm": arm, "reward": 0.0, "key": key})
    for oc in outcomes or []:
        sig = oc.get("sig")
        if not sig or sig not in by_sig:
            continue                      # untraceable: dropped, never guessed
        entry, arm = by_sig[sig]
        key = "ab:%s:%s" % (sig, oc.get("verdict"))
        if key in credited:
            continue
        out.append({"arm": arm, "key": key, "reward": cycle_reward(
            found=True, flagged=entry["ts"] in flagged_ts,
            passed=oc.get("verdict") == "PASSED",
            replicated=sig in (replicated_sigs or set()),
            hours=float(entry.get("seconds") or 3600.0) / 3600.0)})
    return out


STATE_KEY = "rl_director_v1"
PHASE = "all"          # contextless: 63 records will not feed a context vector
REGATE_GAP = 8         # flagged findings owed a second clear before it is forced


def mode():
    """llm (default) | rl | alternate. Anything unreadable is llm, so an
    unset or fat-fingered variable is today's behaviour."""
    v = (os.getenv("GTRADE_AR_DIRECTOR_MODE") or "llm").strip().lower()
    return v if v in ("llm", "rl", "alternate") else "llm"


def chooser_for(cycle):
    """Whose turn it is under `alternate`: odd cycles are the RL director's."""
    return "rl" if int(cycle) % 2 else "llm"


def replication_debt(findings, replicated_sigs, sig_of):
    """Distinct flagged genomes still owed a second independent clear.

    Counted per GENOME. findings_summary's counters, which this used to read,
    are cumulative EVENT totals: a regate cycle appends one adoptable event and
    one replicated event for the same already-replicated genome, so both
    counters rise together and their difference is a constant no regate cycle
    can pay down.
    """
    owed = set()
    seen = replicated_sigs or set()
    for rec in findings or []:
        for w in rec.get("winners") or []:
            if not w.get("adoptable"):
                continue
            sig = sig_of(w.get("genome") or {})
            if sig and sig not in seen:
                owed.add(sig)
    return len(owed)


def forced_arm(debt, history=None):
    """regate when the replication debt is large enough, else None.

    A flagged finding that never replicates is worth nothing, and the gap is
    arithmetic. Teaching a bandit to rediscover arithmetic costs weeks of GPU
    time and buys nothing.

    Never twice in a row. regate re-gates the top-k stored candidates, so a
    genome owed a clear that never ranks in that k is unreachable and the debt
    stands however often the arm is forced. Forcing on a number the forced arm
    could not move latched the loop into 1400 consecutive 6-second regate
    cycles on 2026-08-24, and the bandit never chose again.
    """
    if (debt or 0) < REGATE_GAP:
        return None
    for e in history or []:                # newest-first, as auto_loop stores it
        if e.get("action") == "search":
            return None if arm_of(e.get("settings")) == "regate" else "regate"
    return "regate"


def _load(base_key=None):
    """(scheduler, credited, hours). A corrupt or missing blob reads as fresh.

    When the campaign's base key has moved, the stored evidence was earned on a
    different scale and is halved rather than trusted or thrown away. Same
    treatment ar_rl already gives its search scheduler.
    """
    st = ar_memory.blob_get(STATE_KEY)
    if not isinstance(st, dict):
        st = {}
    sched = ar_rl.Scheduler(st.get("scheduler"), arms=ARMS, phases=(PHASE,))
    if base_key and st.get("base_key") and st["base_key"] != base_key:
        sched.halve()
    credited = st.get("credited")
    hours = st.get("hours")
    return (sched,
            set(credited) if isinstance(credited, list) else set(),
            dict(hours) if isinstance(hours, dict) else {})


def _save(sched, credited, hours, base_key=None):
    ar_memory.blob_put(STATE_KEY, {"version": 1, "scheduler": sched.to_state(),
                                   "credited": sorted(credited),
                                   "hours": hours, "base_key": base_key})


def posteriors():
    """{arm: posterior mean} for the display. Never raises."""
    sched, _c, _h = _load()
    return {a: sched.posterior_mean(a, PHASE) for a in ARMS}


def measured_hours(history, hours):
    """Replace a cost prior with the measured median once an arm has three
    finished cycles. A prior that never updates is a guess that never learns."""
    seen = {}
    for e in history or []:
        arm = arm_of(e.get("settings"))
        if arm and e.get("rc") == 0 and e.get("seconds"):
            seen.setdefault(arm, []).append(float(e["seconds"]) / 3600.0)
    for arm, vals in seen.items():
        if len(vals) >= 3:
            vals.sort()
            hours[arm] = vals[len(vals) // 2]
    return hours


def choose(history, findings, outcomes, replicated_sigs, sig_of, rng=None,
           base_key=None):
    """(arm, settings) for the next cycle, after paying out what has settled.

    A Beta posterior wants a 0/1 outcome and the reward is continuous, so the
    update is stochastic: reward/REWARD_CAP is the success probability. That
    keeps ar_rl's math untouched and makes a large reward likelier to teach
    than a small one without letting one lucky cycle saturate an arm.
    """
    sched, credited, hours = _load(base_key)
    for rec in settle(history, findings, outcomes, replicated_sigs, credited,
                      sig_of):
        p = min(1.0, rec["reward"] / REWARD_CAP)
        sched.update(rec["arm"], PHASE, (rng or _RNG).random() < p)
        credited.add(rec["key"])
    hours = measured_hours(history, hours)
    arm = forced_arm(replication_debt(findings, replicated_sigs, sig_of),
                     history)
    if arm is None:
        arm, _floor = sched.choose(legal_arms(), PHASE)
    _save(sched, credited, hours, base_key)
    return arm, settings_of(arm)


DEAD_ARM = "__control_dead__"       # never earns anything, by construction
BOOT_N = 400


def _boot_ci(vals, rng):
    """Percentile bootstrap interval for a mean. Wide is the honest answer on
    five samples, and a wide interval is what stops a ranking being printed."""
    if not vals:
        return (0.0, 0.0)
    means = []
    for _ in range(BOOT_N):
        means.append(sum(rng.choice(vals) for _ in vals) / len(vals))
    means.sort()
    return (means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N) - 1])


def replay(history, findings, outcomes, replicated_sigs, sig_of, rng=None):
    """What each arm would have earned per hour over the recorded history.

    A dead arm is mixed in on purpose. If the intervals cannot separate it from
    the best real arm, this replay is not measuring anything and the report
    says so instead of printing a ranking.
    """
    rng = rng or random.Random(0)
    per_arm = {}
    per_chooser = {"llm": [], "rl": []}
    read = skipped = 0
    for rec in settle(history, findings, outcomes, replicated_sigs, set(),
                      sig_of):
        per_arm.setdefault(rec["arm"], []).append(rec["reward"])
    for e in history or []:
        if e.get("action") != "search":
            continue
        if arm_of(e.get("settings")) is None:
            skipped += 1
        else:
            read += 1
            by = e.get("chosen_by")
            if by in per_chooser:
                per_chooser[by].append(e)
    per_arm[DEAD_ARM] = [0.0] * max(1, read)
    rows = {}
    for arm, vals in per_arm.items():
        lo, hi = _boot_ci(vals, rng)
        rows[arm] = {"n": len(vals), "mean": sum(vals) / len(vals),
                     "lo": lo, "hi": hi,
                     "hours": RECIPES.get(arm, {}).get("hours")}
    live = {a: r for a, r in rows.items() if a != DEAD_ARM}
    best = max(live.values(), key=lambda r: r["mean"], default=None)
    separated = bool(best and best["lo"] > rows[DEAD_ARM]["hi"])
    return {"rows": rows, "skipped": skipped, "read": read,
            "control_separated": separated,
            "by_chooser": {k: len(v) for k, v in per_chooser.items()}}


def replay_lines(rep):
    """The replay as console text. Refuses to rank when the control failed."""
    out = ["=== CAMPAIGN DIRECTOR REPLAY ===",
           "Read %d of %d recorded search cycles (%d predate the settings field)."
           % (rep["read"], rep["read"] + rep["skipped"], rep["skipped"]),
           "Chosen by: llm %d, rl %d" % (rep["by_chooser"].get("llm", 0),
                                         rep["by_chooser"].get("rl", 0)), ""]
    if not rep["control_separated"]:
        out += ["The control arm, which earns nothing by construction, is NOT",
                "separated from the best real arm. This replay is not measuring",
                "anything yet, so no ranking is printed. Run more cycles.", ""]
        return out
    out.append("REWARD PER HOUR (95 percent bootstrap interval), best arm first")
    for arm, r in sorted(rep["rows"].items(), key=lambda kv: -kv[1]["mean"]):
        if arm == DEAD_ARM:
            continue
        out.append("  %-18s %+.4f  [%+.4f, %+.4f]  n=%d"
                   % (arm, r["mean"], r["lo"], r["hi"], r["n"]))
    return out
