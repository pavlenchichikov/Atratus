"""Stage C of the timing program: keep the Q learning, with an anchor.

208 assets times one daily bar is 208 transitions a day, against roughly
760 000 already in the offline fit. One day is 0.03 percent of what the model
has seen, so anything that lets a day move the policy noticeably is not
learning, it is drift with a fast clock. Everything here follows from that.

The tick refits on a cadence and then has to be talked out of accepting the
result: a new generation must stay inside a trust region around the ADOPTED
Stage-A rules, and it must beat the generation currently in shadow. The anchor
is Stage A permanently and never the previous generation, because a chain of
small approved steps, each compared only to the one before it, can walk
anywhere.

Pure: series in, decision out. `sides_of` is injected rather than imported, the
rule core/ar_rl.py already follows, which is also what keeps train_timing out
of the import graph of a module train_timing uses.
"""
import json
import os

import numpy as np

# A generation that disagrees with the anchor more often than this has stopped
# being a timing overlay on the rules and become a different strategy. That is
# a decision for a human and a full gate, not for a weekly cron.
AGREE_FLOOR = 0.80


def agreement(by_asset, policy, anchor, sides_of):
    """Fraction of bars where the two policies hold the same position.

    Pooled over assets rather than averaged per asset: a 6000-bar asset carries
    more evidence about how far the policy has wandered than a 300-bar one, and
    averaging would give them the same vote.
    """
    same = total = 0
    for series in by_asset.values():
        a = np.asarray(sides_of(policy, series), dtype=int)
        b = np.asarray(sides_of(anchor, series), dtype=int)
        n = min(len(a), len(b))
        if not n:
            continue
        same += int(np.sum(a[:n] == b[:n]))
        total += n
    return (same / total) if total else 0.0


MAX_ROLLBACKS = 2       # two losers in a row means the mechanism is broken


def fresh_state():
    """A stack with nothing on it: the anchor alone."""
    return {"generation": 0, "score": None, "agreement": None,
            "consecutive_rollbacks": 0, "halted": False,
            "anchor": "stage_a", "journal": []}


def decide(state, agree, challenger_score, floor=AGREE_FLOOR):
    """(verdict, reason) for one candidate generation.

    Order matters. The trust region is checked FIRST, because a candidate that
    has wandered outside it is refused whatever it scores: a policy allowed in
    on its score alone is a policy with no anchor, and the whole point of the
    stage is that there is one.
    """
    if state.get("halted"):
        return "HALT", "the schedule is halted; the anchor stage_a is live"
    if agree < floor:
        return "REJECT", ("agreement %.3f is below the trust region floor %.2f"
                          % (agree, floor))
    live = state.get("score")
    if live is not None and challenger_score <= live:
        if state.get("consecutive_rollbacks", 0) + 1 >= MAX_ROLLBACKS:
            return "HALT", ("%d generations in a row lost; falling back to the "
                            "anchor stage_a and stopping"
                            % (state["consecutive_rollbacks"] + 1))
        return "ROLLBACK", ("candidate %.4f lost to the live generation %.4f"
                            % (challenger_score, live))
    return "ACCEPT", "inside the trust region and ahead of the live generation"


def apply_decision(state, verdict, reason, agree, challenger_score, ts):
    """The state after one tick. Journalled either way, including the quiet
    ticks: a week that changed nothing has to be visible as a week that changed
    nothing rather than as a gap in the record."""
    st = dict(state)
    st["journal"] = list(state.get("journal") or [])
    st["journal"].append({"ts": ts, "verdict": verdict, "reason": reason,
                          "agreement": round(float(agree), 4),
                          "score": round(float(challenger_score), 4),
                          "generation": st.get("generation", 0)})
    if verdict == "ACCEPT":
        st["generation"] = st.get("generation", 0) + 1
        st["score"] = float(challenger_score)
        st["agreement"] = float(agree)
        st["consecutive_rollbacks"] = 0
    elif verdict == "ROLLBACK":
        st["consecutive_rollbacks"] = st.get("consecutive_rollbacks", 0) + 1
    elif verdict == "HALT":
        st["halted"] = True
        st["consecutive_rollbacks"] = st.get("consecutive_rollbacks", 0) + 1
    return st


STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "timing_online.json")


def online_on():
    return os.getenv("GTRADE_TIMING_ONLINE") == "1"


def load_state(path=None):
    """The stack, or a fresh one. Never raises: this runs on a schedule, and a
    scheduled job that dies on a malformed file stops silently."""
    try:
        with open(path or STATE_PATH, encoding="utf-8") as fh:
            blob = json.load(fh)
        if not isinstance(blob, dict) or "generation" not in blob:
            return fresh_state()
        st = fresh_state()
        st.update({k: blob[k] for k in st if k in blob})
        return st
    except (OSError, ValueError, TypeError):
        return fresh_state()


def save_state(state, path=None):
    with open(path or STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1, default=float)


CHAMPION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "timing_online_q.cbm")


def save_champion(model, path=None):
    """Keep the accepted generation's Q so the NEXT tick can roll out with it.

    The state file carries a generation number and no model, which was enough
    while every tick refitted from the anchor's trajectory. Collecting data
    under the current champion needs the champion itself to survive the run.

    Never raises: this is a scheduled job, and failing to cache a model is not
    a reason to lose the verdict that was just computed.
    """
    try:
        model.save_model(path or CHAMPION_PATH)
        return True
    except Exception:
        return False


def load_champion(path=None):
    """The accepted generation's Q as an FqiPolicy, or None.

    None on anything unexpected - a missing file on the first ever tick, a
    model written by another CatBoost, a half-written file from a killed run.
    The caller falls back to the anchor, which is the behaviour this replaced
    and is always safe.
    """
    try:
        from catboost import CatBoostRegressor

        from core.timing_fqi import FqiPolicy

        target = path or CHAMPION_PATH
        if not os.path.exists(target):
            return None
        m = CatBoostRegressor()
        m.load_model(target)
        return FqiPolicy(m)
    except Exception:
        return None
