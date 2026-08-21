"""Stage B of the timing program: a fitted-Q challenger to the Stage-A rules.

Stage A is eight interpretable parameters fitted by ES. This is the same
decision, learned as a value function instead: Q(s, a) over a BINARY action
that the current position interprets, exactly as the rules do. The authority
is unchanged and cannot widen, because the forced cases live in the shared
transition function and not in the model: no signal means stay out, a flipped
signal means exit, and the policy never picks a direction or a size.

One transition function, `advance`, is used by the environment that generates
the training data AND by the policy that acts. A separate "simulator" and
"policy" would be two chances to disagree about what an action means, and the
disagreement would show up as a policy that scores well while it is fitted and
badly when it runs.

stdlib + numpy + catboost + core.timing_policy only. Series come from
train_timing.build_asset_series; nothing here reads a database or a model file.
"""
import os

import numpy as np

from core import timing_policy as tp
from core.logger import get_logger

_logger = get_logger("timing_fqi")

# The state vector of spec 4.1, plus the action as the last column so one
# regressor carries Q(s, a) instead of one model per action.
FEATURE_NAMES = ("prob", "prob_d1", "margin", "taleb_hi", "trend_up",
                 "atr_pct", "risky", "is_forex", "pos", "days_held",
                 "pnl_atr", "cool_left", "streak", "action")

TREND_WINDOW = 200
VOL_WINDOW = 60


def _rolling_mean(x, window):
    """Trailing mean, expanding until `window` bars exist. Never NaN."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    csum = np.concatenate(([0.0], np.cumsum(np.nan_to_num(x))))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def _rolling_pct_rank(x, window):
    """Fraction of the trailing window at or below each value, in [0, 1].

    Expanding while the window is not full, so the warm-up carries a real
    reading rather than a NaN the fit would have to be told about.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        seg = x[lo:i + 1]
        out[i] = float(np.mean(seg <= x[i]))
    return out


def series_features(series):
    """The position-independent half of the state, one array per feature."""
    probs = np.asarray(series["probs"], dtype=float)
    close = np.asarray(series["close"], dtype=float)
    atr = np.nan_to_num(np.asarray(series["atr"], dtype=float), nan=0.0)
    buy, sell = float(series["buy_thr"]), float(series["sell_thr"])
    # Signed distance to the threshold the signal on that side has to clear.
    # Positive means the bar is on the far side of its own threshold.
    margin = np.where(probs >= 0.5, probs - buy, sell - probs)
    return {
        "prob": probs,
        "prob_d1": np.diff(probs, prepend=probs[:1]),
        "margin": margin,
        "taleb_hi": np.asarray(series["taleb_hi"], dtype=float),
        "trend_up": (close > _rolling_mean(close, TREND_WINDOW)).astype(float),
        "atr_pct": _rolling_pct_rank(atr, VOL_WINDOW),
        "atr": atr,
        "risky": 1.0 if series.get("risky") else 0.0,
        "is_forex": 1.0 if series.get("is_forex") else 0.0,
        "buy_thr": buy, "sell_thr": sell,
        "n": len(probs),
    }


def state_row(feat, i, pos, days_held, pnl_atr, cool_left, streak, action):
    """One (state, action) row in FEATURE_NAMES order."""
    return [feat["prob"][i], feat["prob_d1"][i], feat["margin"][i],
            feat["taleb_hi"][i], feat["trend_up"][i], feat["atr_pct"][i],
            feat["risky"], feat["is_forex"],
            float(pos), float(days_held), float(pnl_atr), float(cool_left),
            float(streak), float(action)]


ACT_NO, ACT_YES = 0, 1


def _raw_side(feat, i):
    p = float(feat["prob"][i])
    return 1 if p > feat["buy_thr"] else (-1 if p < feat["sell_thr"] else 0)


def forced_action(raw, pos, cool_left):
    """The action spec 4.2 forces here, or None when the model may choose.

    Three cases are not the model's to decide, and keeping them here rather
    than hoping a fitted Q learns them is what stops Stage B from quietly
    holding more authority than Stage A: no signal, a signal that flipped
    against an open position, and a live cooldown.
    """
    if pos == 0:
        if raw == 0 or cool_left > 0:
            return ACT_NO
        return None
    if raw == -pos:
        return ACT_NO          # a flip closes; re-entry is a later ENTER
    return None


def advance(state, feat, i, action):
    """One bar. Returns (new_state, label, reason).

    label is ENTER / STAY_OUT / HOLD / EXIT, the same vocabulary Stage A logs,
    so the telemetry and the display layer read a Stage-B decision unchanged.
    reason is "forced" when spec 4.2 decided it and "model" when the action did.
    """
    st = dict(state)
    raw = _raw_side(feat, i)

    last_raw = st.get("last_raw", 0)
    if raw != 0 and raw == last_raw:
        st["streak"] = st.get("streak", 0) + 1
    elif raw != 0:
        st["streak"] = 1
    else:
        st["streak"] = 0
    st["last_raw"] = raw

    if st.get("cooldown_left", 0) > 0:
        st["cooldown_left"] -= 1

    forced = forced_action(raw, st["pos"], state.get("cooldown_left", 0))
    reason = "forced" if forced is not None else "model"
    act = forced if forced is not None else int(action)

    if st["pos"] == 0:
        if act == ACT_YES:
            st.update(pos=raw, days_held=0, seg_peak=0.0, seg_ret=0.0)
            return st, "ENTER", reason
        return st, "STAY_OUT", reason

    if act == ACT_YES:
        st["days_held"] += 1
        return st, "HOLD", reason
    st.update(pos=0, days_held=0, seg_peak=0.0, seg_ret=0.0)
    return st, "EXIT", reason


def _stage_a_action(policy, feat, i, st):
    """What the Stage-A rules would do here, as a binary action.

    Asked of policy_step itself rather than reimplemented, so the behaviour
    cloned into the transition store is the behaviour that was adopted.
    """
    action, _reason, _new = tp.policy_step(
        policy, float(feat["prob"][i]), feat["buy_thr"], feat["sell_thr"],
        float(feat["atr"][i]), bool(feat["taleb_hi"][i]),
        bool(feat["risky"]), st)
    return ACT_YES if action in ("ENTER", "HOLD") else ACT_NO


def rollout(series, policy, rng, epsilon=0.1, costs=(0.0, 0.0)):
    """Transitions under `policy` with epsilon-greedy noise on the action.

    next_rows carries BOTH candidate actions of the next state, which is what
    the Bellman target needs: max over a' of Q(s', a').
    """
    feat = series_features(series)
    next_ret = np.asarray(series["next_ret"], dtype=float)
    comm, slip = costs
    n = feat["n"]
    rows = np.zeros((n, len(FEATURE_NAMES)), dtype=float)
    next_rows = np.zeros((n, 2, len(FEATURE_NAMES)), dtype=float)
    rewards = np.zeros(n, dtype=float)
    terminal = np.zeros(n, dtype=bool)
    labels = []

    st = dict(tp.FRESH_STATE)
    for i in range(n):
        pnl_atr = (st["seg_ret"] / feat["atr"][i]) if feat["atr"][i] else 0.0
        act = _stage_a_action(policy, feat, i, st)
        if epsilon > 0.0 and rng.random() < epsilon:
            act = 1 - act
        rows[i] = state_row(feat, i, st["pos"], st["days_held"], pnl_atr,
                            st.get("cooldown_left", 0), st.get("streak", 0),
                            act)
        pos_before = st["pos"]
        st, label, _reason = advance(st, feat, i, act)
        labels.append(label)

        r = float(next_ret[i])
        r = 0.0 if np.isnan(r) else r
        traded = abs(st["pos"] - pos_before)
        rewards[i] = st["pos"] * r - traded * (comm + slip)

        if st["pos"] != 0:
            st["seg_ret"] += st["pos"] * r
            st["seg_peak"] = max(st["seg_peak"], st["seg_ret"])

        j = min(i + 1, n - 1)
        pnl_next = (st["seg_ret"] / feat["atr"][j]) if feat["atr"][j] else 0.0
        for a in (ACT_NO, ACT_YES):
            next_rows[i, a] = state_row(
                feat, j, st["pos"], st["days_held"], pnl_next,
                st.get("cooldown_left", 0), st.get("streak", 0), a)
        terminal[i] = (i == n - 1)
    return {"rows": rows, "rewards": rewards, "next_rows": next_rows,
            "terminal": terminal, "labels": labels}


CB_PARAMS = {"iterations": 300, "depth": 6, "learning_rate": 0.05,
             "loss_function": "RMSE", "verbose": 0,
             "allow_writing_files": False}


def q_value(model, row):
    """Q for one (state, action) row."""
    return float(model.predict(np.asarray([row], dtype=float))[0])


def _q_max(model, next_rows):
    """max over the two candidate actions of the next state, per transition."""
    m, _a, f = next_rows.shape
    flat = next_rows.reshape(m * 2, f)
    vals = model.predict(flat).reshape(m, 2)
    return vals.max(axis=1)


def fit_q(batches, iters=6, gamma=0.97, seed=0):
    """Fitted Q-iteration. Returns the model after each iteration.

    Q_0 is the immediate reward, which is the honest starting point: with no
    model yet, the value of a state IS what it paid. Each later iteration adds
    one bar of lookahead, so the list is a ladder of horizons and VAL picks the
    rung, instead of a fixed depth chosen because it sounded right.
    """
    from catboost import CatBoostRegressor
    rows = np.concatenate([b["rows"] for b in batches])
    rewards = np.concatenate([b["rewards"] for b in batches])
    next_rows = np.concatenate([b["next_rows"] for b in batches])
    terminal = np.concatenate([b["terminal"] for b in batches])

    models, targets = [], rewards.copy()
    for k in range(iters):
        model = CatBoostRegressor(random_seed=seed + k, **CB_PARAMS)
        model.fit(rows, targets)
        models.append(model)
        if k + 1 < iters:
            bootstrap = _q_max(model, next_rows)
            bootstrap[terminal] = 0.0
            targets = rewards + gamma * bootstrap
    return models


class FqiPolicy:
    """Greedy over Q, with spec 4.2 still forced outside the model.

    Duck-types RulesPolicy where it matters: apply_series returns the same
    (sides, actions, reasons) triple, so train_timing's evaluator, the gate and
    any later display read a Stage-B decision with no special case.
    """

    def __init__(self, model, params=None):
        self.model = model
        self.params = dict(params or {"stage": "B"})

    def act(self, feat, i, st):
        """The action for one bar: forced when the spec forces it, else argmax."""
        raw = _raw_side(feat, i)
        forced = forced_action(raw, st["pos"], st.get("cooldown_left", 0))
        if forced is not None:
            return forced
        pnl_atr = (st["seg_ret"] / feat["atr"][i]) if feat["atr"][i] else 0.0
        rows = [state_row(feat, i, st["pos"], st["days_held"], pnl_atr,
                          st.get("cooldown_left", 0), st.get("streak", 0), a)
                for a in (ACT_NO, ACT_YES)]
        vals = self.model.predict(np.asarray(rows, dtype=float))
        return ACT_YES if vals[ACT_YES] > vals[ACT_NO] else ACT_NO

    def apply_series(self, series):
        feat = series_features(series)
        next_ret = (np.asarray(series.get("next_ret"), dtype=float)
                    if series.get("next_ret") is not None else None)
        n = feat["n"]
        sides = np.zeros(n, dtype=int)
        actions, reasons = [], []
        st = dict(tp.FRESH_STATE)
        for i in range(n):
            act = self.act(feat, i, st)
            st, label, reason = advance(st, feat, i, act)
            sides[i] = st["pos"]
            actions.append(label)
            reasons.append(reason)
            if st["pos"] != 0 and next_ret is not None:
                r = float(next_ret[i])
                if not np.isnan(r):
                    st["seg_ret"] += st["pos"] * r
                    st["seg_peak"] = max(st["seg_peak"], st["seg_ret"])
        return sides, actions, reasons

# --- Serving --------------------------------------------------------------
# The plan's follow-up trigger, written before the verdict was known: "on an
# ADOPT verdict, the follow-up task is a loader that returns the FQI policy and
# a policy_step equivalent fed by the context core/scoring.py already computes."
# It adopted on 2026-08-22 (mean_d +3.672, p_bh 0.0000, 314 assets), so here it
# is. Default OFF, exactly as Stage A shipped: GTRADE_TIMING_STAGE=b is what
# moves serving off the rules, and without it this file changes nothing.

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "timing_fqi.cbm")


def stage_b_on():
    """True when serving should read the fitted Q instead of the Stage-A rules.

    Separate from GTRADE_TIMING_POLICY on purpose: that flag decides whether a
    timing layer runs at all, this one decides WHICH. Turning Stage B on with
    the timing layer off does nothing, which is the safe direction.
    """
    return (os.getenv("GTRADE_TIMING_STAGE") or "").strip().lower() == "b"


def load_served_policy(path=None):
    """FqiPolicy from timing_fqi.cbm, or None when nothing is adopted.

    Same contract as timing_policy.load_policy: absent, unreadable or nonsense
    all return None so the caller falls back to the rules rather than to
    nothing. An install that never ran `--stage b` behaves as it always did.
    """
    from catboost import CatBoostRegressor

    try:
        model = CatBoostRegressor()
        model.load_model(path or MODEL_PATH)
    except Exception as exc:
        _logger.debug("Stage-B model not loaded: %s", exc)
        return None
    return FqiPolicy(model)


def serve_step(policy, prob, prev_prob, buy_thr, sell_thr, close_hist, atr_hist,
               taleb_hi, risky, is_forex, st):
    """One bar's Stage-B decision. Returns (label, reason, new_state).

    Same shape as timing_policy.policy_step so core/scoring.py reads one or the
    other with no special case.

    Deliberately builds a short series and calls series_features / act / advance
    rather than recomputing the state row inline: trend_up and atr_pct are
    trailing-window features, and a second definition of them at serve time is
    exactly how a served decision drifts from the fitted one without anybody
    seeing it. `close_hist` and `atr_hist` must be the trailing bars ending at
    today - TREND_WINDOW of close and VOL_WINDOW of atr are what the last index
    actually reads, and both rollings expand while short, so a shorter tail
    degrades quietly instead of failing. Pass the full tails.
    """
    close = np.asarray(close_hist, dtype=float)
    atr = np.nan_to_num(np.asarray(atr_hist, dtype=float), nan=0.0)
    n = min(len(close), len(atr))
    if n < 2:
        raise ValueError("serve_step needs at least two trailing bars")
    close, atr = close[-n:], atr[-n:]

    # Only the last two probabilities are read (prob and prob_d1 at index -1);
    # the earlier slots exist so every array in the series has one length.
    probs = np.full(n, float(prev_prob if prev_prob is not None else prob))
    probs[-1] = float(prob)

    feat = series_features({
        "probs": probs, "close": close, "atr": atr,
        "taleb_hi": np.full(n, 1.0 if taleb_hi else 0.0),
        "buy_thr": float(buy_thr), "sell_thr": float(sell_thr),
        "risky": bool(risky), "is_forex": bool(is_forex),
    })
    i = n - 1
    action = policy.act(feat, i, st)
    new_st, label, reason = advance(st, feat, i, action)
    return label, reason, new_st
