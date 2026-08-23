"""Concrete trade levels for manual execution: where to enter, where to bail.

The models emit BUY / SELL / WAIT and core/positions.py turns those into held
positions, but neither ever produces a price. This module does, and only that:
an entry zone around the last close, an emergency stop a few ATR away, and a
trailing version of that stop once a position has been held for more than one
bar.

Serve-side and almost pure: bars and numbers in, a dict out. No database and no
models. The one file it reads is levels_policy.json, the fitted multipliers, and
it falls back to the constants below whenever that file is absent, corrupt or
nonsense, so an install that never fitted anything behaves as it always did. There is deliberately no take-profit: the model is trained on
"tomorrow's close beats today's" (core/features.py:53), so a target would be a
payoff distribution nothing in this project has measured.

The ATR here duplicates the expression in core/features.py:170-176 because
there it is welded into a whole feature frame over a DataFrame and cannot be
called on its own. tests/test_levels.py pins the two together on shared input.
"""

import os

ATR_PERIOD = 14
K_ENTRY = 0.5
K_STOP = 2.0

# Named, not recomputed inside each reader, for the same reason
# core/timing_policy.py names its own: the two functions below have to agree on
# which file is the policy, and a test has to be able to point them somewhere
# else. Without a seam here the suite reads whatever the developer last fitted,
# which is not what the shipped constants say and not what CI sees.
POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "levels_policy.json")

# A Taleb risk above this reads as the fat-tail regime. The same number
# train_timing.py uses to build its taleb_hi flag, so the policy is fitted on
# the regime serving will report.
TALEB_HI = 0.7

# The policy's genes. The four deltas default to zero, so a policy that carries
# only k_entry and k_stop behaves exactly like one that never heard of regimes,
# and the regime-conditioned form has the flat form as its own baseline.
POLICY_DEFAULTS = {
    "k_entry": K_ENTRY, "k_stop": K_STOP,
    "d_entry_hi_taleb": 0.0, "d_entry_risky": 0.0,
    "d_stop_hi_taleb": 0.0, "d_stop_risky": 0.0,
}


def effective_multipliers(params, taleb_hi=False, risky=False):
    """(k_entry, k_stop) for one bar's regime.

    Additive deltas on a base, which is how core/timing_policy resolves its own
    regime variants: both conditions can apply at once, and a zero delta is a
    no-op rather than a special case. Clamped positive because a non-positive
    multiplier is not a wider or tighter level, it is a level on top of the
    close and a stop on the wrong side of it.
    """
    p = dict(POLICY_DEFAULTS)
    p.update(params or {})
    k_entry, k_stop = p["k_entry"], p["k_stop"]
    if taleb_hi:
        k_entry += p["d_entry_hi_taleb"]
        k_stop += p["d_stop_hi_taleb"]
    if risky:
        k_entry += p["d_entry_risky"]
        k_stop += p["d_stop_risky"]
    return max(0.01, float(k_entry)), max(0.01, float(k_stop))


def _side(signal):
    """+1 for a long (BUY), -1 for a short (SELL), 0 for flat (WAIT/other)."""
    s = (signal or "").upper()
    if s == "BUY":
        return 1
    if s == "SELL":
        return -1
    return 0


def _true_ranges(bars):
    """Per-bar true range. The first bar has no previous close, so it falls back
    to high-low, which is what the pandas expression in features.py does (its
    max skips the NaN)."""
    out = []
    prev_close = None
    for b in bars:
        high, low = b["high"], b["low"]
        tr = high - low
        if prev_close is not None:
            tr = max(tr, abs(high - prev_close), abs(low - prev_close))
        out.append(tr)
        prev_close = b["close"]
    return out


def atr_series(bars, period=ATR_PERIOD):
    """ATR aligned one-to-one with `bars`, None until `period` bars exist."""
    tr = _true_ranges(bars)
    out = []
    for i in range(len(tr)):
        if i + 1 < period:
            out.append(None)
        else:
            window = tr[i + 1 - period:i + 1]
            out.append(sum(window) / period)
    return out


def atr_abs(bars, period=ATR_PERIOD):
    """Absolute ATR of the last bar, or None when there is not enough history."""
    if len(bars) < period:
        return None
    return atr_series(bars, period)[-1]


def _trailing_stop(bars, atrs, segment, side, k_stop):
    """The best stop over an open segment: for a long, the highest
    close - k_stop * atr seen since entry (mirrored for a short). Derived from
    bar history, so nothing has to be persisted between runs."""
    start, end = segment.get("start_date"), segment.get("end_date")
    best = None
    seen = False
    for bar, atr in zip(bars, atrs):
        date = bar["date"]
        if start and date < start:
            continue
        if end and date > end:
            continue
        if atr is None or atr <= 0:
            continue
        seen = True
        level = bar["close"] - side * k_stop * atr
        if best is None:
            best = level
        elif side > 0:
            best = max(best, level)
        else:
            best = min(best, level)
    return best if seen else None


def load_policy(path=None):
    """The fitted multipliers, or None when nothing has been adopted.

    Kept BESIDE the constants rather than replacing them: an unfitted install, a
    reverted policy and a corrupt file must all fall back to the numbers
    production has always shipped, never to nothing. Same contract as
    core.timing_policy.load_policy.
    """
    import json

    path = path or POLICY_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            stored = (json.load(fh) or {}).get("params") or {}
        params = dict(POLICY_DEFAULTS)
        params.update({k: float(stored[k]) for k in POLICY_DEFAULTS if k in stored})
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if params["k_entry"] <= 0 or params["k_stop"] <= 0:
        return None
    return params


def policy_evidence(path=None):
    """What a reader needs to know about the levels on screen, or None.

    Levels change silently otherwise: a fitted policy moves every entry zone and
    every stop in the product with nothing saying so, which is the shape of
    change that cannot be checked. Returns the adoption date and the held-out
    evidence the fit was accepted on.
    """
    import json

    path = path or POLICY_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            body = json.load(fh) or {}
        gate = body.get("gate") or {}
        return {"adopted": body.get("adopted"), "p": gate.get("p"),
                "n": gate.get("n"), "mean_d": gate.get("mean_d")}
    except (OSError, ValueError, TypeError, AttributeError):
        return None


_TIMING_SIDE = {"ENTER:+1": "BUY", "ENTER:-1": "SELL",
                "EXIT": "WAIT", "STAY_OUT": "WAIT"}


def acting_side(signal, asset, timing_action):
    """The side levels belong to: the timing layer's, on any bar it decided.

    Levels answer "where to get in and where to bail", which only means
    something for the side actually being held. Once the timing layer decides,
    that side is its position and not the raw call: the two disagree on every
    bar the policy sits a signal out, and on every bar it holds through one
    that went quiet. A bar the layer did not decide - timing switched off, or
    its shadow skipped this asset - falls back to `signal`.

    `train_levels.sides_for` is the fitting half of this same definition, and
    the two have to move together: a journal recorded on one side and a policy
    fitted on another measures levels for trades nobody was in.

    Scope is the JOURNAL, not the screen. The card and the trade sheet still
    draw the conditional levels the raw call implies and mark a disagreeing
    policy with a badge, which answers "if this trade is taken, where", a
    different question from "which side is being held".
    """
    if not timing_action:
        return signal
    side = _TIMING_SIDE.get(timing_action)
    if side is not None:
        return side
    # HOLD names no side. The position it is holding does, and that is only
    # in the log, so a tracker that cannot answer falls back to the signal.
    try:
        from performance_tracker import timing_state
        pos = timing_state(asset)["pos"]
    except Exception:
        return signal
    return {1: "BUY", -1: "SELL"}.get(pos, "WAIT")


def issues_levels(timing_action):
    """Whether this bar OPENS a set of levels, or is only living inside one.

    Levels answer where to get in and where to bail, which is a question a
    position asks once, when it opens. Re-issuing them on every held bar writes
    rows that can never become a trade: 117 of the journal's first 248 closed
    as "not a setup: position already open".

    `train_levels._issue_bars` is the fitting half - it takes the timing
    policy's ENTER bars and nothing else - and the two have to move together or
    the fit is scoring one thing while the journal records another.

    A bar with no timing decision at all keeps issuing the way it always has:
    there is no ENTER to key on, and silently emptying the journal when the
    timing layer is switched off would be worse than an unresolvable row.
    """
    return not timing_action or timing_action.startswith("ENTER")


def levels(bars, signal, segment=None, k_entry=None, k_stop=None,
           taleb_hi=False, risky=False):
    """One sheet row for one asset.

    `bars`: oldest-first OHLC dicts with `date`, `high`, `low`, `close`.
    `signal`: the DISPLAY signal (already live-gated by track_record).
    `segment`: the open position segment from core.positions.build_positions,
    or None for a fresh setup.

    Returns `side`, `close`, `atr`, `entry_low`, `entry_high`, `stop`,
    `trailing`, `status`. Every failure is a `status`, never an exception and
    never a silently dropped row: a sheet with a gap in it is worse than a sheet
    that says why.
    """
    # An explicit multiplier is an experiment and always wins; otherwise the
    # adopted policy decides, resolved for THIS bar's regime.
    fit_entry, fit_stop = effective_multipliers(load_policy(), taleb_hi, risky)
    k_entry = fit_entry if k_entry is None else k_entry
    k_stop = fit_stop if k_stop is None else k_stop
    row = {"side": 0, "close": None, "atr": None, "entry_low": None,
           "entry_high": None, "stop": None, "trailing": False, "status": "ok"}
    side = _side(signal)
    row["side"] = side
    if side == 0:
        row["status"] = "no_signal"
        return row
    if not bars:
        row["status"] = "no_bars"
        return row
    row["close"] = bars[-1]["close"]
    if len(bars) < ATR_PERIOD:
        row["status"] = "short_history"
        return row
    atrs = atr_series(bars)
    atr = atrs[-1]
    if atr is None or atr <= 0:
        row["status"] = "flat_atr"
        return row
    row["atr"] = atr
    close = row["close"]
    row["entry_low"] = close - k_entry * atr
    row["entry_high"] = close + k_entry * atr
    stop = close - side * k_stop * atr
    if segment and segment.get("open") and segment.get("bars", 0) > 1:
        trailed = _trailing_stop(bars, atrs, segment, side, k_stop)
        if trailed is not None:
            stop = trailed
            row["trailing"] = True
    row["stop"] = stop
    # A trailing stop is the best level seen since entry, so on a position that
    # went the wrong way it stays back at the entry bar and the price is already
    # through it. Saying "stop 328.31" under a short trading at 333.40 reads as
    # a level still to come; it is a position that should already be closed.
    if (side > 0 and close <= stop) or (side < 0 and close >= stop):
        row["status"] = "stop_breached"
    return row


def size_for(close, stop, equity, risk_per_trade, max_single_position,
             kelly_pct=None):
    """Position size from the distance to the stop, clipped by the existing
    risk limits, plus the name of whichever limit actually bound it.

    Sizing off the stop is the point: a fixed fraction of capital risks a
    different amount on every asset, because the stop distance is what decides
    the loss. `close` is the reference price, not the zone edge, so the number
    does not move with where inside the zone the fill happens.
    """
    out = {"amount": None, "pct": 0.0, "bound_by": "no_stop"}
    distance = abs(close - stop) if (close and stop is not None) else 0.0
    if distance <= 0:
        return out
    pct = (risk_per_trade * close) / distance
    bound = "risk"
    if kelly_pct is not None and kelly_pct < pct:
        pct, bound = kelly_pct, "kelly"
    if max_single_position < pct:
        pct, bound = max_single_position, "max_single_position"
    out["pct"] = pct
    out["bound_by"] = bound
    # Without an equity number there is nothing to multiply by, so the caller
    # shows the percentage. The binding limit is still the useful part of the
    # answer and is reported either way.
    if equity:
        out["amount"] = equity * pct
    return out


def resolve_trade(side, entry_low, entry_high, stop, bars, sides, leg_cost):
    """What became of one issued set of levels, or None while it is undecided.

    ONE definition, shared by the live scorer (performance_tracker) and the
    policy fitter (train_levels). Two lookalike implementations would let the
    fitter optimise a rule that the journal does not score, which is the same
    class of mistake as optimising a basis nobody checked against the target.

    `bars` are the (open, high, low, close) bars AFTER the bar the levels were
    computed from; `sides` is the signal side in force on each of those bars
    (+1 long, -1 short, 0 flat), same length. The trade ends on the stop or on
    the side turning away from the one it was issued for, which is what the
    asset card tells a person to do and what core/positions.py calls a segment.

    Fills are limit-order conventions taken at the WORSE edge of the zone, and a
    gap through the stop fills at the gap, so nothing here flatters a result.
    """
    if side not in (1, -1):
        return None
    entry_price = None
    entry_index = None
    held = 0
    for i, (op, hi, lo, cl) in enumerate(bars):
        flipped = i < len(sides) and sides[i] != side
        if entry_price is None:
            touched = (lo <= entry_high) if side > 0 else (hi >= entry_low)
            if touched:
                entry_price = min(op, entry_high) if side > 0 else max(op, entry_low)
                entry_index = i
                continue          # costs and stops are judged from the next bar
            if flipped:
                return {"entered": 0, "exit_reason": "no_entry", "ret_net": 0.0,
                        "bars_held": 0}
            continue
        held += 1
        hit = (lo <= stop) if side > 0 else (hi >= stop)
        if hit:
            exit_price = min(op, stop) if side > 0 else max(op, stop)
            reason = "stop"
        elif flipped:
            exit_price = cl
            reason = "signal"
        else:
            continue
        gross = side * (exit_price - entry_price) / entry_price
        return {"entered": 1, "entry_index": entry_index,
                "entry_price": float(entry_price), "exit_index": i,
                "exit_price": float(exit_price), "exit_reason": reason,
                "bars_held": held, "ret_net": float(gross - 2 * leg_cost)}
    return None                    # still running, or not enough bars yet
