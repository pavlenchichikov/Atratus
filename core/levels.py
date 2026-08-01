"""Concrete trade levels for manual execution: where to enter, where to bail.

The models emit BUY / SELL / WAIT and core/positions.py turns those into held
positions, but neither ever produces a price. This module does, and only that:
an entry zone around the last close, an emergency stop a few ATR away, and a
trailing version of that stop once a position has been held for more than one
bar.

Pure and serve-side: bars and numbers in, a dict out. No database, no models,
no file. There is deliberately no take-profit: the model is trained on
"tomorrow's close beats today's" (core/features.py:53), so a target would be a
payoff distribution nothing in this project has measured.

The ATR here duplicates the expression in core/features.py:170-176 because
there it is welded into a whole feature frame over a DataFrame and cannot be
called on its own. tests/test_levels.py pins the two together on shared input.
"""

ATR_PERIOD = 14
K_ENTRY = 0.5
K_STOP = 2.0


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


def levels(bars, signal, segment=None, k_entry=K_ENTRY, k_stop=K_STOP):
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
