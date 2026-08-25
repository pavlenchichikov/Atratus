"""The read-only dossier the analyst reasons over.

Assembled from modules that already exist. No new data source and no new
fetcher: every field here is something the project already computes for some
other panel.

What is NOT here is the design. The ensemble probability, the emitted signal,
the timing action and the sizing decision are excluded, because a second
opinion built on the first one's output is not a second opinion. FORBIDDEN_KEYS
and the test over it are what keep that true after this file stops being read.
"""

import hashlib
import json

from core.levels import ATR_PERIOD, atr_abs
from core.track_record import ohlc_series

FORBIDDEN_KEYS = frozenset({
    "probability", "cb_prob", "lstm_prob", "meta_prob", "signal",
    "sig_shown", "timing_action", "timing_reason", "timing_stage",
    "shadow_action", "gate_reason", "model_version", "correct",
    # actual_next_ret is not an ensemble channel, it is the realized outcome.
    # A dossier carrying it would be look-ahead, which is worse than the
    # failure this set was written to catch.
    "actual_next_ret",
})

HISTORY_BARS = 120
RECENT_BARS = 20


def _pct(a, b):
    if not a or not b:
        return None
    return (b - a) / a


def _safe(fn, default=None):
    """A dossier field is worth less than the run that produces it."""
    try:
        return fn()
    except Exception:
        return default


def _context(asset):
    """The non-price half of the dossier: fundamentals, events, macro.

    Each source reaches a database or the network on its own, so each call is
    wrapped in `_safe`: one dead source must not stop the day's run, and it
    must not stop `build()` from returning a fixed-shape dossier either.
    """
    from config import FULL_ASSET_MAP
    from core import events  # imported here, not at module level: both reach
    from core.dashboard import guru_for_asset  # the network

    verdict = _safe(lambda: guru_for_asset(asset)) or {}
    earnings = _safe(
        lambda: events.earnings_for({asset: FULL_ASSET_MAP[asset]})) or {}
    return {
        "guru_verdict": verdict.get("verdict"),
        "guru_pct": verdict.get("pct"),
        "next_earnings": earnings.get(asset),
        "macro_events": _safe(lambda: [e["name"] for e in events.load_macro()],
                              default=[]),
    }


def build(asset, db_path=None, today=None):
    """One asset's dossier. Missing pieces are None, never absent, so that the
    prompt and the hash have a fixed shape regardless of what was available."""
    bars = ohlc_series(asset, days=HISTORY_BARS, db_path=db_path)
    if today is not None:
        bars = [b for b in bars if b["date"] <= today]

    close = bars[-1]["close"] if bars else None
    atr = atr_abs(bars) if len(bars) >= ATR_PERIOD else None
    closes = [b["close"] for b in bars]

    return {
        "asset": asset,
        "date": bars[-1]["date"] if bars else None,
        "close": close,
        "atr": atr,
        "atr_pct": (atr / close) if (atr and close) else None,
        "ret_1": _pct(closes[-2], closes[-1]) if len(closes) >= 2 else None,
        "ret_5": _pct(closes[-6], closes[-1]) if len(closes) >= 6 else None,
        "ret_20": _pct(closes[-21], closes[-1]) if len(closes) >= 21 else None,
        "high_20": max(closes[-RECENT_BARS:]) if closes else None,
        "low_20": min(closes[-RECENT_BARS:]) if closes else None,
        "bars_available": len(bars),
        **_context(asset),
    }


def dossier_hash(dossier):
    """A stable 16-hex digest, used as the LLM cache key.

    Rounded before hashing: an unchanged dossier must hash the same across
    runs, and raw floats out of sqlite do not reliably do that.
    """
    rounded = {k: (round(v, 8) if isinstance(v, float) else v)
               for k, v in sorted(dossier.items())}
    blob = json.dumps(rounded, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
