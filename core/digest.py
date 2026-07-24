"""Pure diff over prediction_log history rows: what changed, per asset.

Mirror of the client's lib/digest.dart. Paired with a shared fixture
(tests/fixtures/digest_cases.json) that serves as the parity harness,
validating both implementations to prevent silent drift.
Timing behaviour is inherited, not re-implemented: fetch_history_rows already
sets timing_label to None unless the timing policy is on AND the label diverges
from the signal, so with the flag off no timing event can be produced here.
"""

from dataclasses import dataclass

FLIP = "flip"
ENTRY_BUY = "entry_buy"
ENTRY_SELL = "entry_sell"
EXIT = "exit"
TIMING_CHANGE = "timing_change"

SIGNAL_KINDS = (FLIP, ENTRY_BUY, ENTRY_SELL, EXIT)

# flip first, then entries, then exits, then timing-only moves.
_ORDER = {FLIP: 0, ENTRY_BUY: 1, ENTRY_SELL: 1, EXIT: 2, TIMING_CHANGE: 3}

_ACTIONABLE = ("BUY", "SELL")


@dataclass(frozen=True)
class DigestEvent:
    asset: str
    kind: str
    from_signal: str | None
    to_signal: str | None
    from_timing: str | None
    to_timing: str | None
    confidence: float | None
    date: str


def _confidence(signal, prob):
    """Calibrated confidence in the stated direction; None for WAIT."""
    if prob is None:
        return None
    if signal == "BUY":
        return prob
    if signal == "SELL":
        return 1.0 - prob
    return None


def _signal_of(row):
    return (row.get("signal") or "WAIT").upper()


def _classify(asset, frm, to):
    fs, ts = _signal_of(frm), _signal_of(to)
    if fs != ts:
        from_act, to_act = fs in _ACTIONABLE, ts in _ACTIONABLE
        if from_act and to_act:
            kind = FLIP
        elif to_act:
            kind = ENTRY_BUY if ts == "BUY" else ENTRY_SELL
        else:
            kind = EXIT
    elif frm.get("timing_label") != to.get("timing_label"):
        kind = TIMING_CHANGE
    else:
        return None
    return DigestEvent(asset=asset, kind=kind, from_signal=fs, to_signal=ts,
                       from_timing=frm.get("timing_label"),
                       to_timing=to.get("timing_label"),
                       confidence=_confidence(ts, to.get("prob")),
                       date=to.get("date"))


def _baseline(rows, current, since_date):
    """The row the current one is compared against, or None."""
    if since_date is None:
        for row in reversed(rows[:-1]):
            if row["date"] != current["date"]:
                return row
        return None
    for row in reversed(rows):
        if row["date"] <= since_date:
            # A baseline on the current date leaves nothing to report.
            return None if row["date"] == current["date"] else row
    return None


def build_digest(rows, since_date=None):
    """Events per asset between the latest snapshot and its baseline.

    since_date None: baseline is the previous DISTINCT date (default, what the
    push uses). Otherwise the baseline is the latest row dated at or before
    since_date, mirroring the client's "since my last visit" toggle.
    """
    by_asset = {}
    for row in rows:
        by_asset.setdefault(row["asset"], []).append(row)

    events = []
    for asset, group in by_asset.items():
        group = sorted(group, key=lambda r: r["date"])
        if len(group) < 2:
            continue
        current = group[-1]
        base = _baseline(group, current, since_date)
        if base is None:
            continue
        event = _classify(asset, base, current)
        if event is not None:
            events.append(event)

    # Asset breaks ties so the order is fully defined on both sides: exits all
    # carry a None confidence, and Dart's List.sort is not stable.
    events.sort(key=lambda e: (_ORDER[e.kind],
                               -(e.confidence if e.confidence is not None else -1.0),
                               e.asset))
    return events
