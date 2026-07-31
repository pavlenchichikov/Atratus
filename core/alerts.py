"""Encode digest events for delivery, and shape them for the alert log.

Pure: no database, no network, no Firebase. push_signals.py owns the I/O.

The device decides what to display, because it holds the personal positions
journal and the server never sees it. So what goes over the wire is the
CANDIDATE set rather than a composed notification.
"""

from core import digest

VERSION = "v1"

# FCM allows 4096 bytes of data across all keys; the rest of the message is a
# short "screen" key, so this leaves ample room. The worst day on record
# carried 107 signal events, about 1.9 KB.
PAYLOAD_CAP = 3500

_KIND_CODE = {
    digest.FLIP: "F",
    digest.ENTRY_BUY: "E",
    digest.ENTRY_SELL: "E",
    digest.EXIT: "X",
}
_SIGNAL_CODE = {"BUY": "B", "SELL": "S", "WAIT": "W"}
_SEPARATORS = (":", ";", "|")


def _signal_code(signal):
    """B, S or W; empty for anything else so a field never carries a surprise."""
    return _SIGNAL_CODE.get((signal or "").upper(), "")


def _conf_code(confidence):
    """Confidence as an integer percent, or empty when there is none.

    An empty field, not a zero: every exit has no confidence, and a zero would
    read as "0 percent confident", which is a different claim. Formatted with
    %.0f to match what build_push_text has always printed for the same number.
    """
    if confidence is None:
        return ""
    return "%.0f" % (confidence * 100)


def encode_events(events):
    """(payload, n_signal, n_timing) for a digest event list.

    payload is "v1|<n_signal>|<n_timing>|<ev>;<ev>...", each event being
    ASSET:KIND:FROM:TO:CONF. Timing events are counted and never listed: they
    have never triggered a push, and the phone only needs their number.

    An asset whose name contains a field separator is left OUT of the list but
    still counted. No live ticker contains one (all 208 are alphanumeric plus a
    single underscore), but a ticker added later must not be able to corrupt
    the payload silently.
    """
    signal_events = [e for e in events if e.kind in digest.SIGNAL_KINDS]
    n_timing = len(events) - len(signal_events)
    parts = []
    for e in signal_events:
        if any(sep in e.asset for sep in _SEPARATORS):
            continue
        parts.append(":".join((e.asset, _KIND_CODE[e.kind],
                               _signal_code(e.from_signal),
                               _signal_code(e.to_signal),
                               _conf_code(e.confidence))))
    payload = "%s|%d|%d|%s" % (VERSION, len(signal_events), n_timing,
                               ";".join(parts))
    return payload, len(signal_events), n_timing


def alert_rows(events, push_hash, pushed, sent_at):
    """One tuple per candidate event, in alert_log column order.

    Every event is recorded, timing ones included, with kind naming which is
    which: later research can filter, whereas a set already filtered here could
    not be reconstructed.
    """
    flag = 1 if pushed else 0
    return [(sent_at, e.date, e.asset, e.kind, e.from_signal, e.to_signal,
             e.confidence, push_hash, flag) for e in events]
