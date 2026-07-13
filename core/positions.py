"""Turn a stream of per-bar signals into positions.

The models emit BUY / SELL / WAIT every bar; a "position" is what you actually
hold: you go LONG on a BUY and stay long while the signal keeps saying BUY, then
exit when it turns WAIT or SELL (symmetrically for SHORT). Collapsing the raw
per-bar signals into positions makes the "when do I enter / exit / just hold"
question answerable, and drives the position card, the entry/exit chart markers,
the state ribbon and the trade log on the asset page.

Pure and serve-side: no models, no I/O. Position returns are chained from each
bar's realized forward return (`ret`), so no price series is required; a bar with
`ret is None` (not yet reconciled) simply does not contribute.
"""


def _side(signal) -> int:
    """+1 for a long (BUY), -1 for a short (SELL), 0 for flat (WAIT/other)."""
    s = (signal or "").upper()
    if s == "BUY":
        return 1
    if s == "SELL":
        return -1
    return 0


def _segment_return(side: int, seg_bars: list):
    """Chained return of a held position over its bars, or None if no bar in the
    segment has a realized forward return yet."""
    if side == 0:
        return None
    factor = 1.0
    have = False
    for b in seg_bars:
        r = b.get("ret")
        if r is not None:
            factor *= 1.0 + side * r
            have = True
    return factor - 1.0 if have else None


def build_positions(bars: list) -> dict:
    """Collapse chronological per-bar signals into positions.

    `bars`: oldest-first list of dicts, each with `date`, `signal` (BUY/SELL/WAIT)
    and optionally `ret` (this bar's realized forward return).

    Returns a dict with:
      - `segments`: every run of constant side, {side, start_date, end_date, bars,
        ret, open}. `open` is True for a still-held final position.
      - `markers`: enter/exit points for the chart, {date, type, side}.
      - `trades`: closed positions only (the trade log), newest first.
      - `current`: the state card - {"state": "FLAT"} or
        {"state": "LONG"|"SHORT", since, bars, ret, fresh}.
    """
    n = len(bars)
    segments = []
    i = 0
    while i < n:
        side = _side(bars[i]["signal"])
        j = i
        while j + 1 < n and _side(bars[j + 1]["signal"]) == side:
            j += 1
        seg_bars = bars[i:j + 1]
        segments.append({
            "side": side,
            "start_date": bars[i]["date"],
            "end_date": bars[j]["date"],
            "bars": len(seg_bars),
            "ret": _segment_return(side, seg_bars),
            "open": j == n - 1 and side != 0,
        })
        i = j + 1

    markers = []
    for seg in segments:
        if seg["side"] == 0:
            continue
        markers.append({"date": seg["start_date"], "type": "enter", "side": seg["side"]})
        if not seg["open"]:
            markers.append({"date": seg["end_date"], "type": "exit", "side": seg["side"]})

    trades = [
        {"side": s["side"], "start_date": s["start_date"], "end_date": s["end_date"],
         "bars": s["bars"], "ret": s["ret"]}
        for s in reversed(segments) if s["side"] != 0 and not s["open"]
    ]

    current = None
    if segments:
        last = segments[-1]
        if last["side"] == 0:
            current = {"state": "FLAT"}
        else:
            current = {
                "state": "LONG" if last["side"] == 1 else "SHORT",
                "since": last["start_date"],
                "bars": last["bars"],
                "ret": last["ret"],
                "fresh": last["bars"] == 1,
            }

    return {"segments": segments, "markers": markers, "trades": trades, "current": current}
