"""Judgment cells to percentages.

A (direction, conviction, vol_regime) cell is worth whatever it has
historically been worth, shrunk toward a prior while its own evidence is thin.
The prior is the empirical payoff table from train_payoff.py, which is also the
baseline the agent has to beat: one artifact, two jobs, and no day on which the
card has nothing to show.

That prior is NOT the raw historical average for a direction - train_payoff.py
only measures bars the ensemble signalled on, so it is the average payoff
conditional on the ensemble's own choice to call BUY or SELL, not the asset's
or the class's unconditional payoff for that direction.

What it IS, before adjustment, is that choice plus the market underneath it.
_net() below removes the drift train_payoff measured over the same rows, so the
prior states what a side earned beyond simply being open. Without it every RU
long in 2026-08 inherited a -0.116 ATR prior of which -0.101 belonged to a
falling market rather than to the call, and the card flagged all of them.

The adjustment is not cosmetic and it does not only forgive. It also takes the
RU short's +0.087 down to -0.013, because a falling market handed the short
+0.101 for free and the calls gave back a little of it. Both sides of both
large classes come out slightly negative; crypto BUY, at +0.083 excess, is the
only cell measured so far that survives its own market.

Nothing here decides that a high conviction means a large move. If conviction 5
has historically lost money, this file produces a negative number for it. That
is not a safeguard bolted on; it is what fitting a cell to its outcomes means.
"""

import statistics

from core.analyst.payoff import shrink, to_pct

MIN_CELL_OWN = 100
# Below this a cell's own quantiles are not used; the prior's are. A ten-row
# cell's q10 is its second-smallest observation, which is not a tail estimate.

_SIDE_OF = {"up": "BUY", "down": "SELL", "flat": None}


def cell_key(judgment):
    return (judgment["direction"], int(judgment["conviction"]),
            judgment["vol_regime"])


def _net(cell):
    """(mean, q10, q90) with the window's own market drift removed.

    The quantiles shift by the same constant as the mean. They describe how far
    an outcome scatters, which the drift does not change; leaving them put would
    put the point estimate outside its own 80% band.

    A table built before train_payoff learned to measure drift carries no
    `drift` key, and is returned untouched: an existing payoff_stats.json still
    reads, it just keeps the old contaminated number until it is refit.
    """
    drift = cell.get("drift", 0.0)
    return cell["mean"] - drift, cell["q10"] - drift, cell["q90"] - drift


def _prior(payoff_table, asset, asset_class, side):
    """The empirical payoff for this side, asset shrunk toward class, net of
    the drift each of them was measured against."""
    cls = (payoff_table.get("class", {}).get(asset_class, {}).get(side)
           or {"n": 0, "mean": 0.0, "q10": -1.0, "q90": 1.0})
    cls_mean, cls_lo, cls_hi = _net(cls)
    own = payoff_table.get("asset", {}).get(asset, {}).get(side)
    if not own:
        return cls_mean, cls_lo, cls_hi
    own_mean, own_lo, own_hi = _net(own)
    mean = shrink(own["n"], own_mean, cls_mean)
    if own["n"] >= MIN_CELL_OWN:
        return mean, own_lo, own_hi
    return mean, cls_lo, cls_hi


def fit(scored_rows, payoff_table, asset_class_of):
    """The measured half of the calibration: {cell_key: {n, mean, q10, q90}}.

    Only rows the backfill has scored take part. A judgment without an outcome
    is an opinion, and opinions do not calibrate anything.

    These cells are RAW, unlike the prior they are shrunk toward: a judgment's
    realized_atr_units carries the same market drift, and there is nothing on
    the row to subtract it with. Harmless while the log is small - at n=1 a
    cell's own weight is 1/51 - but the moment any cell approaches MIN_CELL_OWN
    it starts reintroducing the drift the prior just had removed. Before that
    happens, store the same-window drift alongside realized_atr_units at
    backfill time and net it here.
    """
    buckets = {}
    for r in scored_rows:
        if r.get("realized_atr_units") is None:
            continue
        try:
            key = cell_key(r)
        except (KeyError, TypeError, ValueError):
            continue
        buckets.setdefault(key, []).append(r["realized_atr_units"])

    cells = {}
    for key, values in buckets.items():
        values.sort()
        n = len(values)
        cells[key] = {"n": n, "mean": statistics.fmean(values),
                      "q10": values[max(0, n // 10 - 1)],
                      "q90": values[min(n - 1, (9 * n) // 10)]}
    return cells


def forecast(judgment, cells, asset, asset_class, atr_today, close_today,
             payoff_table):
    """The percent forecast for one judgment, with its interval and provenance.

    `source` has three states, split at the same `MIN_CELL_OWN` that decides
    whether the interval comes from the cell or the prior — one threshold,
    one story:
      "prior"    the cell has no observations; the number is the prior alone.
      "blended"  the cell has some observations but fewer than MIN_CELL_OWN;
                 shrink() means the prior still dominates the point estimate
                 (at n=1, k=50, the cell's own weight is 1/51; even at n=99
                 it is under half), so calling this "measured" would overstate
                 it and calling it "prior" would understate it.
      "measured" the cell has reached MIN_CELL_OWN; its own quantiles are used
                 too. The card prints this, because a number backed by 500
                 observations and one backed by the class average are
                 different claims and the reader is entitled to tell them
                 apart.
    """
    side = _SIDE_OF.get(judgment["direction"])
    if side is None:                       # a flat call claims no move at all
        # The BUY side's quantiles ARE the raw return's distribution in ATR
        # units (BUY payoff = +ret / (atr/close)); SELL's are that same
        # return's negation, the short's payoff. A flat judgment claims the
        # raw return will be small, so BUY's quantiles are the right band
        # regardless of asset_class or SELL's shape - deliberate, not an
        # oversight, and always n=0 so "prior" is the correct label here too.
        _, lo_atr, hi_atr = _prior(payoff_table, asset, asset_class, "BUY")
        return {"pct": 0.0,
                "lo": to_pct(lo_atr, atr_today, close_today),
                "hi": to_pct(hi_atr, atr_today, close_today),
                "n": 0, "source": "prior"}

    prior_mean, prior_lo, prior_hi = _prior(payoff_table, asset, asset_class,
                                            side)
    cell = cells.get(cell_key(judgment)) or {"n": 0, "mean": prior_mean,
                                             "q10": prior_lo, "q90": prior_hi}
    mean_atr = shrink(cell["n"], cell["mean"], prior_mean)
    if cell["n"] >= MIN_CELL_OWN:
        lo_atr, hi_atr = cell["q10"], cell["q90"]
        source = "measured"
    else:
        lo_atr, hi_atr = prior_lo, prior_hi
        source = "blended" if cell["n"] else "prior"

    return {"pct": to_pct(mean_atr, atr_today, close_today),
            "lo": to_pct(min(lo_atr, mean_atr), atr_today, close_today),
            "hi": to_pct(max(hi_atr, mean_atr), atr_today, close_today),
            "n": cell["n"], "source": source}
