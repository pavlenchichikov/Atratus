"""May the overlay override the ensemble's DIRECTION, and if so where.

The last and widest authority in the program, and the one the 2026-07-18 spec
originally refused. It is built here only because the live reading forced the
question: over 44 trading days the emitted signals were 48.3 percent accurate,
and accuracy FELL as confidence rose, to 46.4 percent in the most confident
bucket, where the mean return in the signal's direction is negative.

So the rule is deliberately the smallest thing that can express what was
measured, and nothing more: above a confidence threshold, either stand aside or
take the other side. Two parameters, both readable, and `follow` is the
incumbent, which means the identity is in the search space by construction.

Pure: rows in, rows out. Fitting and gating live in train_direction.py.
"""
MODES = ("follow", "aside", "invert")
DEFAULT_PARAMS = {"mode": "follow", "thr": 0.20}
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

_FLIP = {"BUY": "SELL", "SELL": "BUY"}


def apply_direction(rows, mode="follow", thr=0.20):
    """The same rows with the signal rewritten where the rule applies.

    A row whose probability is missing is left alone rather than guessed at: an
    override needs the confidence it is conditioned on, and inventing one would
    put the rule in charge of rows it cannot actually read.
    """
    if mode == "follow":
        return list(rows)
    out = []
    for r in rows:
        p = r.get("probability")
        sig = (r.get("signal") or "").upper()
        r = dict(r)
        if p is not None and sig in _FLIP and abs(float(p) - 0.5) >= thr:
            r["signal"] = "WAIT" if mode == "aside" else _FLIP[sig]
        out.append(r)
    return out
