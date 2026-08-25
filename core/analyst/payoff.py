"""Payoff arithmetic in ATR units.

A percentage from two years ago is not comparable to today's: both the price
and the volatility of the asset have moved. Everything here converts to and
from ATR units so that a payoff measured in 2024 can be stated as a percentage
of today's close.

Pure. No database, no models, no file reads, for the same reason core/levels.py
is pure: the numbers have to be testable without the project's state around
them.
"""

K_SHRINK = 50.0
# Fifty observations before a cell's own mean outweighs its prior. Chosen
# against the data rather than by taste: prediction_log holds 5089 verified
# rows over 208 assets, about 24 per asset, so a k much below this would let a
# twelve-observation asset speak for itself, and a k much above it would leave
# every asset permanently indistinguishable from its class.


def ret_atr(ret, atr, close, side=1):
    """A return expressed in ATR units, signed by the side of the position.

    `atr` and `close` are the values AT THE TIME of the return, not today's:
    the point of the unit is to strip out the volatility regime the move
    happened in. Returns None when the scale is missing or degenerate, which is
    an asset without enough history to have an ATR yet.
    """
    if atr is None or close is None:
        return None
    if atr <= 0 or close <= 0:
        return None
    return side * ret / (atr / close)


def to_pct(value_atr, atr_today, close_today):
    """An ATR-unit quantity restated as a fraction of today's close."""
    if value_atr is None or atr_today is None or close_today is None:
        return None
    if close_today <= 0:
        return None
    return value_atr * (atr_today / close_today)


def shrink(n, mean, prior, k=K_SHRINK):
    """A cell's own mean pulled toward its prior in proportion to its evidence.

    n=0 gives the prior unchanged, n=k gives the midpoint, large n gives the
    mean. This is the same shape core/guru.py uses to weight its council from a
    thin forward log, and it is what keeps an asset with nine observations from
    printing a confident number.
    """
    if n <= 0:
        return prior
    return (n * mean + k * prior) / (n + k)
