"""Same-day linkage between a price move and that day's news.

The claim is co-occurrence, not cause. A move and a headline landing on the
same day is not evidence that one produced the other, so nothing here reports
a reason. These functions measure only whether the move was unusual for this
asset and whether sentiment agreed with its direction, and disagreement is
reported rather than hidden: it is what tells the reader no explanation was
found.

Pure and offline. Everything operates on bars and news items already fetched.
"""

import statistics
from email.utils import parsedate_to_datetime

NOTABLE_K = 1.5      # the move must exceed this many daily sigmas
SIGMA_WINDOW = 60    # returns used for the volatility estimate
MIN_RETURNS = 20     # below this a sigma is not worth trusting
UNCLEAR_BAND = 0.1   # |sentiment| under this reads as no opinion


def _closes(bars):
    return [b["close"] for b in bars if b.get("close") is not None]


def daily_move(bars):
    """Last close over the previous close, minus one. None when unmeasurable."""
    closes = _closes(bars)
    if len(closes) < 2 or closes[-2] <= 0:
        return None
    return closes[-1] / closes[-2] - 1.0


def move_sigma(bars, window=SIGMA_WINDOW):
    """Standard deviation of daily returns over the trailing window.

    None with too few returns: a sigma taken from three bars would call
    everything notable.
    """
    closes = _closes(bars)[-(window + 1):]
    rets = [cur / prev - 1.0
            for prev, cur in zip(closes, closes[1:]) if prev > 0]
    if len(rets) < MIN_RETURNS:
        return None
    return statistics.pstdev(rets)


def is_notable(move, sigma, k=NOTABLE_K):
    """Whether the move is large for THIS asset.

    Volatility-relative on purpose: 3 percent is a quiet day for BTC and an
    event for DAX.
    """
    if move is None or sigma is None or sigma <= 0:
        return False
    return abs(move) > k * sigma


def _published_date(item):
    raw = (item.get("published") or "").strip()
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        return None


def same_day(items, date):
    """Items published on `date`.

    An item whose timestamp will not parse is excluded, never assumed to be
    today: guessing would park unrelated news beside a move.
    """
    return [i for i in items if _published_date(i) == date]


def mean_sentiment(items):
    """Mean weighted_score across the items, or None when there are none."""
    scores = [i["weighted_score"] for i in items
              if i.get("weighted_score") is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def consistency(move, sentiment):
    """Whether sentiment agreed with the move.

    Returns "consistent", "conflicting", "unclear" or "no_news".

    "unclear" is a real answer rather than a fallback: the day's news was read
    and carried no lean, which is what the reader needs before inventing an
    explanation. "no_news" is the different case where there was nothing to
    read at all. Collapsing the two would let a data gap read as a measurement.
    """
    if move is None:
        return "unclear"
    if sentiment is None:
        return "no_news"
    if move == 0 or abs(sentiment) < UNCLEAR_BAND:
        return "unclear"
    return "consistent" if (move > 0) == (sentiment > 0) else "conflicting"


def context_row(asset, bars, items):
    """The news_context row for one asset, or None without bars.

    The day compared against is the date of the last bar, which is the same day
    daily_move measured. It is deliberately not today: a stale asset would
    otherwise be matched against news from a day it never moved on.

    `bars` must be in ascending date order, which is how
    push_signals.fetch_history_rows returns them.
    """
    if not bars:
        return None
    date = bars[-1]["date"]
    move = daily_move(bars)
    todays = same_day(items, date)
    return {
        "asset": asset,
        "date": date,
        "move_pct": move,
        "notable": is_notable(move, move_sigma(bars)),
        "consistency": consistency(move, mean_sentiment(todays)),
    }
