"""What an asset actually did over a period, the way a research note states it.

Everything here reads market.db and nothing else: 19 years of daily bars are
already local, and so is every benchmark the classes map onto, so a performance
table costs one query rather than a network round trip.

Three things this deliberately does NOT do, because getting them wrong is how a
performance table turns into a sales document:

  Annualising a short window. A month that returned 4% did not return 60% a
  year, and quoting it that way is the oldest number in the business.
  `annualised` is None under a year and the caller prints a dash.

  Calling a price return a total return. market.db stores prices, not
  dividends, so a 13% yielder shows only what its price did. The dict says so
  in `includes_dividends`, which is False, and the summary line prints the
  current yield beside it rather than silently folding it in.

  Comparing against a benchmark it did not trade alongside. The comparison is
  computed on the DATES BOTH HAVE, not on each series' own window, or a
  Moscow name measured over a week the index was shut reads as outperformance.

And one thing it does that a textbook would not: the annualisation factor comes
from the asset's OWN bars rather than from 252. Moscow trades weekends now, so
SBER has about 334 bars a year against IMOEX's 255, and crypto has 365. A
single constant understates a weekend-trading instrument's volatility by
sqrt(252/334), which is 13% of it, and crypto's by 20%.
"""

import datetime
import math
import sqlite3

BASE_WINDOWS = ("1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX")

_DAYS = {"1M": 31, "3M": 92, "6M": 183, "1Y": 365, "2Y": 730, "3Y": 1096,
         "5Y": 1826, "10Y": 3653}
TRADING_DAYS = 252
MIN_BARS_PER_YEAR = 12    # below this the series is not daily and the scaling
                          # would be guesswork; fall back to the constant.


def parse_window(window, today=None):
    """The first date a window admits, or None for MAX.

    Accepts the named windows, "YTD", and a bare day count ("90"), so the
    caller can ask for a period nobody thought to name.
    """
    # The dossier rewinds with an ISO STRING, the CLI with nothing at all.
    # Coercing here rather than at each caller: a TypeError inside a window
    # calculation surfaces as a blank block, which reads like missing data.
    if isinstance(today, str):
        today = datetime.date.fromisoformat(today[:10])
    today = today or datetime.date.today()
    key = str(window).strip().upper()
    if key in ("MAX", "ALL", ""):
        return None
    if key == "YTD":
        return datetime.date(today.year, 1, 1)
    if key in _DAYS:
        return today - datetime.timedelta(days=_DAYS[key])
    try:
        return today - datetime.timedelta(days=int(key))
    except ValueError as exc:
        raise ValueError("unknown window %r" % window) from exc


def _table(asset):
    return "".join(c if c.isalnum() else "_" for c in str(asset)).lower()


def closes(asset, db_path, since=None, until=None):
    """[(date, close)] ascending, or [] when the asset has no table.

    `until` is not decoration. core/analyst/dossier.py builds this block for a
    REWOUND date when backfilling old judgments, and a window bounded only at
    the start would hand a judgment dated in May the year that followed it.
    That is the exact trap the rest of the rewind machinery exists to close.
    """
    sql = 'SELECT Date, close FROM "%s" WHERE close IS NOT NULL' % _table(asset)
    args = []
    if since:
        sql += " AND Date >= ?"
        args.append(str(since))
    if until:
        sql += " AND Date <= ?"
        args.append(str(until))
    try:
        con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
        try:
            rows = con.execute(sql + " ORDER BY Date", args).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return []
    return [(str(d)[:10], float(c)) for d, c in rows if c]


def _returns(prices):
    return [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))
            if prices[i - 1]]


def _max_drawdown(prices):
    peak, worst = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        worst = min(worst, p / peak - 1.0)
    return worst


def _stdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def bars_per_year(rows):
    """How often this instrument actually prints a bar, from the data itself.

    A US name gives about 252, a Moscow one about 334 since the exchange added
    weekend sessions, crypto 365. Used to annualise, so the number describes
    the instrument rather than an assumption about it.
    """
    if len(rows) < 3:
        return TRADING_DAYS
    span = (datetime.date.fromisoformat(rows[-1][0])
            - datetime.date.fromisoformat(rows[0][0])).days
    if span < 30:
        return TRADING_DAYS
    rate = len(rows) * 365.0 / span
    return rate if rate >= MIN_BARS_PER_YEAR else TRADING_DAYS


def _beta(asset_rets, bench_rets):
    """Covariance over variance on the paired days, or None when flat."""
    n = min(len(asset_rets), len(bench_rets))
    if n < 20:
        return None
    a, b = asset_rets[-n:], bench_rets[-n:]
    mean_a, mean_b = sum(a) / n, sum(b) / n
    var_b = sum((x - mean_b) ** 2 for x in b)
    if var_b <= 0:
        return None
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    return cov / var_b


def _paired(rows_a, rows_b):
    """The two series on the dates they share, in order."""
    by_date = dict(rows_b)
    both = [(d, c, by_date[d]) for d, c in rows_a if d in by_date]
    return [c for _d, c, _b in both], [b for _d, _c, b in both]


def summary(asset, window="1Y", db_path=None, benchmark=None, today=None):
    """One period's performance, as a research note states it.

    `benchmark` is an asset code; None asks for none. The caller decides, so
    that the class map lives in one place (core/analyst/dossier.py) rather than
    being duplicated here.
    """
    db_path = db_path or _default_db()
    since = parse_window(window, today=today)
    rows = closes(asset, db_path, since=since, until=today)
    if len(rows) < 3:
        return {"asset": asset, "window": str(window).upper(), "bars": len(rows),
                "error": "not enough bars in this window"}

    prices = [c for _d, c in rows]
    rets = _returns(prices)
    total = prices[-1] / prices[0] - 1.0
    span_days = (datetime.date.fromisoformat(rows[-1][0])
                 - datetime.date.fromisoformat(rows[0][0])).days
    per_year = bars_per_year(rows)
    vol = _stdev(rets) * math.sqrt(per_year) if rets else 0.0

    out = {
        "asset": asset,
        "window": str(window).upper(),
        "start": rows[0][0], "end": rows[-1][0], "bars": len(rows),
        "start_price": prices[0], "end_price": prices[-1],
        "total_return": total,
        # Under a year an annual rate is an extrapolation, not a measurement.
        "annualised": ((1.0 + total) ** (365.0 / span_days) - 1.0
                       if span_days >= 360 and total > -1 else None),
        "volatility": vol,
        "bars_per_year": round(per_year, 1),
        "sharpe": ((sum(rets) / len(rets)) * per_year / vol
                   if rets and vol > 0 else None),
        "max_drawdown": _max_drawdown(prices),
        "best_day": max(rets) if rets else None,
        "worst_day": min(rets) if rets else None,
        "positive_days": (sum(1 for r in rets if r > 0) / len(rets)
                          if rets else None),
        "high": max(prices), "low": min(prices),
        "off_high": prices[-1] / max(prices) - 1.0,
        # market.db carries prices. A dividend is not in them, and pretending
        # otherwise understates every high yielder in the Moscow half.
        "includes_dividends": False,
        "benchmark": None,
    }

    if benchmark and benchmark != asset:
        b_rows = closes(benchmark, db_path, since=since, until=today)
        a_paired, b_paired = _paired(rows, b_rows)
        if len(a_paired) >= 20:
            b_total = b_paired[-1] / b_paired[0] - 1.0
            a_total = a_paired[-1] / a_paired[0] - 1.0
            out["benchmark"] = {
                "asset": benchmark,
                "shared_bars": len(a_paired),
                "return": b_total,
                # On the shared dates for both sides, so the difference is a
                # comparison rather than two different periods subtracted.
                "excess": a_total - b_total,
                "beta": _beta(_returns(a_paired), _returns(b_paired)),
            }
    return out


def _default_db():
    import os
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "market.db")


def _pct(value, width=8):
    return " " * width if value is None else ("%+*.2f%%" % (width - 1, 100 * value))


def table(asset, windows=BASE_WINDOWS, db_path=None, benchmark=None, today=None):
    """The performance block as printable lines, one row per window."""
    head = ("%-6s %10s %9s %9s %8s %8s %7s   %s"
            % ("period", "return", "ann.", "vol", "maxDD", "vs bmk", "beta",
               "from"))
    lines = [head, "-" * len(head)]
    for window in windows:
        s = summary(asset, window, db_path=db_path, benchmark=benchmark,
                    today=today)
        if s.get("error"):
            lines.append("%-6s %s" % (s["window"], s["error"]))
            continue
        bmk = s.get("benchmark") or {}
        lines.append(
            "%-6s %s %s %s %s %s %7s   %s"
            % (s["window"], _pct(s["total_return"], 10), _pct(s["annualised"], 9),
               _pct(s["volatility"], 9), _pct(s["max_drawdown"], 8),
               _pct(bmk.get("excess"), 8),
               "  -  " if bmk.get("beta") is None else "%5.2f" % bmk["beta"],
               s["start"]))
    lines.append("")
    lines.append("Price return only: market.db stores prices, so dividends are "
                 "not in any of these numbers. 'ann.' is blank under a year "
                 "because annualising a short window invents a rate nobody "
                 "earned. 'vs bmk' is measured on the dates both series have, "
                 "and volatility is annualised on this instrument's own bar "
                 "count rather than on 252.")
    return lines
