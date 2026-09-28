"""Read-only tools over the project's own market.db for the analyst.

Every tool honours `today` (only bars dated before it) and none of them reads a
model output: prediction_log, signals, guru_log and the champion registry stay
out, the same rule dossier.FORBIDDEN_KEYS applies. my_record reads only the
analyst's OWN past judgments, and only those whose outcome was known before
`today`.
"""

import math
import sqlite3
from itertools import pairwise

from core.analyst import store
from core.analyst.tools import Tool, register

MAX_DAYS = 250


def _table(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _known(asset):
    from config import FULL_ASSET_MAP
    return asset in FULL_ASSET_MAP


def _bars(asset, today, n, db_path=None):
    """Last n (date, open, high, low, close) before today, ascending."""
    sql = f'SELECT substr(Date,1,10), Open, High, Low, Close FROM "{_table(asset)}"'
    args = []
    if today:
        sql += " WHERE substr(Date,1,10) < ?"
        args.append(str(today)[:10])
    sql += " ORDER BY Date DESC LIMIT ?"
    args.append(int(n))
    con = sqlite3.connect(db_path or store.DB_PATH)
    try:
        rows = con.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    return [r for r in reversed(rows) if None not in r]


def _days(value, default):
    try:
        return max(2, min(MAX_DAYS, int(value)))
    except (TypeError, ValueError):
        return default


def _rets(closes):
    return [(b / a - 1.0) for a, b in pairwise(closes) if a]


def _pct(a, b):
    return None if not a else round((b / a - 1.0) * 100, 3)


def _corr(x, y):
    n = min(len(x), len(y))
    if n < 5:
        return None
    x, y = x[-n:], y[-n:]
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if not sx or not sy:
        return None
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy), 3)


def _price_history(asset, today=None, days=60):
    rows = _bars(asset, today, _days(days, 60) + 1)
    out = []
    for prev, cur in pairwise(rows):
        d, _o, h, lo, c = cur
        out.append({"date": d, "close": round(c, 6), "ret_pct": _pct(prev[4], c),
                    "range_pct": round((h - lo) / c * 100, 3) if c else None})
    return {"asset": asset, "bars": out}


def _compare(asset, today=None, assets="", days=20):
    n = _days(days, 20)
    names = [a.strip().upper() for a in str(assets or "").split(",") if a.strip()][:5]
    unknown = [a for a in names if not _known(a)]
    mine = [r[4] for r in _bars(asset, today, n + 1)]
    out = {"asset": asset, "days": n, "unknown": unknown,
           "ret_pct": {asset: _pct(mine[0], mine[-1]) if len(mine) > 1 else None},
           "corr_with_asset": {}}
    for a in names:
        if a in unknown:
            continue
        other = [r[4] for r in _bars(a, today, n + 1)]
        out["ret_pct"][a] = _pct(other[0], other[-1]) if len(other) > 1 else None
        out["corr_with_asset"][a] = _corr(_rets(mine), _rets(other))
    return out


def _sector_of(asset):
    from config import SECTOR_MAP
    return next((s for s, names in SECTOR_MAP.items() if asset in names), None)


def _sector_peers(asset, today=None):
    from config import SECTOR_MAP
    sector = _sector_of(asset)
    if sector is None:
        return {"asset": asset, "sector": None, "peers": []}
    peers = []
    for a in [p for p in SECTOR_MAP[sector] if p != asset][:10]:
        c = [r[4] for r in _bars(a, today, 21)]
        peers.append({"asset": a,
                      "ret5_pct": _pct(c[-6], c[-1]) if len(c) >= 6 else None,
                      "ret20_pct": _pct(c[0], c[-1]) if len(c) >= 21 else None})
    return {"asset": asset, "sector": sector, "peers": peers}


def _resolved_before(asset, date, horizon, today):
    """True when the horizon-th bar after `date` is dated before `today`."""
    if not today:
        return True
    dates = [r[0] for r in _bars(asset, today, 400)]
    later = {d for d in dates if d > date}     # a duplicated bar is still one day
    return len(later) >= int(horizon or 1)


def _my_record(asset, today=None, limit=10):
    try:
        limit = max(1, min(30, int(limit)))
    except (TypeError, ValueError):
        limit = 10
    con = sqlite3.connect(store.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        sql = ("SELECT date, horizon, direction, conviction, thesis, realized_ret "
               "FROM analyst_log WHERE asset=? AND realized_ret IS NOT NULL")
        args = [asset]
        if today:
            sql += " AND date < ?"
            args.append(str(today)[:10])
        rows = [dict(r) for r in con.execute(sql + " ORDER BY date DESC", args)]
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    out = []
    for r in rows:
        if not _resolved_before(asset, r["date"], r["horizon"], today):
            continue
        hit = None
        if r["direction"] in ("up", "down"):
            hit = (r["direction"] == "up") == (r["realized_ret"] > 0)
        out.append({"date": r["date"], "horizon": r["horizon"],
                    "direction": r["direction"], "conviction": r["conviction"],
                    "thesis": (r["thesis"] or "")[:200],
                    "realized_ret_pct": round(r["realized_ret"] * 100, 3), "hit": hit})
        if len(out) >= limit:
            break
    return {"asset": asset, "judgments": out}


register(Tool(name="price_history", args={"days": "bars to return, 2-250"},
              rewinds=True, run=_price_history,
              describe="daily closes, returns and ranges of this asset from the project database"))
register(Tool(name="compare", args={"assets": "up to 5 other assets, comma-separated",
                                    "days": "window in bars"},
              rewinds=True, run=_compare,
              describe="returns over a window and daily-return correlation of this asset with others"))
register(Tool(name="sector_peers", args={}, rewinds=True, run=_sector_peers,
              describe="5 and 20 bar returns of the other assets in this asset's sector"))
register(Tool(name="my_record", args={"limit": "how many past judgments, 1-30"},
              rewinds=True, run=_my_record,
              describe="your own earlier judgments on this asset with what actually happened"))
