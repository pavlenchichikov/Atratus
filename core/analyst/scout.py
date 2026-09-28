"""The analyst's scout: which assets deserve a look today, and why.

It sees three things and nothing else: the biggest recent moves in market.db,
the analyst's own recent misses, and panel assets it has not judged lately. No
field of the ensemble's. If the LLM gives nothing usable, a fixed order takes
over (misses, then movers, then stale) and says so.
"""

import datetime
import json
import sqlite3

from core.analyst import store
from core.analyst.agent import _first_json_object, plain

PROMPT = (
    "You pick which assets an analyst should study today. Below: the biggest "
    "recent moves (ret_1 and ret_5 in percent), the analyst's recent misses, and "
    "assets it has not looked at lately.\n\n%s\n\nChoose at most %d. Return "
    "STRICT JSON: {\"assets\": [{\"asset\": \"NAME\", \"reason\": \"one sentence\"}]}")

AUTO_DDL = """
CREATE TABLE IF NOT EXISTS analyst_auto_log (
    date TEXT, asset TEXT, reason TEXT, picked_by TEXT, brain TEXT,
    PRIMARY KEY (date, asset)
)
"""


def _known(asset):
    from config import FULL_ASSET_MAP
    return asset in FULL_ASSET_MAP


def _is_stale(last, today, days):
    """No judgment yet, or the last one is `days` or more calendar days old."""
    if last is None:
        return True
    end = datetime.date.fromisoformat((today or datetime.date.today().isoformat())[:10])
    return (end - datetime.date.fromisoformat(last[:10])).days >= days


def summary(today=None, panel=(), db_path=None, n_movers=12, n_misses=6, stale_days=7):
    from config import FULL_ASSET_MAP
    from core.analyst.project_tools import _bars

    movers = []
    for a in FULL_ASSET_MAP:
        c = [r[4] for r in _bars(a, today, 6, db_path=db_path)]
        if len(c) == 6 and c[0] and c[-2]:
            movers.append({"asset": a, "ret_1": round((c[-1] / c[-2] - 1) * 100, 2),
                           "ret_5": round((c[-1] / c[0] - 1) * 100, 2)})
    movers.sort(key=lambda m: -abs(m["ret_5"]))

    misses = []
    for r in reversed(store.scored_rows(db_path=db_path)):
        if today and r["date"] >= today:
            continue
        d = r.get("direction")
        if d in ("up", "down") and (d == "up") != (r["realized_ret"] > 0):
            misses.append({"asset": r["asset"], "date": r["date"], "direction": d,
                           "ret": round(100 * r["realized_ret"], 2)})
        if len(misses) >= n_misses:
            break

    stale = []
    con = sqlite3.connect(db_path or store.DB_PATH)
    try:
        con.execute(store.DDL)
        for a in panel:
            sql, args = "SELECT MAX(date) FROM analyst_log WHERE asset=?", [a]
            if today:
                sql += " AND date < ?"
                args.append(today)
            last = con.execute(sql, args).fetchone()[0]
            if _is_stale(last, today, stale_days):
                stale.append(a)
    finally:
        con.close()
    return {"movers": movers[:n_movers], "misses": misses, "stale": stale}


def _fallback(s, max_assets):
    order = ([(m["asset"], "fallback: recent miss") for m in s["misses"]]
             + [(m["asset"], "fallback: big move") for m in s["movers"]]
             + [(a, "fallback: not judged lately") for a in s["stale"]])
    out, seen = [], set()
    for a, why in order:
        if a not in seen and _known(a):
            seen.add(a)
            out.append({"asset": a, "reason": why, "by": "fallback"})
        if len(out) >= max_assets:
            break
    return out


def pick(call, s, max_assets=5):
    """Up to max_assets {"asset", "reason", "by"}; the fallback when the LLM
    gives nothing usable. TerminalCallError / ProviderUnavailable propagate."""
    from core.llm_proposer import ProviderUnavailable, TerminalCallError

    try:
        answer = call(PROMPT % (json.dumps(s, indent=1, ensure_ascii=True), max_assets))
    except (TerminalCallError, ProviderUnavailable):
        raise
    except Exception:
        answer = ""
    data = _first_json_object(answer or "")
    items = data.get("assets") if isinstance(data, dict) else None
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        a = str(item.get("asset") or "").strip().upper()
        if a in seen or not _known(a):
            continue
        seen.add(a)
        out.append({"asset": a, "reason": plain(str(item.get("reason") or ""))[:200],
                    "by": "scout"})
        if len(out) >= max_assets:
            break
    return out or _fallback(s, max_assets)


def log_picks(date, picks, brain, db_path=None):
    con = sqlite3.connect(db_path or store.DB_PATH)
    try:
        with con:
            con.execute(AUTO_DDL)
            for p in picks:
                con.execute("INSERT OR REPLACE INTO analyst_auto_log VALUES (?,?,?,?,?)",
                            (date, p["asset"], p["reason"], p["by"], brain))
    finally:
        con.close()
