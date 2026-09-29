"""The day's top-down view, once a day, before any asset is judged.

Raw inputs only: FRED rates, dollar, oil, credit spread and VIX; the Fed and
Bank of Russia key rates; the MOEX index, BTC, gold and the S&P from market.db;
the release calendar. The brain may search and read further (web tools,
raw_source). Its view is kept only when every driver it names is backed by a
raw key from the block or by outside evidence that survives
core/analyst/evidence.py. Every asset judgment that day receives the regime,
the drivers and the line for its own market.
"""

import datetime
import json
import sqlite3

from core.analyst import evidence
from core.analyst.store import DB_PATH

SERIES = ("us_10y_yield", "us_2y_yield", "us_curve_10y_2y", "usd_broad_index", "brent",
          "high_yield_spread", "vix")
PRICES = ("SP500", "IMOEX", "BTC", "GOLD")
MARKETS = ("us", "ru", "eu", "crypto", "commodity", "fx")
MAX_AGE_DAYS = 4          # a Friday view still serves Monday and Tuesday morning

DDL = "CREATE TABLE IF NOT EXISTS analyst_macro (date TEXT PRIMARY KEY, brain TEXT, json TEXT)"

PROMPT = (
    "You are the head of a global macro desk. Below is today's RAW data: rates, "
    "the dollar, oil, credit, volatility, key policy rates, index and crypto "
    "moves, the release calendar. You may ask for more raw material with the "
    "tools. Form YOUR OWN view; ratings, forecasts and consensus are refused by "
    "code.\n\nRaw data (cite a key as \"<group>.<name>\", e.g. \"fred.us_10y_yield\"):\n"
    "%s\n\nReturn STRICT JSON: {\"regime\": \"one line\", \"drivers\": [{\"what\": "
    "\"one line\", \"evidence\": [\"fred.us_10y_yield\" or {\"source\": \"<url or "
    "tool>\", \"kind\": \"data|filing|statistic|price|news\", \"value\": \"...\", "
    "\"asof\": \"YYYY-MM-DD\"}]}], \"by_market\": {%s}}. 2 to 6 drivers.")


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def raw_block(today=None):
    from core.analyst import project_tools, tools
    from core.macro import policy_rate

    day = datetime.date.fromisoformat(str(today)[:10]) if today else datetime.date.today()
    fred = {}
    for name in SERIES:
        got = _safe(lambda n=name: tools._macro_series(None, day.isoformat(), n))
        if isinstance(got, dict) and got.get("last"):
            fred[name] = {k: got.get(k) for k in ("last", "change_1m", "change_3m")}
    prices = {}
    for asset in PRICES:
        bars = _safe(lambda a=asset: project_tools._bars(
            a, (day + datetime.timedelta(days=1)).isoformat(), 21)) or []
        if len(bars) >= 6:
            c = [b[4] for b in bars]
            prices[asset] = {"date": bars[-1][0], "close": c[-1],
                             "ret_5": round(100 * (c[-1] / c[-6] - 1), 2),
                             "ret_20": round(100 * (c[-1] / c[0] - 1), 2)}
    rates = {r: _safe(lambda r=r: policy_rate(r, today=None)) for r in ("us", "ru")}
    from core.analyst import dossier

    calendar = []
    for asset in ("AAPL", "SBER"):
        calendar += _safe(lambda a=asset: dossier._macro_for(a)) or []
    return {"fred": fred, "prices": prices,
            "key_rates": {k: v for k, v in rates.items() if v}, "calendar": calendar}


def _keys(raw):
    return {"%s.%s" % (g, k) for g, v in raw.items() if isinstance(v, dict) for k in v} | {
        g for g, v in raw.items() if v}


def validate(obj, raw):
    """The view with only its evidenced drivers, or None."""
    if not isinstance(obj, dict) or not isinstance(obj.get("regime"), str):
        return None
    known, drivers = _keys(raw), []
    for d in obj.get("drivers") or []:
        if not isinstance(d, dict) or not d.get("what"):
            continue
        items = d.get("evidence") if isinstance(d.get("evidence"), list) else []
        kept, _ = evidence.check([e for e in items if isinstance(e, dict)])
        named = [e for e in items if isinstance(e, str) and e in known]
        if named or kept:
            drivers.append({"what": str(d["what"])[:300], "evidence": named + kept})
    if not drivers:
        return None
    by_market = obj.get("by_market") if isinstance(obj.get("by_market"), dict) else {}
    return {"regime": obj["regime"][:300], "drivers": drivers,
            "by_market": {m: str(by_market[m])[:400] for m in MARKETS if by_market.get(m)}}


def _con(db_path=None):
    con = sqlite3.connect(db_path or DB_PATH)
    con.execute(DDL)
    return con


def save(day, brain, view, db_path=None):
    con = _con(db_path)
    try:
        with con:
            con.execute("INSERT OR REPLACE INTO analyst_macro VALUES (?,?,?)",
                        (day, brain, json.dumps(view, ensure_ascii=False)))
    finally:
        con.close()


def for_date(day, market, db_path=None):
    """The compact view a dossier carries, or None when there is no recent one."""
    con = _con(db_path)
    try:
        row = con.execute("SELECT date, json FROM analyst_macro WHERE date <= ? "
                          "ORDER BY date DESC LIMIT 1", (str(day)[:10],)).fetchone()
    finally:
        con.close()
    if not row:
        return None
    age = (datetime.date.fromisoformat(str(day)[:10]) - datetime.date.fromisoformat(row[0])).days
    if age > MAX_AGE_DAYS:
        return None
    view = json.loads(row[1])
    return {"date": row[0], "regime": view["regime"],
            "drivers": [d["what"] for d in view["drivers"]],
            "this_market": view.get("by_market", {}).get(market)}


def has(day, db_path=None):
    con = _con(db_path)
    try:
        return con.execute("SELECT 1 FROM analyst_macro WHERE date = ?", (day,)).fetchone() \
            is not None
    finally:
        con.close()


def run(call, today=None, brain="", db_path=None):
    """Build today's view with `call`; True when one was saved."""
    from core.analyst import agent

    day = str(today or datetime.date.today().isoformat())[:10]
    raw = raw_block(day)
    prompt = PROMPT % (json.dumps(raw, ensure_ascii=False, default=str, indent=1),
                       ", ".join('"%s": "one line"' % m for m in MARKETS))
    view = agent.ask_json(prompt, call, lambda o: validate(o, raw))
    if view is None:
        return False
    save(day, getattr(call, "last_label", None) or brain, view, db_path=db_path)
    return True
