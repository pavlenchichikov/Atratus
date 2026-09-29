"""The source hunter: any brain proposes raw data endpoints, code verifies them.

A brain (claude-code, cloud or local Ollama) is asked for free, machine-
readable, RAW endpoints for one market: statistics, exchange data, filings,
prices. Nothing it says is trusted. Each candidate is fetched with a real asset
substituted and kept only if the reply parses as rows (JSON, CSV or MOEX ISS),
has a date field and a numeric field, is fresh for its frequency, and is not
on an opinion site. Kept ones go to `analyst_sources`; the `raw_source` tool
hands them to every brain. Only GET, nothing fetched is ever executed.
"""

import csv
import datetime
import io
import json
import re
import sqlite3

from core.analyst import evidence, tools
from core.analyst.store import DB_PATH

MARKETS = ("us", "ru", "eu", "crypto", "commodity", "fx", "macro")
EXAMPLE = {"us": "AAPL", "ru": "SBER", "eu": "SAP", "crypto": "BTC",
           "commodity": "GOLD", "fx": "EURUSD", "macro": None}
FRESH_DAYS = {"daily": 10, "weekly": 17, "monthly": 35, "quarterly": 100}
PLACEHOLDERS = ("ticker", "secid", "pair", "date")
MIN_ROWS = 3
_DATE = re.compile(r"^\d{4}-?\d{2}-?\d{2}")

DDL = """
CREATE TABLE IF NOT EXISTS analyst_sources (
    name TEXT PRIMARY KEY, market TEXT, url_template TEXT, format TEXT,
    symbol_kind TEXT, frequency TEXT, fields TEXT, status TEXT, found_by TEXT,
    verified_at TEXT, sample TEXT
)
"""

PROMPT = (
    "You are building a data desk for a professional analyst covering the %s "
    "market. Find FREE, machine-readable endpoints (JSON or CSV, no login, no "
    "paid key) that return RAW material: official statistics, exchange data, "
    "volumes, positioning, rates, filings, prices. Never a rating, a price "
    "target, a forecast, a consensus or anybody's opinion. Use the web tools to "
    "find and check them.\n\n"
    "The URL may contain only these placeholders: {ticker} (Yahoo symbol, e.g. "
    "AAPL), {secid} (MOEX code, e.g. SBER), {pair} (Binance pair, e.g. BTCUSDT), "
    "{date} (YYYY-MM-DD). Example asset for this market: %s.\n\n"
    "Return STRICT JSON: {\"sources\": [{\"name\": \"short_snake_case\", "
    "\"url_template\": \"https://...\", \"format\": \"json|csv\", \"fields\": "
    "\"what the columns are\", \"frequency\": \"daily|weekly|monthly|quarterly\", "
    "\"symbol_kind\": \"ticker|secid|pair|none\", \"why\": \"what it measures\"}]}. "
    "At most 8, the most useful first.")


def market_of(asset):
    from config import FULL_ASSET_MAP, radar_category

    if str(FULL_ASSET_MAP.get(asset, "")).endswith("=X"):
        return "fx"
    return radar_category(asset)


def symbol(asset, kind):
    from config import FULL_ASSET_MAP

    sym = str(FULL_ASSET_MAP.get(asset, asset))
    if kind == "pair":
        return sym.split("-")[0] + "USDT"
    if kind == "secid":
        return asset
    return sym


def fill(template, asset, kind, today=None):
    today = (today or datetime.date.today()).isoformat()
    out = template.replace("{date}", today)
    if asset is not None:
        out = out.replace("{%s}" % kind, symbol(asset, kind)) if kind in PLACEHOLDERS else out
    return out


def parse_rows(text):
    """A list of row dicts from JSON, MOEX ISS JSON or CSV; [] for anything else."""
    text = (text or "").strip()
    if text[:1] in ("[", "{"):
        try:
            data = json.loads(text)
        except ValueError:
            return []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict) and isinstance(v.get("columns"), list) \
                        and isinstance(v.get("data"), list):
                    return [dict(zip(v["columns"], row, strict=False)) for row in v["data"]]
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if not isinstance(data, list):
            return []
        # [metadata, [rows]] (World Bank): the rows are the longest inner list.
        inner = [v for v in data if isinstance(v, list)]
        if inner:
            data = max(inner, key=len)
        return [r for r in data if isinstance(r, dict)]
    if text[:1] == "<":
        return []
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows if rows and len(rows[0]) >= 2 else []


def _share(rows, test):
    return sum(1 for r in rows if test(r)) >= 0.8 * len(rows)


def _is_num(v):
    try:
        float(str(v).replace(",", ""))
        return True
    except ValueError:
        return False


def _as_date(v):
    s = re.sub(r"\D", "", str(v))[:8]
    try:
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _date_key(rows):
    """The field that holds a date in most rows, or None."""
    return next((k for k in rows[0] if _share(rows, lambda r, k=k: bool(
        _DATE.match(str(r.get(k) or ""))))), None) if rows else None


def verify(cand, fetch, asset=None, today=None):
    """(ok, reason, sample): the candidate checked against a real reply."""
    today = today or datetime.date.today()
    template = str(cand.get("url_template") or "")
    if not template.startswith(("http://", "https://")):
        return False, "not an http(s) url", None
    if evidence.blocked(template):
        return False, "opinion site", None
    url = fill(template, asset, cand.get("symbol_kind"), today)
    try:
        rows = parse_rows(fetch(url))
    except Exception as exc:
        return False, "fetch failed: %s" % type(exc).__name__, None
    if not rows:
        return False, "no rows (not JSON/CSV data)", None
    if len(rows) < MIN_ROWS:
        return False, "too few rows (%d): a record, not a series" % len(rows), None
    keys = list(rows[0])
    date_key = _date_key(rows)
    if date_key is None:
        return False, "no date field", None
    if not any(_share(rows, lambda r, k=k: _is_num(r.get(k))) for k in keys if k != date_key):
        return False, "no numeric field", None
    newest = max((d for d in (_as_date(r.get(date_key)) for r in rows) if d), default=None)
    limit = FRESH_DAYS.get(str(cand.get("frequency")), FRESH_DAYS["daily"])
    if newest is None or (today - newest).days > limit:
        return False, "stale: newest %s, limit %d days" % (newest, limit), None
    return True, "ok", json.dumps(rows[-3:], ensure_ascii=False, default=str)[:800]


def _con(db_path=None):
    con = sqlite3.connect(db_path or DB_PATH)
    con.execute(DDL)
    return con


def save(cand, found_by, sample, db_path=None):
    con = _con(db_path)
    try:
        with con:
            con.execute("INSERT OR REPLACE INTO analyst_sources VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                cand["name"], cand.get("market"), cand["url_template"], cand.get("format"),
                cand.get("symbol_kind"), cand.get("frequency"), cand.get("fields"),
                "verified", found_by, datetime.date.today().isoformat(), sample))
    finally:
        con.close()


def listing(market=None, status=None, db_path=None):
    con = _con(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute("SELECT * FROM analyst_sources ORDER BY market, name")]
    finally:
        con.close()
    markets = None if market is None else ({market} if isinstance(market, str) else set(market))
    return [r for r in rows if (markets is None or r["market"] in markets)
            and (status is None or r["status"] == status)]


def set_status(name, status, db_path=None, checked=False):
    """checked=True also stamps today as the last verification."""
    con = _con(db_path)
    try:
        with con:
            if checked:
                return con.execute("UPDATE analyst_sources SET status=?, verified_at=? "
                                   "WHERE name=?", (status, datetime.date.today()
                                                    .isoformat(), name)).rowcount
            return con.execute("UPDATE analyst_sources SET status=? WHERE name=?",
                               (status, name)).rowcount
    finally:
        con.close()


def _validate(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("sources"), list):
        return None
    out = [s for s in obj["sources"] if isinstance(s, dict) and s.get("name")
           and s.get("url_template")]
    return out or None


def run(market, call, fetch=None, brain="", db_path=None, today=None, log=print):
    """Ask `call` for candidates, verify each, save the good ones. Report lines."""
    from core.analyst import agent

    fetch = fetch or tools._get
    example = EXAMPLE.get(market)
    cands = agent.ask_json(PROMPT % (market, example or "none (macro series)"), call,
                           _validate, asset=example) or []
    report = []
    for c in cands:
        name = re.sub(r"[^a-z0-9_]", "_", str(c["name"]).lower())[:40]
        c = {**c, "name": name, "market": market}
        ok, why, sample = verify(c, fetch, asset=example, today=today)
        if ok:
            save(c, brain, sample, db_path=db_path)
        report.append("%s %s: %s" % ("KEPT" if ok else "no  ", name, why))
        log("  " + report[-1])
    if not cands:
        report.append("the brain returned no usable candidates")
        log("  " + report[-1])
    return report


def recheck(fetch=None, db_path=None, today=None):
    """Re-verify every verified or broken source; returns (still_ok, broken)."""
    fetch = fetch or tools._get
    good = bad = 0
    for r in listing(db_path=db_path):
        if r["status"] == "off":
            continue
        ok, _, _ = verify(r, fetch, asset=EXAMPLE.get(r["market"]), today=today)
        set_status(r["name"], "verified" if ok else "broken", db_path=db_path, checked=True)
        good, bad = good + ok, bad + (not ok)
    return good, bad


def _raw_source(asset, today=None, name=""):
    market = market_of(asset) if asset else "macro"
    if not name:
        return [{"name": r["name"], "market": r["market"], "fields": r["fields"],
                 "frequency": r["frequency"]}
                for r in listing(market=(market, "macro"), status="verified")]
    row = next((r for r in listing(status="verified") if r["name"] == name), None)
    if row is None:
        return {"error": "no verified source named %r; call raw_source with no name "
                         "for the list" % name}
    asset_for = asset if row["market"] != "macro" else None
    rows = parse_rows(tools._get(fill(row["url_template"], asset_for, row["symbol_kind"])))
    key = _date_key(rows)
    if key:        # some feeds list the newest first; the tail must be the newest
        rows.sort(key=lambda r: _as_date(r.get(key)) or datetime.date.min)
    return rows[-30:]


tools.register(tools.Tool(
    name="raw_source", args={"name": "a source name, or empty for the list"},
    rewinds=False,
    describe="raw data sources found and verified for this market (empty name "
             "lists them; a name returns its latest rows)",
    run=_raw_source))
