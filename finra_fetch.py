"""FINRA daily short-sale volume for the US names, into market.db.

FINRA publishes CNMSshvolYYYYMMDD.txt (consolidated NMS short volume per symbol)
after each US session, so a row dated d is known before d+1 opens. The only free
flow source that held up on 2026-09-29: the 60-day z of short/total volume has a
cross-sectional IC of -0.016 with the next day's return, t -6.2 over 783 days,
negative in every year (memory: project_atratus_flow_data_sources). Stored as
rows (date, asset table name, short, total); core.features.add_finra_features
turns them into model inputs, used only when named in GTRADE_EXTRA_FEATURES.

    python finra_fetch.py              fetch what is missing (first run: ~2 years)
    python finra_fetch.py --days 30    only the last 30 calendar days
"""

import argparse
import datetime as dt
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol%s.txt"
BACKFILL_DAYS = 750

DDL = """
CREATE TABLE IF NOT EXISTS finra_shvol (
    date TEXT, asset TEXT, short REAL, total REAL,
    PRIMARY KEY (date, asset)
)
"""


def symbol_map():
    """{FINRA symbol: market.db table} for the US names of the asset map."""
    from config import FULL_ASSET_MAP, radar_category

    out = {}
    for name, sym in FULL_ASSET_MAP.items():
        if radar_category(name) != "us" or any(c in sym for c in "=^") or "." in sym:
            continue
        table = name.lower().replace("^", "").replace(".", "").replace("-", "")
        out[sym.replace("-", ".")] = table
    return out


def parse(text, sym2table):
    """[(date, table, short, total)] for our symbols in one FINRA file."""
    rows = []
    for line in (text or "").splitlines()[1:]:
        p = line.split("|")
        if len(p) < 5 or p[1] not in sym2table:
            continue
        try:
            d = "%s-%s-%s" % (p[0][:4], p[0][4:6], p[0][6:8])
            rows.append((d, sym2table[p[1]], float(p[2]), float(p[4])))
        except ValueError:
            continue
    return rows


def _fetch_day(day, sym2table):
    import net

    try:
        r = net.http_get(URL % day.strftime("%Y%m%d"), route="auto", retries=2)
    except Exception:
        return []
    if r.status_code != 200:
        return []                      # a holiday, or not published yet
    return parse(r.text, sym2table)


def update(days=None, db_path=None, workers=8, log=print):
    """Fetch the missing days; returns how many rows were stored."""
    con = sqlite3.connect(db_path or DB_PATH)
    try:
        con.execute(DDL)
        last = con.execute("SELECT MAX(date) FROM finra_shvol").fetchone()[0]
    finally:
        con.close()
    today = dt.date.today()
    start = today - dt.timedelta(days=days or BACKFILL_DAYS)
    if last and not days:
        start = max(start, dt.date.fromisoformat(last) + dt.timedelta(days=1))
    wanted = [start + dt.timedelta(days=i) for i in range((today - start).days + 1)]
    wanted = [d for d in wanted if d.weekday() < 5]
    if not wanted:
        log("  FINRA: up to date")
        return 0
    sym2table = symbol_map()
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(lambda d: _fetch_day(d, sym2table), wanted) for r in chunk]
    con = sqlite3.connect(db_path or DB_PATH)
    try:
        with con:
            con.execute(DDL)
            con.executemany("INSERT OR REPLACE INTO finra_shvol VALUES (?,?,?,?)", rows)
    finally:
        con.close()
    log("  FINRA: %d row(s) over %d day(s) asked, %s..%s"
        % (len(rows), len(wanted), wanted[0], wanted[-1]))
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=None,
                    help="calendar days back to (re)fetch; default: what is missing")
    args = ap.parse_args(argv)
    import urllib3  # GTRADE_SSL_VERIFY=0 behind the proxy, as in data_engine
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    update(days=args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
