"""Pull hourly bars into intraday.db. All the I/O of the intraday spike.

Deliberately separate from data_engine.py: this phase must not be able to
write to market.db, and a separate file makes that structural rather than
careful. The transport is the existing net.http_get, so route selection,
retries and the SOCKS5 fallback are inherited rather than rewritten.
"""
import os
import sqlite3
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "intraday.db")

YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/"
             "{sym}?interval=1h&range={range}")
# The first fetch takes everything the vendor will give. A top-up asks for a
# short window instead: the store already holds the history, INSERT OR REPLACE
# heals any bar written before it was final, and asking 844 assets for two
# years each to collect nine days is the reason nothing ever topped this up.
YAHOO_FULL_RANGE = "730d"
YAHOO_TOPUP_RANGE = "1mo"

MOEX_TZ = "Europe/Moscow"
MOEX_URL = ("https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
            "securities/{sym}/candles.json?interval=60&from={start}&start={offset}")


def parse_yahoo(payload, now=None):
    """(bars, exchange timezone name) from a Yahoo chart payload.

    The timezone comes from the payload rather than from a hardcoded exchange
    table, because Yahoo already knows it (America/New_York, Europe/Berlin,
    UTC) and a table would go stale silently. Epoch seconds are UTC, so the
    conversion is exact and happens here, once, on the way in.

    An hour that has not ended is not a bar yet and is dropped: the daily
    fetchers stored unfinished sessions and never replaced them (2026-09-11).
    """
    now = int(datetime.now(UTC).timestamp()) if now is None else now
    try:
        res = payload["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        return [], None
    tz = res.get("meta", {}).get("exchangeTimezoneName")
    stamps = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    out = []
    for i, t in enumerate(stamps):
        row = {k: (q.get(k) or [None] * len(stamps))[i]
               for k in ("open", "high", "low", "close", "volume")}
        # A null close, or an hour that has not ended, is not a bar yet.
        if row["close"] is None or int(t) + 3600 > now:
            continue
        bar = {k: (float(v) if v is not None else None) for k, v in row.items()}
        bar["ts"] = datetime.fromtimestamp(int(t), tz=UTC).isoformat()
        out.append(bar)
    return out, tz


def parse_moex(payload):
    """Bars from one ISS candles page, converted out of Moscow local time.

    ISS returns a NAIVE local string ("2026-09-04 10:00:00"). Yahoo returns
    epoch seconds in UTC. Storing both as they arrive is a silent three-hour
    shift on every Russian asset, which is why the conversion happens here and
    the column is UTC everywhere downstream.
    """
    try:
        cols = payload["candles"]["columns"]
        data = payload["candles"]["data"]
    except (KeyError, TypeError):
        return []
    idx = {c: i for i, c in enumerate(cols)}
    now_msk = datetime.now(ZoneInfo(MOEX_TZ)).replace(tzinfo=None)
    out = []
    for row in data:
        # the candle of the hour still in progress is not a bar yet
        if "end" in idx and datetime.strptime(row[idx["end"]], "%Y-%m-%d %H:%M:%S") > now_msk:
            continue
        naive = datetime.strptime(row[idx["begin"]], "%Y-%m-%d %H:%M:%S")
        ts = naive.replace(tzinfo=ZoneInfo(MOEX_TZ)).astimezone(UTC)
        out.append({"ts": ts.isoformat(),
                    "open": float(row[idx["open"]]),
                    "high": float(row[idx["high"]]),
                    "low": float(row[idx["low"]]),
                    "close": float(row[idx["close"]]),
                    "volume": float(row[idx["volume"]])})
    return out


SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1h (
    asset  TEXT NOT NULL,
    ts     TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (asset, ts)
)
"""


def store(asset, bars, db_path=None):
    """Insert bars; a bar already present is replaced by the fresh one, so a
    refetch heals a bar stored before it was final. Returns rows added.

    One table for every asset, not one table each: this store is deleted
    wholesale when the gate closes, and its queries are cross-sectional.
    """
    con = sqlite3.connect(db_path or DB_PATH)
    try:
        con.execute(SCHEMA)
        before = con.execute("SELECT count(*) FROM bars_1h WHERE asset = ?",
                             (asset,)).fetchone()[0]
        con.executemany(
            "INSERT OR REPLACE INTO bars_1h "
            "(asset, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            [(asset, b["ts"], b.get("open"), b.get("high"), b.get("low"),
              b.get("close"), b.get("volume")) for b in bars])
        con.commit()
        after = con.execute("SELECT count(*) FROM bars_1h WHERE asset = ?",
                            (asset,)).fetchone()[0]
        return after - before
    finally:
        con.close()


def last_ts(asset, db_path=None):
    """The newest stored bar's timestamp for one asset, or None.

    Separate from load() on purpose: deciding whether an asset needs a top-up
    means reading ONE value, and fetch_all used to answer it by loading every
    bar the asset has - 514042 rows across 844 assets to learn 844 booleans.
    Like load(), it must never create the database.
    """
    path = db_path or DB_PATH
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(path)
    try:
        if not con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name='bars_1h'").fetchone():
            return None
        row = con.execute("SELECT max(ts) FROM bars_1h WHERE asset = ?",
                          (asset,)).fetchone()
        return row[0] if row and row[0] else None
    finally:
        con.close()


def load(asset, db_path=None):
    """Every stored bar for one asset, oldest first, or [] when there is none.

    Deliberately does NOT create the database: a reader that runs CREATE TABLE
    conjures a store wherever it is imported, which is how the test suite once
    grew a stub market.db.
    """
    path = db_path or DB_PATH
    if not os.path.exists(path):
        return []
    con = sqlite3.connect(path)
    try:
        if not con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name='bars_1h'").fetchone():
            return []
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT ts, open, high, low, close, volume FROM bars_1h "
                           "WHERE asset = ? ORDER BY ts", (asset,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


MOEX_INDEX_URL = ("https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/"
                  "securities/{sym}/candles.json?interval=60&from={start}&start={offset}")
MOEX_START = "2015-01-01"
MOEX_PAGE = 500


def source_for(asset):
    """Which exchange serves this asset. config.MOEX_ASSETS is the one list."""
    import config
    return "moex" if asset in config.MOEX_ASSETS else "yahoo"


def vendor_symbol(asset):
    """What the source calls this asset.

    config.FULL_ASSET_MAP is already the single source of that mapping (SP500 is
    ^GSPC to Yahoo, GOLD is GC=F), so the universe file deliberately does not
    carry a second copy that could drift out of step with it.
    """
    import config
    return config.FULL_ASSET_MAP.get(asset, asset)


def fetch_asset(asset, http=None, since=None, db_path=None):
    """Fetch one asset's hourly history and store it. (rows written, tz).

    `http` is injectable so tests never open a socket; production passes None
    and gets net.http_get, which already carries route selection and retries.

    `since` is a date string (YYYY-MM-DD) to start from, and it is what makes a
    top-up cheap rather than a second full download. It only narrows the REQUEST:
    the store is keyed on (asset, ts) and replaces, so a narrow window can add
    bars and heal recent ones but can never truncate the history.
    """
    if http is None:
        import net

        def http(url, **kw):
            return net.http_get(url, route="auto", retries=2, timeout=(10, 40))

    sym = vendor_symbol(asset)

    if source_for(asset) == "moex":
        # An index lives on a different board than a share. data_engine already
        # special-cases IMOEX for the daily fetch; the same split applies here.
        url = MOEX_INDEX_URL if asset == "IMOEX" else MOEX_URL
        start = since or MOEX_START
        bars, offset = [], 0
        while True:
            page = http(url.format(sym=sym, start=start, offset=offset)).json()
            got = parse_moex(page)
            bars.extend(got)
            if len(got) < MOEX_PAGE:
                break
            offset += MOEX_PAGE
        store_tz(asset, MOEX_TZ)
        return store(asset, bars, db_path=db_path), MOEX_TZ

    rng = YAHOO_TOPUP_RANGE if since else YAHOO_FULL_RANGE
    payload = http(YAHOO_URL.format(sym=sym, range=rng)).json()
    bars, tz = parse_yahoo(payload)
    store_tz(asset, tz)
    return store(asset, bars, db_path=db_path), tz


def fetch_all(assets, http=None, refetch=False, db_path=None, log=print, mode=None):
    """Fetch every asset. Returns {asset: "skip" | "<rows> rows" | "error: ..."}.

    Three modes, and `mode` wins when given; otherwise it is derived from the
    old `refetch` flag so every existing caller keeps its exact meaning.

      new      (default, refetch=False) only assets with nothing stored.
      refetch  (refetch=True) everything, from the beginning.
      top_up   only the bars since each asset's newest stored one. This is the
               one that was missing, and its absence is why the store ran from
               2 to 9 days stale per asset: "new" skips an asset that has any
               bars at all, so there was no setting that meant "keep it fresh".

    An error is recorded and the loop carries on: one dead ticker must not stop
    the rest.
    """
    if mode is None:
        mode = "refetch" if refetch else "new"
    if mode not in ("new", "refetch", "top_up"):
        raise ValueError("unknown mode %r" % (mode,))

    out = {}
    for i, asset in enumerate(assets, 1):
        newest = last_ts(asset, db_path=db_path)
        if mode == "new" and newest:
            out[asset] = "skip"
            continue
        # Only an asset that HAS history can be topped up from a date; one that
        # has none falls through to a full fetch rather than being skipped.
        since = newest[:10] if (mode == "top_up" and newest) else None
        try:
            rows, _tz = fetch_asset(asset, http=http, since=since, db_path=db_path)
            out[asset] = "%d rows" % rows if rows else "error: no bars"
        except Exception as exc:
            out[asset] = "error: %s" % str(exc)[:80]
        log("[%d/%d] %-12s %s" % (i, len(assets), asset, out[asset]))
    return out


TZ_SCHEMA = """
CREATE TABLE IF NOT EXISTS asset_tz (
    asset TEXT PRIMARY KEY,
    tz    TEXT NOT NULL
)
"""


def store_tz(asset, tz, db_path=None):
    """Remember which exchange zone an asset's bars are stamped against.

    Bars alone are not enough: to_sessions needs the zone, and defaulting to
    UTC silently mis-groups every asset that trades across midnight. Measured
    2026-09-08, guessing UTC scored forex at 0.65 and gold at 0.11 on the
    round-trip while the same assets are fine once their real zone is used.
    """
    if not tz:
        return
    con = sqlite3.connect(db_path or DB_PATH)
    try:
        con.execute(TZ_SCHEMA)
        con.execute("INSERT OR REPLACE INTO asset_tz (asset, tz) VALUES (?,?)",
                    (asset, tz))
        con.commit()
    finally:
        con.close()


def load_tz(asset, db_path=None):
    """The stored zone, or None. Never a default: a wrong zone is worse than a
    missing one, because it produces plausible sessions that are quietly off."""
    path = db_path or DB_PATH
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(path)
    try:
        if not con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name='asset_tz'").fetchone():
            return None
        row = con.execute("SELECT tz FROM asset_tz WHERE asset = ?", (asset,)).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def main(argv=None):
    """The entry point this file did not have.

    Without it `python intraday_fetch.py` did nothing at all, the only caller of
    fetch_all in the tree was intraday_research.py in a worktree, and the card
    on the asset page read a store that nothing in the serving checkout ever
    refreshed.
    """
    import argparse

    import config

    ap = argparse.ArgumentParser(description="Hourly bars for intraday.db.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--top-up", action="store_true",
                   help="only bars newer than each asset's last stored one "
                        "(the daily/periodic setting)")
    g.add_argument("--new", action="store_true",
                   help="only assets with nothing stored yet (default)")
    g.add_argument("--refetch", action="store_true",
                   help="every asset from the beginning; slow, heals everything")
    ap.add_argument("--assets", default=None,
                    help="comma-separated subset, for a smoke run")
    args = ap.parse_args(argv)

    assets = ([a.strip().upper() for a in args.assets.split(",") if a.strip()]
              if args.assets else list(config.FULL_ASSET_MAP))
    mode = "top_up" if args.top_up else ("refetch" if args.refetch else "new")

    out = fetch_all(assets, mode=mode)
    added = sum(1 for s in out.values() if s.endswith("rows"))
    skipped = sum(1 for s in out.values() if s == "skip")
    bad = sorted(a for a, s in out.items() if s.startswith("error"))
    print("\nmode %s: %d fetched, %d skipped, %d error(s)"
          % (mode, added, skipped, len(bad)))
    if bad:
        print("errors: %s" % ", ".join(bad[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
