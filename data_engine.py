import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import urllib3
from sqlalchemy import create_engine, text

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

import net
from core.logger import get_logger

logger = get_logger("data_engine")

try:
    from config import FULL_ASSET_MAP, MOEX_ASSETS
except ImportError:
    sys.exit(f"[ERR] config.py not found in {BASE_DIR}")

engine = create_engine(f'sqlite:///{os.path.join(BASE_DIR, "market.db")}')
_db_lock = threading.Lock()  # serialize SQLite writes


def _env_int(name, default):
    """int from an environment variable, else default (empty/garbage also falls back to default)."""
    try:
        return int((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default


# Everything is configurable via the environment, no hardcoded constants.
DAILY_WORKERS = _env_int("GTRADE_DAILY_WORKERS", 12)
WEEKLY_WORKERS = _env_int("GTRADE_WEEKLY_WORKERS", 10)
RETRY_SWEEPS = _env_int("GTRADE_RETRY_SWEEPS", 2)   # extra passes over assets with ERROR
MOEX_RETRIES = _env_int("GTRADE_MOEX_RETRIES", 5)   # attempts per request within a single pass

# How far back to fetch on a first/backfill pull. Default 15y (verified that the
# Yahoo chart endpoint returns clean daily over a 15y explicit window - the
# quarterly-degradation issue only affects range=max, not period1/period2).
HISTORY_DAYS = _env_int("GTRADE_HISTORY_DAYS", 15 * 365)
# Backfill mode: re-fetch the full HISTORY_DAYS window for every asset (last_date
# forced to None) so existing tables, which the incremental fetch only extends
# forward, get their missing OLDER bars. _save_df dedupes by date, so this only
# inserts bars not already stored. One-time: GTRADE_BACKFILL=1 python data_engine.py
BACKFILL = (os.getenv("GTRADE_BACKFILL") or "").strip() in ("1", "true", "True")
# MOEX history start used on a first/backfill pull (ISS returns only what exists,
# so requesting earlier than a ticker's listing is harmless).
MOEX_START_STR = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')
# Which keys come from MOEX rather than Yahoo. The list lives in config beside
# the asset map: kept here as a second copy, a ticker added to the map and
# forgotten here was fetched from Yahoo, returned nothing and stayed stale.
MOEX_TARGETS = list(MOEX_ASSETS)

def get_last_date(table_name):
    """Find the date of the most recent bar in the database."""
    try:
        with engine.connect() as conn:
            # Check whether the table exists
            check = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
            if not check.fetchone(): return None

            # Get the most recent date
            # Quoted: a ticker whose table name is a SQL keyword, ALL for
            # Allstate, made this a syntax error. The except below then
            # returned None, which reads as "no table", so the whole history
            # was refetched and appended on every run.
            res = conn.execute(text(f'SELECT MAX(Date) FROM "{table_name}"'))
            last_date_str = res.fetchone()[0]
            if last_date_str:
                return pd.to_datetime(last_date_str)
    except Exception as e:
        print(f"SQL Error: {e}")
    return None


def _drop_existing_dates(df, table_name):
    """Remove dates from df that are already in the DB - guards against duplicates."""
    try:
        with engine.connect() as conn:
            check = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
            if not check.fetchone():
                return df  # table doesn't exist - everything is new
            existing = pd.read_sql(f'SELECT DISTINCT Date FROM "{table_name}"', conn)
            if existing.empty:
                return df
            existing_set = set(existing['Date'].astype(str).str[:10])
            mask = ~df.index.astype(str).str[:10].isin(existing_set)
            return df[mask]
    except Exception:
        return df  # on error - write as-is

def _daily_table(name):
    """The daily table an asset key is stored in."""
    return name.lower().replace("^", "").replace(".", "").replace("-", "")


# A bar is stored only once its session has closed. The fetchers used to store
# the bar of a session in progress and then resume from the NEXT day, so the
# snapshot was never replaced: measured 2026-09-11 against a fresh fetch,
# stored closes were off by more than 0.1% on 38 of 60 recent days for SBER,
# 43 of 59 for BTC, 40 of 52 for GOLD, in every month of daily runs since March
# 2026 and in none of the bulk-fetched months before it.
def _drop_unfinished_session(df, meta, now_ts):
    """Drop Yahoo daily bars that belong to the trading period still running.

    Yahoo says when the current regular session starts and ends; any bar
    stamped at or after that start, while it has not ended, is still being
    written. Without the metadata nothing is dropped.
    """
    reg = ((meta or {}).get("currentTradingPeriod") or {}).get("regular") or {}
    start, end = reg.get("start"), reg.get("end")
    if start is None or end is None or now_ts >= end:
        return df
    return df[df["Date"] < pd.Timestamp(datetime.fromtimestamp(start))]


def _moex_today():
    """The Moscow calendar date. A MOEX candle for this date is still being
    written until the evening session closes."""
    return datetime.now(ZoneInfo("Europe/Moscow")).date()


def _weekly_from_daily(table, stamps):
    """Weekly bars built from our own daily rows, for weeks the vendor returned
    empty. Yahoo serves TON's weeks after 2026-06-15 with every field null while
    its daily series is intact; dropping them froze TON's weekly table."""
    try:
        daily = pd.read_sql(f'SELECT Date, open, high, low, close, volume FROM "{table}"',
                            engine, parse_dates=["Date"])
    except Exception:
        return pd.DataFrame(columns=["Date", "Open", "Close", "High", "Low", "Volume"])
    rows = []
    for stamp in stamps:
        start = pd.Timestamp(stamp).normalize()
        wk = daily[(daily["Date"] >= start) & (daily["Date"] < start + pd.Timedelta(days=7))]
        wk = wk.sort_values("Date")
        if wk.empty:
            continue
        rows.append({"Date": stamp, "Open": wk["open"].iloc[0], "Close": wk["close"].iloc[-1],
                     "High": wk["high"].max(), "Low": wk["low"].min(),
                     "Volume": wk["volume"].sum()})
    return pd.DataFrame(rows, columns=["Date", "Open", "Close", "High", "Low", "Volume"])


def _yahoo_chart_ok(r) -> bool:
    """True only for a valid Yahoo response. A 200 with error/no result means the
    route landed on the wrong endpoint, so net.http_get tries another one. An empty
    response still has a 'result' key."""
    if r.status_code != 200:
        return False
    try:
        chart = r.json().get('chart', {})
    except Exception:
        return False
    return not chart.get('error') and bool(chart.get('result'))


def fetch_yahoo_smart(symbol, last_date):
    y_sym = FULL_ASSET_MAP.get(symbol, symbol)
    if symbol in ['SOL', 'BNB', 'DOGE', 'XRP', 'TON', 'BTC', 'ETH'] and '-USD' not in y_sym:
        y_sym = f"{symbol}-USD"

    now_ts = int(datetime.now().timestamp())
    _orig_last_date = last_date

    if last_date is not None:
        start_ts = int(last_date.timestamp()) + 86400
        if start_ts >= now_ts:
            print(f"   - [YAHOO] {y_sym:<12} [OK] (UP_TO_DATE)")
            return None
    else:
        # Start from HISTORY_DAYS ago using an explicit period1/period2 window.
        # range=max with interval=1d returns quarterly data for long histories;
        # an explicit window does not, so this stays clean daily even at 15y.
        start_ts = now_ts - HISTORY_DAYS * 86400

    base_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_sym}?interval=1d"
    end_ts = max(now_ts - 300, start_ts + 1)
    url = f"{base_url}&period1={start_ts}&period2={end_ts}"

    print(f"   - [YAHOO] {y_sym:<12}", end=" ", flush=True)

    def _parse_response(r, route_label):
        if r.status_code != 200:
            snippet = (r.text or '')[:180].replace('\n', ' ')
            msg = f"HTTP {r.status_code} | route={route_label} | body={snippet}"
            return None, msg

        payload = r.json()
        chart = payload.get('chart', {})
        err = chart.get('error')
        if err:
            return None, f"Yahoo chart error | route={route_label}: {err}"

        result = chart.get('result') or []
        if not result:
            return None, f"empty result | route={route_label}"

        res = result[0]
        if 'timestamp' not in res:
            return "UP_TO_DATE", None

        q = res['indicators']['quote'][0]
        df = pd.DataFrame({
            'Date': [datetime.fromtimestamp(ts) for ts in res['timestamp']],
            'Open': q['open'], 'Close': q['close'], 'High': q['high'], 'Low': q['low'], 'Volume': q['volume']
        }).dropna()

        df['Date'] = pd.to_datetime(df['Date'])
        df = _drop_unfinished_session(df, res.get('meta'), now_ts)
        if _orig_last_date is not None:
            df = df[df['Date'] > _orig_last_date]

        if df.empty:
            return "NO_NEW", None

        return df.set_index('Date'), None

    # net.http_get picks the working route itself (direct/SOCKS5) and does a
    # failover on a network error or when validate rejects a 200 with an empty
    # or error body from Yahoo.
    try:
        r = net.http_get(url, route="auto", validate=_yahoo_chart_ok)
    except requests.exceptions.RequestException as exc:
        logger.error(f"{y_sym} yahoo fail: {exc}")
        print(f"[ERR] (RequestException: {exc})")
        return None

    parsed, err = _parse_response(r, "auto")
    if err is not None:
        logger.error(f"{y_sym} yahoo fail: {err}")
        print(f"[ERR] ({err})")
        return None
    if isinstance(parsed, str) and parsed == "UP_TO_DATE":
        print("[OK] (Up to date)")
        return None
    if isinstance(parsed, str) and parsed == "NO_NEW":
        print("[OK] (No new data)")
        return None
    print(f"[OK] (+{len(parsed)} new bars)")
    return parsed

def fetch_moex_smart(symbol, last_date):
    clean = symbol.split('.')[0]
    print(f"   - [MOEX] {clean:<12}", end=" ", flush=True)

    # Already up to date: the next bar would start in the future, MOEX would return
    # nothing. Skip the request (same as Yahoo's start_ts >= now check) to avoid
    # hammering the unstable direct route and piling up transient timeouts in the log.
    if last_date is not None and last_date.date() >= datetime.now().date():
        print("[OK] (Up to date)")
        return None

    # MOEX lets us specify a start date as 'YYYY-MM-DD'
    start_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d') if last_date else MOEX_START_STR

    all_data = []
    net_failed = False  # distinguish "connection died" from "no new data"
    # Download in chunks (in case the history is large)
    for offset in [0, 500, 1000]:
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{clean}/candles.json?interval=24&from={start_str}&start={offset}"
        if clean == "IMOEX":
             url = f"https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json?interval=24&from={start_str}&start={offset}"

        try:
            r = net.http_get(url, route="auto", retries=MOEX_RETRIES)
            data = r.json()['candles']['data']
            if not data: break
            all_data.extend(data)
            cols = r.json()['candles']['columns']
            if len(data) < 500: break # End of data
        except Exception as exc:
            # A failure on the FIRST page means we never reached MOEX - this is
            # an error, NOT "up to date". MOEX needs a Russian IP; a foreign VPN
            # exit gets connection-reset. Report it instead of masking.
            if offset == 0:
                net_failed = True
                logger.warning("MOEX fetch failed for %s: %s", clean, exc)
            break

    if not all_data:
        if net_failed:
            print("[ERR] (MOEX unreachable - need RU IP / direct route)")
            return None
        print("[OK] (Up to date)"); return None

    df = pd.DataFrame(all_data, columns=cols).rename(columns={'begin': 'Date', 'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
    df['Date'] = pd.to_datetime(df['Date'])
    # Today's candle is still being written: see _drop_unfinished_session.
    df = df[df['Date'].dt.date < _moex_today()]

    if last_date:
        df = df[df['Date'] > last_date]

    if df.empty:
        print("[OK] (No new data)"); return None

    print(f"[OK] (+{len(df)} new bars)")
    return df.set_index('Date')

def fetch_moex_weekly(symbol, last_date):
    """Fetch weekly (interval=7) OHLCV bars from MOEX ISS API."""
    clean = symbol.split('.')[0]
    print(f"   - [WEEKLY RU] {clean:<12}", end=" ", flush=True)

    # Already up to date: skip the request instead of hitting the unstable route.
    if last_date is not None and last_date.date() >= datetime.now().date():
        print("[OK] (Up to date)")
        return None

    start_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d') if last_date else MOEX_START_STR

    all_data = []
    cols = None
    for offset in [0, 500, 1000, 1500]:
        if clean == "IMOEX":
            url = (f"https://iss.moex.com/iss/engines/stock/markets/index/"
                   f"boards/SNDX/securities/IMOEX/candles.json"
                   f"?interval=7&from={start_str}&start={offset}")
        else:
            url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/"
                   f"boards/TQBR/securities/{clean}/candles.json"
                   f"?interval=7&from={start_str}&start={offset}")
        try:
            r = net.http_get(url, route="auto", retries=MOEX_RETRIES)
            payload = r.json()['candles']
            data = payload['data']
            if not data:
                break
            cols = payload['columns']
            all_data.extend(data)
            if len(data) < 500:
                break
        except Exception:
            break

    if not all_data:
        print("[OK] (Up to date)")
        return None

    df = pd.DataFrame(all_data, columns=cols).rename(columns={
        'begin': 'Date', 'open': 'Open', 'close': 'Close',
        'high': 'High', 'low': 'Low', 'volume': 'Volume',
    })
    df['Date'] = pd.to_datetime(df['Date'])

    # Same rule as the Yahoo weekly fetch, and it belongs here too: ISS also
    # serves the week IN PROGRESS, so a daily update stored a bar whose close
    # was that day rather than the week's. Measured 2026-09-09, after the Yahoo
    # side was fixed, exactly 181 tables still carried one - the 181 MOEX names,
    # which is how this second copy of the defect announced itself.
    #
    # A bar stamped at the start of week W is complete once seven days have
    # passed. Age, not the gap to the previous bar: legitimate holes of 8, 14
    # and 35 days exist all over the history.
    df = df[df['Date'] <= pd.Timestamp(datetime.now()) - pd.Timedelta(days=7)]

    if last_date:
        df = df[df['Date'] > last_date]

    if df.empty:
        print("[OK] (No new data)")
        return None

    print(f"[OK] (+{len(df)} weekly bars)")
    return df.set_index('Date')


def fetch_yahoo_weekly(symbol, last_date):
    """Fetch weekly (1wk) OHLCV bars from Yahoo Finance for multi-timeframe analysis."""
    y_sym = FULL_ASSET_MAP.get(symbol, symbol)
    if symbol in ['SOL', 'BNB', 'DOGE', 'XRP', 'TON', 'BTC', 'ETH'] and '-USD' not in y_sym:
        y_sym = f"{symbol}-USD"

    now_ts = int(datetime.now().timestamp())
    _orig_last_date = last_date  # saved for filtering

    # A long window even for a one-day top-up: a three-day window returns a
    # single bar and nothing to align it against, so the partial-bar rule below
    # would have no previous stamp to measure. _orig_last_date still decides
    # what gets written, so this costs one call and no duplicate rows.
    start_ts = now_ts - HISTORY_DAYS * 86400
    if last_date is not None and int(last_date.timestamp()) + 86400 >= now_ts:
        print(f"   - [WEEKLY] {y_sym:<12} [OK] (UP_TO_DATE)")
        return None

    base_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_sym}?interval=1wk"
    url = f"{base_url}&period1={start_ts}&period2={now_ts}"

    print(f"   - [WEEKLY] {y_sym:<12}", end=" ", flush=True)

    try:
        r = net.http_get(url, route="auto", validate=_yahoo_chart_ok)
    except requests.exceptions.RequestException as exc:
        logger.warning("Weekly fetch error for %s: %s", y_sym, exc)
        print("[ERR] (failed all routes)")
        return None

    chart = r.json().get('chart', {})
    res = chart['result'][0]
    if 'timestamp' not in res:
        print("[OK] (UP_TO_DATE)")
        return None
    q = res['indicators']['quote'][0]
    raw = pd.DataFrame({
        'Date':   [datetime.fromtimestamp(ts) for ts in res['timestamp']],
        'Open':   q['open'], 'Close': q['close'],
        'High':   q['high'], 'Low':   q['low'],
        'Volume': q['volume'],
    })
    empty = raw['Close'].isna()
    df = raw[~empty].dropna()
    if empty.any():
        df = pd.concat([df, _weekly_from_daily(_daily_table(symbol), raw.loc[empty, 'Date'])])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # DROP EVERY WEEK THAT HAS NOT FINISHED. Measured 2026-09-08: interval=1wk
    # appends a bar for the week in progress, stamped with the day of the
    # request rather than the week's start, at every window length:
    #
    #     20-day window -> Mon 08-17, Mon 08-24, Mon 08-31, Mon 09-07, Tue 09-08
    #
    # A daily update wrote one such row per day, and 665 of the 850 weekly
    # tables filled with daily-spaced rows dated after 2026-03-04.
    #
    # The rule is AGE, not the gap between the last two stamps and not the
    # weekday. A gap rule was tried first and is wrong twice over: the bar above
    # sits 8 days after its predecessor and slips through, while legitimate
    # holes of 8, 14 and 35 days exist all over the history (969 rows before
    # 2026 alone). And when the fetch happens to run ON the grid weekday, the
    # partial bar carries a perfectly ordinary stamp - it would be stored with
    # one day's data as the week's close, and never overwritten afterwards,
    # because the incremental filter only takes dates AFTER the newest row.
    #
    # A bar stamped at the start of week W is complete once seven days have
    # passed. Anything younger is still being written.
    cutoff = pd.Timestamp(datetime.fromtimestamp(now_ts)) - pd.Timedelta(days=7)
    df = df[df['Date'] <= cutoff]
    if df.empty:
        print("[OK] (no completed week yet)")
        return None
    if _orig_last_date is not None:
        # By date: the stored row is a bare date, the vendor stamp carries a
        # time of day, and 03:00 compared raw is "later" than the row it is.
        df = df[df['Date'].dt.normalize() > pd.Timestamp(_orig_last_date).normalize()]
    if df.empty:
        print("[OK] (No new data)")
        return None
    print(f"[OK] (+{len(df)} weekly bars)")
    return df.set_index('Date')


def scrub_ohlc(df):
    """Repair or drop bars whose prices are not positive, before they are stored.

    A provider that returns a zero is not reporting a price of nothing, it is
    failing to report. Stored as-is, a bar with high = low = 0 gives a true
    range the size of the whole price and an ATR spike to match, and a trade
    priced off it returns inf - which is how the trade-levels gate came to
    report `mean_d nan` over 316 assets on 2026-08-22 because ONE bar of AZN
    read zero.

    Two treatments, because dropping every bad bar would also throw away real
    trading days: when the close is good but open/high/low are not, fill them
    from what IS known (a bar with only a close has no intraday range,
    and inventing one would be worse than recording none); when the close
    itself is not positive there is no price at all, so the bar goes. A gap is
    honest, a zero is not.

    Returns (clean_df, n_repaired, n_dropped).
    """
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if "close" not in cols or df.empty:
        return df, 0, 0
    df = df.copy()
    price = df[cols].apply(pd.to_numeric, errors="coerce")
    bad = (price <= 0).any(axis=1) | price.isna().any(axis=1)
    if not bad.any():
        return df, 0, 0

    no_price = bad & (~(price["close"] > 0))
    fixable = bad & (price["close"] > 0)

    if fixable.any():
        close = price.loc[fixable, "close"]
        for c in ("open", "high", "low"):
            if c in cols:
                col = price.loc[fixable, c]
                df.loc[fixable, c] = col.where(col > 0, close)
        # keep the bar internally consistent: the high is the top of what
        # happened that day, the low the bottom.
        known = df.loc[fixable, [c for c in ("open", "high", "low", "close")
                                 if c in cols]].apply(pd.to_numeric, errors="coerce")
        if "high" in cols:
            df.loc[fixable, "high"] = known.max(axis=1)
        if "low" in cols:
            df.loc[fixable, "low"] = known.min(axis=1)

    return df[~no_price], int(fixable.sum()), int(no_price.sum())


def _save_df(df, table_name):
    """Normalize and append DataFrame to SQLite."""
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex) and 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index).normalize()
    df.index = df.index.strftime('%Y-%m-%d')
    df = df[~df.index.duplicated(keep='last')]
    df, _fixed, _dropped = scrub_ohlc(df)
    if _fixed or _dropped:
        # Never silent: a repaired bar is a provider fault worth seeing, and a
        # dropped one is a gap somebody may have to explain later.
        print("   - [CLEAN] %s: %d bar(s) repaired, %d dropped (non-positive "
              "prices)" % (table_name, _fixed, _dropped))
        logger.warning("%s: %d bar(s) repaired, %d dropped (non-positive prices)",
                       table_name, _fixed, _dropped)
    df = _drop_existing_dates(df, table_name)
    if not df.empty:
        df.to_sql(table_name, engine, if_exists='append', index=True)
    return len(df)


def _route_reachable(url):
    """Best-effort reachability probe for the startup banner (one quick try)."""
    try:
        net.http_get(url, route="auto", retries=2, timeout=(5, 8))
        return True
    except Exception:
        return False


def route_status_line():
    """Startup banner: report which data routes actually work right now.

    Yahoo is geo-blocked from a bare RU IP and becomes reachable either via the
    SOCKS5 proxy or a full-tunnel VPN (where the direct route already exits
    abroad). So probe real reachability instead of guessing from the proxy
    state alone, which would falsely warn "Yahoo will fail" while a full-tunnel
    VPN is up and working.
    """
    proxy = ("SOCKS5 proxy up" if (net.SOCKS5_PROXY and net.is_proxy_alive(force=True))
             else "no SOCKS5 proxy")
    yahoo_ok = _route_reachable(
        "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=5d")
    if yahoo_ok:
        return f"  Routes OK: Yahoo reachable, MOEX direct ({proxy})."
    return (f"  Yahoo UNREACHABLE ({proxy}): start your VPN or SOCKS5 proxy, "
            "otherwise Yahoo fetches will fail. MOEX still uses the direct route.")


def _progress_bar(current, total, width=20):
    filled = int(width * current / total) if total else 0
    bar = '#' * filled + '.' * (width - filled)
    return f"[{bar}] {current}/{total}"


# ---------------------------------------------------------------------------
# Thread-safe stdout proxy: worker threads capture print() to thread-local
# buffers instead of writing to real stdout (contextlib.redirect_stdout is
# NOT thread-safe and corrupts sys.stdout when used from multiple threads).
# ---------------------------------------------------------------------------
_tls = threading.local()
_real_stdout = None  # set in main()


class _StdoutProxy:
    """Proxy that routes write() to a thread-local buffer if one is set."""

    def write(self, text):
        buf = getattr(_tls, 'buf', None)
        if buf is not None:
            buf.write(text)
        elif _real_stdout is not None:
            _real_stdout.write(text)

    def flush(self):
        buf = getattr(_tls, 'buf', None)
        if buf is not None:
            buf.flush()
        elif _real_stdout is not None:
            _real_stdout.flush()

    def isatty(self):
        return _real_stdout.isatty() if _real_stdout else False

    def __getattr__(self, name):
        return getattr(_real_stdout, name) if _real_stdout else None


def _fetch_and_save_daily(n, s):
    """Fetch one asset (daily) and save to DB. Thread-safe."""
    table_name = _daily_table(n)
    last_dt = None if BACKFILL else get_last_date(table_name)
    _tls.buf = io.StringIO()
    try:
        df = fetch_moex_smart(s, last_dt) if n in MOEX_TARGETS else fetch_yahoo_smart(n, last_dt)
        raw_out = _tls.buf.getvalue()
    except Exception as e:
        logger.error(f"Fetch error {n}: {e}")
        return n, 'ERR', 0
    finally:
        _tls.buf = None
    bars = 0
    if df is not None:
        with _db_lock:
            bars = _save_df(df, table_name)
        status = 'NEW'
    elif '[ERR]' in raw_out or ('ERR' in raw_out.upper() and 'UP_TO_DATE' not in raw_out):
        status = 'ERR'
    else:
        status = 'UP_TO_DATE'
    return n, status, bars


def _fetch_and_save_weekly(n, s):
    """Fetch one asset (weekly) and save to DB. Thread-safe."""
    table_name = _daily_table(n) + "_weekly"
    last_dt = None if BACKFILL else get_last_date(table_name)
    _tls.buf = io.StringIO()
    try:
        df = fetch_moex_weekly(s, last_dt) if n in MOEX_TARGETS else fetch_yahoo_weekly(n, last_dt)
        raw_out = _tls.buf.getvalue()
    except Exception as e:
        logger.error(f"Weekly fetch error {n}: {e}")
        return n, 'ERR', 0
    finally:
        _tls.buf = None
    bars = 0
    if df is not None:
        with _db_lock:
            bars = _save_df(df, table_name)
        status = 'NEW'
    elif '[ERR]' in raw_out:
        status = 'ERR'
    else:
        status = 'UP_TO_DATE'
    return n, status, bars


def _retry_failed(fetch_fn, sym_for, results, *, sweeps, workers, label):
    """Extra passes over assets with ERR status. Blackholes to iss.moex.com are
    random on each attempt, so a fresh pass usually clears them.
    results (a list of (name, status, bars)) is updated in place."""
    for sweep in range(sweeps):
        failed = [n for n, st, _ in results if st == 'ERR']
        if not failed:
            break
        net.reset_route_cache()  # pick a fresh route on retry
        print(f"  retry {label}: sweep {sweep + 1}/{sweeps}, {len(failed)} assets")
        fixed = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_fn, n, sym_for[n]): n for n in failed}
            for fut in as_completed(futs):
                n, st, b = fut.result()
                fixed[n] = (st, b)
        results[:] = [(n, *fixed[n]) if n in fixed else (n, st, b)
                      for n, st, b in results]
    return results



# Repairing BEFORE the fetch, not after, is the whole point: the weekly repair
# deletes rows and the fetch that follows in this same run is what refills them.
# Run afterwards, the tables would stay short until tomorrow, and a trainer
# started in between would read the hole. Set GTRADE_AUTO_REPAIR=0 to skip.
MARKET_DB = os.path.join(BASE_DIR, "market.db")
AUTO_REPAIR = (os.getenv("GTRADE_AUTO_REPAIR") or "1").strip() not in (
    "0", "false", "False", "no")
REPAIR_PASSES = 8

# A repair rewrites and deletes the very bars a running trainer is reading, and
# a half-repaired history is worse than an unrepaired one. train_chunked touches
# both of these while it works, so a recent stamp means hands off. Skipping a
# repair costs one day; corrupting a 24-hour training run costs the day and the
# champions. Deliberately a file stamp and not a process scan: psutil has not
# been present in every environment this runs in, and a process scan that
# answers wrongly fails in the dangerous direction.
TRAINING_MARKERS = ("_chunk_progress.txt", "_chunk_logs")
# Measured on the 847-asset run of 2026-09-09: the ledger is written once per
# finished chunk and gaps between chunks ran 0.7, 17, 21, 22, 23, 33 and 61
# minutes. The per-chunk log is touched continuously and is what actually keeps
# this fresh, but the window has to cover a stalled writer too, and the two
# errors are not symmetric: a false "busy" costs a deferred repair, a false
# "quiet" costs the training run.
TRAINING_QUIET_MINUTES = 60


def _training_looks_active():
    """(True, minutes) when a chunked training run touched its files recently."""
    newest = None
    for name in TRAINING_MARKERS:
        path = os.path.join(BASE_DIR, name)
        if not os.path.exists(path):
            continue
        stamps = [os.path.getmtime(path)]
        if os.path.isdir(path):
            stamps += [os.path.getmtime(os.path.join(path, f))
                       for f in os.listdir(path)]
        newest = max(stamps) if newest is None else max(newest, max(stamps))
    if newest is None:
        return False, None
    age = (time.time() - newest) / 60.0
    return age < TRAINING_QUIET_MINUTES, age


def _preflight_repair(width):
    """Scan market.db for the two known defects, fix them, and say what it did.

    Both scans are cheap enough to run every time: measured 2026-09-09, 6.8s for
    the daily scan over 850 tables and 0.7s for the weekly one, against a fetch
    that takes minutes. The alternative is what happened before: the defects sat
    in the database for months and training kept reading them.

    See core/features.py and fetch_yahoo_weekly for what each defect is.
    """
    print()
    print('  DATA HEALTH')
    print('  ' + '-' * (width - 2))
    if not AUTO_REPAIR:
        print('  skipped (GTRADE_AUTO_REPAIR=0)')
        return
    active, age = _training_looks_active()
    if active:
        print('  SKIPPED: a training run touched its files %.0f minute(s) ago.'
              % age)
        print('  Repairing now would rewrite bars it is reading. Fetching is')
        print('  safe and continues; run [F] DB Fix once training is done.')
        return
    try:
        import repair_aggregated_bars as agg
        import repair_weekly_tables as wk
    except Exception as exc:                       # never block a data update
        print('  unavailable: %s' % str(exc)[:60])
        return

    try:
        plan = wk.cut_plan(MARKET_DB)
        if plan:
            gone = wk.apply_cuts(MARKET_DB, plan)
            print('  Weekly  : %d table(s) cut, %d row(s) removed - the fetch '
                  'below refills them' % (len(plan), gone))
        else:
            print('  Weekly  : clean')
    except Exception as exc:
        print('  Weekly  : scan failed: %s' % str(exc)[:60])

    try:
        fixed = dropped = 0
        for n in range(REPAIR_PASSES):
            # The filter is relative to each asset's own median, so removing the
            # worst rows exposes the next layer. Four passes were needed on
            # 2026-09-08; it settles when a scan finds nothing left to move.
            f, d, skipped, confirmed = agg.one_pass(MARKET_DB, apply=True, quiet=True)
            fixed += f
            dropped += d
            if not f and not d:
                break
        if fixed or dropped:
            print('  Daily   : %d aggregated row(s) rewritten, %d phantom day(s) '
                  'dropped, in %d pass(es)' % (fixed, dropped, n + 1))
        else:
            print('  Daily   : clean')
        if confirmed or skipped:
            print('            (%d confirmed wide, %d left alone - MOEX or no '
                  'vendor cover)' % (confirmed, skipped))
    except Exception as exc:
        print('  Daily   : scan failed: %s' % str(exc)[:60])


def main():
    global _real_stdout
    _real_stdout = sys.stdout
    sys.stdout = _StdoutProxy()

    t_start = time.time()
    W = 60  # line width

    print()
    print('=' * W)
    print('  ATRATUS DATA ENGINE')
    print(f'  {datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}')
    print('=' * W)
    print(route_status_line())

    _preflight_repair(W)

    assets = list(FULL_ASSET_MAP.items())
    total = len(assets)

    # -- DAILY --------------------------------------------------------------
    print()
    print('  DAILY BARS')
    print('  ' + '-' * (W - 2))

    # Categorize assets into groups (used for summary display)
    from config import ASSET_TYPES
    cat_map = {}
    for cat, members in ASSET_TYPES.items():
        for m in members:
            cat_map[m] = cat

    group_label = {
        'INDICES & MACRO': 'INDICES', 'COMMODITIES': 'COMMODITIES',
        'US TECH': 'US TECH', 'US HEALTHCARE': 'US OTHER', 'US FINANCE': 'US OTHER',
        'US CONSUMER': 'US OTHER', 'US INDUSTRIAL': 'US OTHER',
        'US SEMI': 'US OTHER', 'US SOFTWARE': 'US OTHER',
        'EU INDICES': 'EUROPE', 'EU STOCKS': 'EUROPE',
        'CRYPTO': 'CRYPTO',
        'FOREX MAJORS': 'FOREX', 'FOREX CROSSES': 'FOREX', 'FOREX EXOTIC': 'FOREX',
        'RUS BLUE CHIPS': 'MOEX', 'RUS FINANCE': 'MOEX',
        'RUS TECH': 'MOEX', 'RUS METALS': 'MOEX', 'RUS INFRA': 'MOEX',
        'RUS CONSUMER': 'MOEX', 'RUS PROPERTY': 'MOEX',
    }

    results = []
    _done = [0]
    _done_lock = threading.Lock()
    _is_tty = sys.stdout.isatty()

    def _track_daily(future):
        res = future.result()
        results.append(res)
        name, status, bars = res
        with _done_lock:
            _done[0] += 1
            if _is_tty:
                sys.stdout.write(f"\r  Fetching {_done[0]}/{total}  {name:<12}")
                sys.stdout.flush()
            else:
                # GUI / pipe: one line per asset
                if status == 'NEW':
                    tag = f'+{bars} bars'
                elif status == 'ERR':
                    tag = 'ERROR'
                else:
                    tag = 'up to date'
                print(f"  [{_done[0]:>2}/{total}] {name:<14} {tag}")

    with ThreadPoolExecutor(max_workers=DAILY_WORKERS) as executor:
        futs = {executor.submit(_fetch_and_save_daily, n, s): (n, s)
                for n, s in assets}
        for fut in as_completed(futs):
            _track_daily(fut)
    if _is_tty:
        print()

    _retry_failed(_fetch_and_save_daily, dict(assets), results,
                  sweeps=RETRY_SWEEPS, workers=DAILY_WORKERS, label='daily')

    stats = {
        'ok':       sum(1 for _, st, _ in results if st == 'NEW'),
        'uptodate': sum(1 for _, st, _ in results if st == 'UP_TO_DATE'),
        'err':      sum(1 for _, st, _ in results if st == 'ERR'),
        'new_bars': sum(b for _, _, b in results),
    }

    # Quick summary line (visible in GUI)
    if stats['ok'] == 0 and stats['err'] == 0:
        print(f"  >> All {stats['uptodate']} assets up to date")
    else:
        parts = []
        if stats['ok']:
            parts.append(f"{stats['ok']} updated (+{stats['new_bars']} bars)")
        if stats['uptodate']:
            parts.append(f"{stats['uptodate']} up to date")
        if stats['err']:
            parts.append(f"{stats['err']} errors")
        print(f"  >> {' | '.join(parts)}")

    # Print grouped summary (terminal only - GUI already shows per-asset lines)
    if _is_tty:
        result_map = {n: (st, b) for n, st, b in results}
        group_order = ['INDICES', 'COMMODITIES', 'US TECH', 'US OTHER', 'EUROPE', 'CRYPTO', 'MOEX', 'FOREX', 'OTHER']
        label_map = {
            'INDICES': 'Indices & Macro', 'COMMODITIES': 'Commodities',
            'US TECH': 'US Tech', 'US OTHER': 'US Sectors', 'EUROPE': 'Europe',
            'CRYPTO': 'Crypto',
            'MOEX': 'MOEX (Russia)', 'FOREX': 'Forex', 'OTHER': 'Other',
        }
        for grp in group_order:
            members = [n for n, s in assets
                       if group_label.get(cat_map.get(n, ''), 'OTHER') == grp and n in result_map]
            if not members:
                continue
            print()
            print(f"  {label_map.get(grp, grp)}")
            for name in members:
                status, bars = result_map[name]
                if status == 'NEW':
                    tag = f'+{bars} bars'
                    mark = '[+]'
                elif status == 'ERR':
                    tag = 'ERROR'
                    mark = '[!]'
                else:
                    tag = 'up to date'
                    mark = '   '
                print(f"    {mark} {name:<14} {tag}")

    # -- WEEKLY -------------------------------------------------------------
    weekly_assets = list(assets)  # all assets including MOEX
    total_w = len(weekly_assets)

    print()
    print()
    print('  WEEKLY BARS')
    print('  ' + '-' * (W - 2))

    w_results = []
    _done_w = [0]

    def _track_weekly(future):
        res = future.result()
        w_results.append(res)
        name, status, bars = res
        with _done_lock:
            _done_w[0] += 1
            if _is_tty:
                sys.stdout.write(f"\r  Fetching {_done_w[0]}/{total_w}  {name:<12}")
                sys.stdout.flush()
            else:
                if status == 'NEW':
                    tag = f'+{bars} bars'
                elif status == 'ERR':
                    tag = 'ERROR'
                else:
                    tag = 'up to date'
                print(f"  [{_done_w[0]:>2}/{total_w}] {name:<14} {tag}")

    with ThreadPoolExecutor(max_workers=WEEKLY_WORKERS) as executor:
        futs_w = {executor.submit(_fetch_and_save_weekly, n, s): (n, s)
                  for n, s in weekly_assets}
        for fut in as_completed(futs_w):
            _track_weekly(fut)
    if _is_tty:
        print()

    _retry_failed(_fetch_and_save_weekly, dict(weekly_assets), w_results,
                  sweeps=RETRY_SWEEPS, workers=WEEKLY_WORKERS, label='weekly')

    w_ok   = sum(1 for _, st, _ in w_results if st == 'NEW')
    w_upd  = sum(1 for _, st, _ in w_results if st == 'UP_TO_DATE')
    w_err  = sum(1 for _, st, _ in w_results if st == 'ERR')
    w_bars = sum(b for _, _, b in w_results)

    # Quick weekly summary line
    if w_ok == 0 and w_err == 0:
        print(f"  >> All {w_upd} weekly assets up to date")
    else:
        parts = []
        if w_ok:
            parts.append(f"{w_ok} updated (+{w_bars} bars)")
        if w_upd:
            parts.append(f"{w_upd} up to date")
        if w_err:
            parts.append(f"{w_err} errors")
        print(f"  >> {' | '.join(parts)}")

    # -- SUMMARY ------------------------------------------------------------
    elapsed = time.time() - t_start
    print()
    print('=' * W)
    print('  SUMMARY')
    print('  ' + '-' * (W - 2))
    print(f'  Daily   : {stats["ok"]} updated  |  {stats["uptodate"]} current  |  {stats["err"]} errors  |  +{stats["new_bars"]} bars')
    print(f'  Weekly  : {w_ok} updated  |  {w_upd} current  |  {w_err} errors  |  +{w_bars} bars')
    print(f'  Time    : {elapsed:.1f}s')
    print('=' * W)
    print()

if __name__ == "__main__": main()
