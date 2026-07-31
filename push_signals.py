"""Push the latest signals snapshot to Supabase for the Atratus landing.

Reads the per-asset latest signal + accuracy from the local prediction journal
(no models loaded) and upserts:
  - `signals`      : per-asset rows (gated behind the allow-list by RLS)
  - `public_stats` : one anonymized aggregate row (anon-readable teaser)

Run it locally after predict.py / reconcile. It only needs the DB and two env
vars, so it can be scheduled (Task Scheduler) once you are happy with it.

    set SUPABASE_URL=...            (Project URL)
    set SUPABASE_SERVICE_KEY=...    (service_role key - keep secret, local only)
    python push_signals.py

If SOCKS5_PROXY is set in .env, all Supabase traffic is routed through it and
every request is retried with backoff - Supabase is EU-hosted and a direct
connection is reset on some networks, which used to abort the bulk history push.
"""

import datetime
import hashlib
import json
import os
import socket
import sys
import time
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from config import FULL_ASSET_MAP
from core import (alerts, digest, events as core_events, news_export,
                  news_link, timing_policy, track_record)
from net import yf_session

BAR_LIMIT = 180   # bars per asset exported for the mobile price chart
SIG_LIMIT = 90    # prediction_log rows per asset for the mobile track record

# The digest must not announce an event the Today screen cannot render: Today
# diffs the same signal_history data, but through api.allHistory(21), which
# becomes a Supabase gte('date', today - 21 days) filter. If we diffed the
# full, unbounded SIG_LIMIT row set here, an asset whose previous row is more
# than 21 days back (newly trained, or a stale data source) would produce an
# event on a baseline the client's window cannot see - the client would then
# see only one row, render "no changes", and the deep-link push would point at
# a screen showing nothing.
DIGEST_WINDOW_DAYS = 21   # must match the client's api.allHistory(21) window


def build_payload(signals: list):
    """Turn track_record.latest_signals() into (signal_rows, stats_row)."""
    rows = []
    n_buy = n_sell = n_wait = 0
    accs = []
    max_date = None
    show_timing = timing_policy.timing_on() and timing_policy.load_policy() is not None

    for s in signals:
        action = (s.get("signal") or "WAIT").upper()
        if action == "BUY":
            n_buy += 1
        elif action == "SELL":
            n_sell += 1
        else:
            n_wait += 1

        acc = (s.get("acc") or {}).get("acc")
        if acc is not None:
            accs.append(acc)

        date = s.get("date")
        if date and (max_date is None or date > max_date):
            max_date = date

        if show_timing:
            t_act = s.get("timing_action")
            t_rsn = s.get("timing_reason")
            _text, _div = timing_policy.display_label(t_act, t_rsn)
            t_label = _text if _div else None
        else:
            t_act = t_rsn = t_label = None

        rows.append({
            "asset": s["asset"],
            "action": action,
            "prob": s.get("probability"),
            "mode": None,
            "taleb": None,
            "accuracy": acc,
            "snapshot_date": date,
            "timing_action": t_act,
            "timing_reason": t_rsn,
            "timing_label": t_label,
        })

    total = len(signals) or 1
    stats = {
        "id": 1,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_wait": n_wait,
        "accuracy": (sum(accs) / len(accs)) if accs else None,
        "breadth": (n_buy + n_sell) / total,
        "regime": None,
        "sentiment": None,
        "snapshot_date": max_date or datetime.date.today().isoformat(),
    }
    return rows, stats


def _rest_base(url: str) -> str:
    """Normalize a Supabase URL to its REST base, tolerating a pasted
    '/rest/v1' suffix so we never end up with '.../rest/v1/rest/v1'."""
    root = url.strip().rstrip("/")
    if root.endswith("/rest/v1"):
        root = root[: -len("/rest/v1")]
    return root + "/rest/v1"


CHUNK = 300  # rows per PostgREST upsert; smaller bodies survive a shaky tunnel

# Errors worth retrying: a VPN tunnel drops the connection mid-transfer
# (ConnectionReset / SSL EOF / timeout). A non-2xx HTTP status is NOT retried.
_TRANSIENT = (
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
)


_proxy_resolved = False
_proxy_value = None


def _alive(host, port, timeout=1.5):
    """True if something is listening on host:port (a quick TCP probe)."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def _proxies():
    """Route Supabase traffic through SOCKS5_PROXY when it is actually up
    (Supabase is EU and a direct connection is reset on some networks). If the
    proxy is unset or not listening, connect directly - so a system-wide VPN
    keeps working without a local proxy. Resolved once per run (like net.py)."""
    global _proxy_resolved, _proxy_value
    if _proxy_resolved:
        return _proxy_value
    _proxy_resolved = True
    p = os.getenv("SOCKS5_PROXY")
    if not p:
        return None
    parsed = urlparse(p)
    if parsed.hostname and parsed.port and _alive(parsed.hostname, parsed.port):
        _proxy_value = {"http": p, "https": p}
    else:
        print(f"note: SOCKS5_PROXY {parsed.hostname}:{parsed.port} not reachable "
              "- connecting directly")
    return _proxy_value


def _send(method, url, *, retries=5, **kw):
    """One Supabase request via the optional SOCKS5 proxy, retried with backoff
    on transient network errors so a single reset does not abort a bulk push.
    Raises on a non-2xx status or once retries are exhausted."""
    kw.setdefault("proxies", _proxies())
    delay = 1.5
    for attempt in range(retries):
        try:
            r = requests.request(method, url, **kw)
            r.raise_for_status()
            return r
        except _TRANSIENT:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20)
    raise RuntimeError("unreachable")


def _chunked(seq, n=CHUNK):
    """Yield n-sized slices so PostgREST request bodies stay small."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_history_rows(db_path=None, bar_limit=BAR_LIMIT, sig_limit=SIG_LIMIT):
    """Per-asset OHLC bars and prediction_log excerpts for the mobile app.

    Reads market.db directly (price tables are named asset.lower() and store
    columns Date/open/high/low/close). Assets without a price table yet (new,
    untrained) are skipped silently.
    """
    import sqlite3
    db = db_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "market.db")
    bars, hist = [], []
    show_timing = (timing_policy.timing_on()
                   and timing_policy.load_policy() is not None)
    con = sqlite3.connect(db)
    try:
        cur = con.cursor()
        for asset in FULL_ASSET_MAP:
            try:
                rows = cur.execute(
                    f'SELECT Date, open, high, low, close FROM "{asset.lower()}" '
                    "ORDER BY Date DESC LIMIT ?", (bar_limit,)).fetchall()
            except sqlite3.OperationalError:
                continue
            for d, o, h, lo, c in reversed(rows):
                bars.append({"asset": asset, "date": str(d)[:10], "open": o,
                             "high": h, "low": lo, "close": c})
            try:
                sig = cur.execute(
                    "SELECT date, signal, probability, actual_next_ret, correct, "
                    "timing_action, timing_reason FROM prediction_log "
                    "WHERE asset = ? ORDER BY date DESC LIMIT ?",
                    (asset, sig_limit)).fetchall()
            except sqlite3.OperationalError:
                sig = []
            for d, s, p, r, cr, t_act, t_rsn in reversed(sig):
                if show_timing and t_act is not None:
                    _text, _div = timing_policy.display_label(t_act, t_rsn)
                    t_label = _text if _div else None
                else:
                    t_act = t_label = None
                hist.append({"asset": asset, "date": str(d)[:10], "signal": s,
                             "prob": p, "actual_next_ret": r, "correct": cr,
                             "timing_action": t_act, "timing_label": t_label})
    finally:
        con.close()
    return bars, hist


def _acc_number(val):
    """Best-effort (accuracy, n) out of guru_accuracy()'s council payload."""
    if isinstance(val, dict):
        acc = val.get("acc")
        if not isinstance(acc, (int, float)):
            acc = val.get("accuracy")
        n = None
        for key in ("n", "count", "total"):
            candidate = val.get(key)
            if isinstance(candidate, int):
                n = candidate
                break
        return (acc if isinstance(acc, (int, float)) else None,
                n if isinstance(n, int) else None)
    if isinstance(val, (int, float)):
        return val, None
    return None, None


def fetch_guru_rows():
    """Latest Guru Council verdicts + one accuracy summary row (or None)."""
    from core import dashboard
    rows = [{"asset": g["asset"], "verdict": g["verdict"],
             "council_pct": g["pct"], "lynch": g["lynch"],
             "buffett": g["buffett"], "graham": g["graham"],
             "munger": g["munger"], "source": g["source"],
             "date": str(g["date"])[:10] if g.get("date") else None,
             "correct_5d": g["correct_5d"]}
            for g in dashboard.guru_latest()]
    acc, n = _acc_number(dashboard.guru_accuracy().get("council"))
    stats = ({"id": 1, "accuracy": acc, "n": n, "horizon": "60d"}
             if acc is not None else None)
    return rows, stats


def push_history(url, key, bars, hist, guru_rows, guru_stats):
    """Full-refresh upsert of the four mobile snapshot tables."""
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    merge = {**headers, "Prefer": "resolution=merge-duplicates"}
    for table, rows in (("bars", bars), ("signal_history", hist),
                        ("guru", guru_rows)):
        _send("DELETE", f"{base}/{table}?asset=not.is.null",
              headers=headers, timeout=60)
        for chunk in _chunked(rows):
            _send("POST", f"{base}/{table}", headers=merge, json=chunk,
                  timeout=120)
    if guru_stats:
        _send("POST", f"{base}/guru_stats?on_conflict=id", headers=merge,
              json=guru_stats, timeout=30)


def build_push_text(events):
    """Notification title/body from digest events.

    The title counts every event so it reconciles with the body counters. The
    body names up to 4 signal events (flip, entry, exit), then reports how many
    signal events were left out, then how many timing-only moves happened.
    ASCII only, no arrows - repo convention.
    """
    signal_events = [e for e in events if e.kind in digest.SIGNAL_KINDS]
    n_timing = len(events) - len(signal_events)

    named = []
    for e in signal_events[:4]:
        if e.kind == digest.FLIP:
            named.append(f"{e.asset} {e.from_signal}>{e.to_signal}")
        elif e.kind == digest.EXIT:
            named.append(f"{e.asset} exit")
        elif e.confidence is None:
            named.append(f"{e.asset} {e.to_signal}")
        else:
            named.append(f"{e.asset} {e.to_signal} {e.confidence * 100:.0f}%")

    parts = list(named)
    left_out = len(signal_events) - len(named)
    if left_out > 0:
        parts.append(f"+{left_out} more")
    if n_timing:
        parts.append(f"+{n_timing} timing")

    n = len(events)
    title = f"Atratus: {n} change" if n == 1 else f"Atratus: {n} changes"
    return title, " | ".join(parts)


PUSH_STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "push_state.json")


def _event_hash(events):
    """Stable fingerprint of an event set.

    The date is part of the key on purpose: without it an identical event set on
    a later day would hash equal and be silently suppressed. Timing labels are
    included so a changed timing counter still reads as new.
    """
    canon = "\n".join(sorted(
        f"{e.date}|{e.asset}|{e.kind}|{e.from_signal}|{e.to_signal}|"
        f"{e.from_timing}|{e.to_timing}" for e in events))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _load_push_state(path=None):
    """Last pushed fingerprint, or {} when there is none we can read."""
    try:
        with open(path or PUSH_STATE, encoding="utf-8") as fh:
            state = json.load(fh)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _save_push_state(event_hash, snapshot_date, path=None):
    """Record what was just sent. A failure here costs one duplicate push."""
    target = path or PUSH_STATE
    try:
        with open(target, "w", encoding="utf-8") as fh:
            json.dump({"hash": event_hash, "snapshot_date": snapshot_date,
                       "sent_at": datetime.datetime.now().isoformat(
                           timespec="seconds")}, fh)
    except Exception as e:
        print(f"note: could not write {target}: {e}")


ALERT_DDL = """CREATE TABLE IF NOT EXISTS alert_log (
    sent_at     TEXT,
    date        TEXT,
    asset       TEXT,
    kind        TEXT,
    from_signal TEXT,
    to_signal   TEXT,
    confidence  REAL,
    push_hash   TEXT,
    pushed      INTEGER)"""


def _alert_db(db_path=None):
    return db_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "market.db")


def log_alerts(events, push_hash, db_path=None):
    """Record a candidate set once, keyed on its fingerprint. Returns the rows
    written.

    Keyed rather than per-run because send_push returns early when
    GTRADE_FCM_CREDS is unset, BEFORE the duplicate check, so a per-run write
    would store the same unchanged set every run and fill the sample with
    duplicates.

    Written before delivery is attempted and with pushed = 0, so a crash inside
    the FCM path still leaves the candidates recorded; mark_pushed states the
    outcome afterwards.
    """
    if not events:
        return 0
    import sqlite3
    con = sqlite3.connect(_alert_db(db_path))
    try:
        con.execute(ALERT_DDL)
        seen = con.execute("SELECT 1 FROM alert_log WHERE push_hash=? LIMIT 1",
                           (push_hash,)).fetchone()
        if seen:
            return 0
        sent_at = datetime.datetime.now().isoformat(timespec="seconds")
        rows = alerts.alert_rows(events, push_hash, False, sent_at)
        con.executemany(
            "INSERT INTO alert_log (sent_at, date, asset, kind, from_signal,"
            " to_signal, confidence, push_hash, pushed)"
            " VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        return len(rows)
    finally:
        con.close()


def mark_pushed(push_hash, db_path=None):
    """Flip a logged set to delivered. Returns the rows updated."""
    import sqlite3
    con = sqlite3.connect(_alert_db(db_path))
    try:
        con.execute(ALERT_DDL)
        cur = con.execute("UPDATE alert_log SET pushed=1 WHERE push_hash=?",
                          (push_hash,))
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def send_push(url, key, events):
    """Personal FCM push to allow-listed devices. Silent no-op without creds.

    Sends only when something worth acting on changed: at least one signal event
    (flip, entry, exit), and an event set that differs from the last one pushed.
    Timing-only diffs ride along as a counter but never trigger a push on their
    own. Requires GTRADE_FCM_CREDS (path to the Firebase service-account JSON,
    secret, never committed) and the allowed_device_tokens() RPC. Tokens that
    FCM reports as unregistered are deleted (self-cleanup after reinstalls).

    With GTRADE_ALERT_FILTER=1 the message carries the candidate set as data and
    NO notification block, so the device decides what to display against its own
    positions journal. The flag exists because a client without a background
    handler shows nothing at all for a data-only message: the format may only
    change once the rebuilt app is installed.
    """
    if not any(e.kind in digest.SIGNAL_KINDS for e in events):
        print("note: no signal changes this run - staying quiet")
        return 0

    # Logged before anything can return early, so the research sample
    # accumulates even while FCM is unconfigured. Guarded because the log is a
    # research convenience and the push is the product: a locked or unwritable
    # market.db must cost a log row, never the day's notification.
    event_hash = _event_hash(events)
    try:
        log_alerts(events, event_hash)
    except Exception as e:
        print(f"note: could not record the alert candidates: {e}")

    creds_path = os.getenv("GTRADE_FCM_CREDS")
    if not creds_path:
        return 0
    if not os.path.exists(creds_path):
        # Configured but the service-account JSON is not there yet: treat it as
        # "push not set up" and skip quietly, rather than raising every run.
        print(f"note: GTRADE_FCM_CREDS points to {creds_path}, which does not "
              "exist - skipping FCM push (add the Firebase service-account "
              "JSON there to enable notifications)")
        return 0

    if _load_push_state().get("hash") == event_hash:
        print("note: same changes as the last push - staying quiet")
        return 0

    import firebase_admin
    from firebase_admin import credentials, messaging
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(creds_path))

    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    r = _send("POST", f"{base}/rpc/allowed_device_tokens", headers=headers,
              json={}, timeout=30)
    tokens = [row["token"] for row in r.json()]
    if not tokens:
        print("note: no allow-listed device tokens - staying quiet")
        return 0

    notification = None
    android = None
    data = {"screen": "today"}
    payload = alerts.encode_events(events)[0]
    if (os.getenv("GTRADE_ALERT_FILTER") == "1"
            and len(payload) <= alerts.PAYLOAD_CAP):
        # Data-only: a notification block would be rendered by Android's tray
        # while the app is backgrounded, and the device filter could not
        # suppress it.
        data["alerts"] = payload
        # FCM defaults a data-only message to normal priority, which Doze
        # defers - the device would wake to yesterday's alert. A notification
        # message does not need this because the tray shows it directly.
        android = messaging.AndroidConfig(priority="high")
    else:
        # Either the flag is off, or the candidate list does not fit. Falling
        # back to the composed notification is honest degradation; truncating
        # the list could drop precisely the held asset the filter exists for.
        title, body = build_push_text(events)
        notification = messaging.Notification(title=title, body=body)

    msg = messaging.MulticastMessage(notification=notification, data=data,
                                     tokens=tokens, android=android)
    resp = messaging.send_each_for_multicast(msg)
    for i, res in enumerate(resp.responses):
        if res.success:
            continue
        name = res.exception.__class__.__name__ if res.exception else ""
        if name in ("UnregisteredError", "SenderIdMismatchError"):
            _send("DELETE", f"{base}/device_tokens?token=eq.{tokens[i]}",
                  headers=headers, timeout=30)
    if resp.success_count:
        # Only a delivered push updates the fingerprint, so a total failure is
        # retried on the next run instead of being deduped away.
        #
        # mark_pushed FIRST, and the fingerprint saved only if it succeeded.
        # Saving regardless would dedupe this set away on every later run while
        # its rows still read pushed=0, recording a delivered push as
        # undelivered forever. Skipping the save costs one duplicate
        # notification tomorrow, which is visible and self-correcting; a
        # silently wrong research sample is neither.
        try:
            mark_pushed(event_hash)
            _save_push_state(event_hash, max(e.date for e in events))
        except Exception as e:
            print(f"note: delivered, but the alert log could not be marked, so "
                  f"the fingerprint is not recorded and the next run will send "
                  f"this again: {e}")
    else:
        print(f"WARNING: FCM push delivered to 0 of {len(tokens)} device(s) - "
              "not recording the fingerprint, will retry next run")
    return resp.success_count


def push(rows, stats, url: str, key: str):
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    merge = {**headers, "Prefer": "resolution=merge-duplicates"}
    # Full snapshot: clear the table, then insert fresh (drops delisted assets).
    _send("DELETE", f"{base}/signals?asset=not.is.null", headers=headers, timeout=30)
    for chunk in _chunked(rows):
        _send("POST", f"{base}/signals", headers=merge, json=chunk, timeout=120)
    _send("POST", f"{base}/public_stats?on_conflict=id", headers=merge,
          json=stats, timeout=30)


def _digest_rows(hist, today=None):
    """The subset of hist rows the digest is allowed to diff: the trailing
    DIGEST_WINDOW_DAYS window, matching what the client's Today screen fetches.

    This does NOT touch what gets pushed to signal_history - push_history keeps
    shipping the full row set unfiltered. Only the rows fed to the digest are
    bounded, so an asset whose baseline falls outside the window produces no
    event here, which is correct: Today would show nothing for it either.
    """
    day = today or datetime.date.today()
    floor = (day - datetime.timedelta(days=DIGEST_WINDOW_DAYS)).isoformat()
    return [row for row in hist if row["date"] >= floor]


def fetch_news_rows(hist, today=None):
    """General digest plus per-asset news for today's actionable calls.

    Returns (rows, items_by_asset). The raw items come back too because the
    news context needs the same ones, and re-fetching would lean on
    news_analyzer's 30-minute cache still being warm.

    Every fetch is guarded on its own: news is an extra, and one dead feed must
    not cost the rest of the export.
    """
    import news_analyzer

    day = (today or datetime.date.today()).isoformat()
    rows = []
    try:
        items = news_analyzer.fetch_authority_digest(max_per_source=5,
                                                     fetch_summaries=False)
    except Exception as e:
        print(f"note: general news unavailable: {e}")
        items = []
    rows.extend(news_export.general_rows(items, day))
    by_asset = {}
    for asset in news_export.pick_assets(hist, day):
        try:
            found = news_analyzer.fetch_news(
                asset, max_articles=news_export.PER_ASSET,
                fetch_summaries=False)
        except Exception as e:
            print(f"note: news unavailable for {asset}: {e}")
            continue
        by_asset[asset] = found
        rows.extend(news_export.asset_rows(asset, found, day))
    return rows, by_asset


def fetch_news_context(bars, items_by_asset):
    """One news_context row per asset that has bars.

    Assets whose news was never fetched still get a row: the move and its
    notability are worth reporting on their own, but their news verdict is
    recorded as unchecked (news_link.context_row's "not_checked") rather than
    as an absence, since no fetch was ever made to find out.
    """
    by_asset = {}
    for bar in bars:
        by_asset.setdefault(bar["asset"], []).append(bar)
    rows = []
    for asset, asset_bars in by_asset.items():
        row = news_link.context_row(asset, asset_bars,
                                    items_by_asset.get(asset))
        if row is not None:
            rows.append(row)
    return rows


def push_news_context(url, key, rows):
    """Full-refresh upsert of the news_context table.

    Deletes on asset, which is this table's primary key and never null. The
    news table differs because its asset is nullable.
    """
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    merge = {**headers, "Prefer": "resolution=merge-duplicates"}
    _send("DELETE", f"{base}/news_context?asset=not.is.null", headers=headers,
          timeout=60)
    for chunk in _chunked(rows):
        _send("POST", f"{base}/news_context?on_conflict=asset", headers=merge,
              json=chunk, timeout=120)


def push_news(url, key, rows):
    """Full-refresh upsert of the news table.

    Deletes on id, not asset: general-feed rows carry a null asset, and an
    asset-based filter would leave them behind to accumulate.
    """
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    merge = {**headers, "Prefer": "resolution=merge-duplicates"}
    _send("DELETE", f"{base}/news?id=not.is.null", headers=headers, timeout=60)
    for chunk in _chunked(rows):
        _send("POST", f"{base}/news?on_conflict=id", headers=merge, json=chunk,
              timeout=120)


def fetch_event_rows(assets, today=None):
    """Upcoming earnings for `assets` plus the macro calendar, within horizon.

    The two sources are independent on purpose: a dead proxy costs the earnings
    half and leaves the macro half intact.
    """
    day = today or datetime.date.today()
    symbols = {a: FULL_ASSET_MAP[a] for a in assets if a in FULL_ASSET_MAP}
    earnings = {}
    try:
        earnings = core_events.earnings_for(symbols, session=yf_session())
    except Exception as e:
        print(f"note: earnings calendar unavailable: {e}")
    macro = core_events.load_macro()
    rows = core_events.event_rows(earnings, macro)
    return core_events.upcoming(rows, day)


def push_events(url, key, rows):
    """Full-refresh upsert of the events table.

    Deletes on id: macro rows carry a null asset, so an asset filter would
    orphan them.
    """
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    merge = {**headers, "Prefer": "resolution=merge-duplicates"}
    _send("DELETE", f"{base}/events?id=not.is.null", headers=headers,
          timeout=60)
    for chunk in _chunked(rows):
        _send("POST", f"{base}/events?on_conflict=id", headers=merge,
              json=chunk, timeout=120)


def main():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (see .env.example).")

    signals = track_record.latest_signals()
    rows, stats = build_payload(signals)
    push(rows, stats, url, key)
    print(f"pushed {len(rows)} signals | "
          f"buy/sell/wait={stats['n_buy']}/{stats['n_sell']}/{stats['n_wait']} | "
          f"as_of={stats['snapshot_date']}")

    # Optional extras for the mobile app - each fail-safe: a failure here must
    # never undo or block the signals upsert above.
    bars = None
    hist = None
    history_ok = False
    try:
        bars, hist = fetch_history_rows()
        guru_rows, guru_stats = fetch_guru_rows()
        push_history(url, key, bars, hist, guru_rows, guru_stats)
        history_ok = True
        print(f"pushed history: {len(bars)} bars | {len(hist)} signal rows | "
              f"{len(guru_rows)} guru verdicts")
    except Exception as e:
        print(f"WARNING: history push failed: {e}")
    try:
        if not history_ok:
            # Nothing to diff, and the phone could not show the events a
            # notification would announce. No fallback text: a push that cannot
            # say what changed has no reason to exist.
            print("note: history unavailable - skipping the change push")
        else:
            events = digest.build_digest(_digest_rows(hist))
            sent = send_push(url, key, events)
            if sent:
                print(f"FCM push sent to {sent} device(s)")
    except Exception as e:
        print(f"WARNING: FCM push failed: {e}")

    if os.getenv("GTRADE_NEWS_EXPORT") == "1":
        if not history_ok:
            print("note: history unavailable - skipping the news export")
        else:
            # Picked once, independent of the RSS fetch below: stage C (earnings)
            # must not lose assets just because stage A's fetch for some of them
            # failed or stage A raised outright.
            day = datetime.date.today()
            picked_assets = news_export.pick_assets(hist, day.isoformat())

            news_items = None
            try:
                news_rows, news_items = fetch_news_rows(hist, today=day)
                push_news(url, key, news_rows)
                print(f"pushed news: {len(news_rows)} rows")
            except Exception as e:
                print(f"WARNING: news push failed: {e}")

            try:
                if news_items is None:
                    print("note: news items unavailable - skipping the news "
                          "context export")
                else:
                    ctx = fetch_news_context(bars, news_items)
                    push_news_context(url, key, ctx)
                    print(f"pushed news context: {len(ctx)} rows")
            except Exception as e:
                print(f"WARNING: news context push failed: {e}")

            try:
                ev_rows = fetch_event_rows(picked_assets, today=day)
                push_events(url, key, ev_rows)
                print(f"pushed events: {len(ev_rows)} rows")
            except Exception as e:
                print(f"WARNING: events push failed: {e}")


if __name__ == "__main__":
    main()
