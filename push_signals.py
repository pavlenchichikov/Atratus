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
from core import digest, timing_policy, track_record

BAR_LIMIT = 180   # bars per asset exported for the mobile price chart
SIG_LIMIT = 90    # prediction_log rows per asset for the mobile track record


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


def send_push(url, key, events):
    """Personal FCM push to allow-listed devices. Silent no-op without creds.

    Sends only when something worth acting on changed: at least one signal event
    (flip, entry, exit), and an event set that differs from the last one pushed.
    Timing-only diffs ride along as a counter but never trigger a push on their
    own. Requires GTRADE_FCM_CREDS (path to the Firebase service-account JSON,
    secret, never committed) and the allowed_device_tokens() RPC. Tokens that
    FCM reports as unregistered are deleted (self-cleanup after reinstalls).
    """
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

    if not any(e.kind in digest.SIGNAL_KINDS for e in events):
        print("note: no signal changes this run - staying quiet")
        return 0

    event_hash = _event_hash(events)
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
        return 0

    title, body = build_push_text(events)
    msg = messaging.MulticastMessage(
        notification=messaging.Notification(title=title, body=body),
        data={"screen": "today"},
        tokens=tokens,
    )
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
        _save_push_state(event_hash, max(e.date for e in events))
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
            events = digest.build_digest(hist)
            sent = send_push(url, key, events)
            if sent:
                print(f"FCM push sent to {sent} device(s)")
    except Exception as e:
        print(f"WARNING: FCM push failed: {e}")


if __name__ == "__main__":
    main()
