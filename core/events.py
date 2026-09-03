"""Known-date event risk: company earnings and macro releases.

Two independent sources. Earnings come from yfinance, which reports ESTIMATED
dates alongside confirmed ones - an estimate carries confirmed=False and must
never be rendered as fact. Macro dates come from a hand-maintained file,
because FOMC and CPI schedules are published a year ahead and an API for them
is not worth a dependency.

Nothing here raises. A calendar is an extra, and a dead one must not take the
export down with it.
"""

import datetime
import hashlib
import json
import os

HORIZON_DAYS = 14

MACRO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "macro_calendar.json")


def _as_date(value):
    """A date from an ISO string, date, datetime or pandas Timestamp. Else None."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def days_until(date, today):
    """Whole days from `today` to `date`. Negative once it has passed."""
    parsed = _as_date(date)
    return None if parsed is None else (parsed - today).days


def event_id(kind, asset, date, name):
    """Deterministic row id, so a re-run upserts instead of duplicating."""
    key = "{}|{}|{}|{}".format(kind, asset or "", date, name)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _next_earnings(calendar):
    """{"date", "confirmed"} from a yfinance calendar dict, or None.

    yfinance reports a confirmed earnings date as a single date and an estimate
    as a RANGE of two. The list length is the only confirmation signal it
    offers, so that is what confirmed is read from.
    """
    dates = (calendar or {}).get("Earnings Date") or []
    if isinstance(dates, (str, datetime.date, datetime.datetime)):
        dates = [dates]
    try:
        candidates = list(dates)
    except TypeError:
        return None
    parsed = [d for d in (_as_date(x) for x in candidates) if d is not None]
    if not parsed:
        return None
    # Confirmation is the SHAPE yfinance returned, not how many of its entries
    # parsed. A two-element range with one unparseable bound is still an
    # estimate, and reporting it as confirmed would be the exact lie this
    # field exists to prevent.
    return {"date": min(parsed).isoformat(), "confirmed": len(candidates) == 1}


def _yf_calendar(symbol, session):
    import yfinance as yf

    return yf.Ticker(symbol, session=session).calendar


# Yahoo symbol shapes that cannot report earnings: an index, a crypto pair, an
# FX pair, a futures contract. Asking anyway costs one 404 and one round trip
# each, which on the full map was 147 needless requests and a wall of yfinance
# error logging that made a healthy run look broken.
_NO_EARNINGS_MARKS = ("^", "=X", "=F", "-USD")
_NO_EARNINGS_SYMBOLS = frozenset({"DX-Y.NYB"})   # the dollar index, dot and all


def _is_moex(asset):
    """True for a Moscow-listed name. Never raises, so a config problem here
    only costs one needless lookup rather than the whole scan."""
    try:
        from config import radar_category

        return radar_category(asset) == "ru"
    except Exception:
        return False


def can_have_earnings(symbol, asset=None):
    """False for a symbol whose earnings this source cannot serve.

    Two separate reasons, and the second needs the asset name rather than the
    symbol. An index, an FX pair, a futures contract and a crypto pair are not
    companies and never report. A Moscow-listed name is a company and does
    report, but its ticker is carried here bare (VTBR, SBER), which Yahoo does
    not resolve at all, so asking buys one 404 per asset and nothing else. The
    fundamentals for those come from Smart-Lab elsewhere and are unaffected.
    """
    if not symbol or symbol in _NO_EARNINGS_SYMBOLS:
        return False
    if any(m in symbol for m in _NO_EARNINGS_MARKS):
        return False
    return not (asset is not None and _is_moex(asset))


def earnings_for(symbols_by_asset, session=None, fetch=None):
    """{asset: {"date": iso, "confirmed": bool}} for assets that have a date.

    `symbols_by_asset` maps the internal asset name to its Yahoo symbol, so
    core does not need the asset map. `fetch(symbol, session)` is injectable so
    tests never reach the network.

    Assets with no earnings at all (crypto, forex, indices) are simply absent.
    A per-asset failure is skipped rather than raised.
    """
    getter = fetch or _yf_calendar
    out = {}
    for asset, symbol in symbols_by_asset.items():
        if not can_have_earnings(symbol, asset):
            continue
        try:
            found = _next_earnings(getter(symbol, session))
        except Exception:
            continue
        if found is not None:
            out[asset] = found
    return out


def load_macro(path=None):
    """Validated macro events from the calendar file.

    A missing file, invalid JSON, a non-list payload, or an entry without a
    parseable date and a non-empty name all read as no events.
    """
    try:
        with open(path or MACRO_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    out = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        date = _as_date(entry.get("date"))
        name = (entry.get("name") or "").strip()
        if date is None or not name:
            continue
        out.append({
            "kind": "macro",
            "asset": None,
            "date": date.isoformat(),
            "name": name,
            "importance": entry.get("importance") or None,
            # Carried through, not dropped. The validator used to keep only
            # name/date/importance, so a consumer had nothing to filter on and
            # every Moscow name was handed the FOMC calendar while every US
            # name got the Bank of Russia's. Absent reads as global.
            "region": (str(entry.get("region") or "").strip().upper() or None),
            # Macro dates are published schedules, not estimates.
            "confirmed": True,
        })
    return out


def upcoming(events, today, horizon_days=HORIZON_DAYS):
    """Events from today through the horizon, soonest first.

    Past events are dropped: an event risk flag is only useful before the fact.
    """
    out = []
    for event in events:
        delta = days_until(event.get("date"), today)
        if delta is not None and 0 <= delta <= horizon_days:
            out.append(event)
    out.sort(key=lambda e: (e["date"], e.get("name") or ""))
    return out


def event_rows(earnings_by_asset, macro_events):
    """Supabase rows for both sources, each with a deterministic id."""
    rows = []
    for asset, info in sorted(earnings_by_asset.items()):
        name = "Earnings"
        rows.append({
            "id": event_id("earnings", asset, info["date"], name),
            "kind": "earnings",
            "asset": asset,
            "date": info["date"],
            "name": name,
            "importance": None,
            "confirmed": info["confirmed"],
        })
    for event in macro_events:
        rows.append({
            "id": event_id("macro", None, event["date"], event["name"]),
            "kind": "macro",
            "asset": None,
            "date": event["date"],
            "name": event["name"],
            "importance": event.get("importance"),
            "confirmed": True,
        })
    return rows
