"""The analyst's intraday question: the next session, asked once, scored per claim.

One call per asset answers four things about the session that follows the
dossier's last bar, and each answer has its own baseline, because each is a
different claim:

    direction   open -> close of the session            vs a coin
    vol_regime  the session's high-low range, read as    vs HAR (core.levels), the
                narrow / normal / wide                      volatility model the
                                                            card already shows
    stand_aside whether the session should be left alone vs "stand aside when the
                                                           calendar has an event that day"
    gap         the open against the last close          information only

Why the range carries the weight: every intraday measurement in this project
put direction at IC 0 and volatility at IC 0.67, and HAR forecasts tomorrow's
range at IC 0.41 from the bars alone. "Always normal" was a baseline a model
that only copies HAR would beat, so the analyst now has to beat HAR itself, and
sees HAR's number in the dossier (session_range).

The gap is no longer a scored claim. The US-close lead it was built on is an
asynchronous-close artifact: futures and US-listed ETFs on the same markets
price the US move the same day (next-day IC -0.05..-0.09, 2026-09-28), and the
gap happens before anyone can trade at the last close. It is still asked and
shown, because where a session opens matters to whoever trades it.

Stored in its own table, analyst_intraday_log, so the daily score and the
daily calibration never see a session row.
"""

import datetime as _dt
import itertools
import json
import os
import sqlite3
import statistics

from core.analyst import dossier, store

ohlc_series = store.bars

# Without GTRADE_ANALYST_INTRADAY_PANEL the panel is the watchlist's default
# list, the names the owner actually trades; this is only the fallback.
PANEL = ("SBER", "IMOEX", "SP500", "NVDA", "GOLD", "BTC", "ETH")
LEADER = "SP500"
WATCHLIST = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "watchlist.json")
# Rewound runs start here: a model that read the news of a date can "foresee"
# that date's session. Claude Opus 5.5 knows up to June 2026; set
# GTRADE_ANALYST_REWIND_FROM to the cutoff of the model you replay with.
REWIND_FROM = "2026-07-01"

# Sessions the range classes and the surprise ratio are read from, and the
# sample every question needs before its verdict can say SHIP.
HISTORY_SESSIONS = 60
SHIP_FLOOR = 100
MIN_STAND_ASIDE = 20

DDL = """
CREATE TABLE IF NOT EXISTS analyst_intraday_log (
    date TEXT, asset TEXT,
    direction TEXT, gap TEXT, conviction INTEGER, vol_regime TEXT,
    stand_aside INTEGER, stand_aside_reason TEXT,
    key_risk TEXT, thesis TEXT, evidence_json TEXT,
    dossier_hash TEXT, llm_model TEXT,
    atr_at_signal REAL, close_at_signal REAL, us_last_session_ret REAL,
    calendar_json TEXT, judged_at TEXT,
    session_date TEXT, session_open REAL, session_high REAL,
    session_low REAL, session_close REAL, session_plan TEXT,
    PRIMARY KEY (date, asset)
)
"""
_FIELDS = ("date", "asset", "direction", "gap", "conviction", "vol_regime",
           "stand_aside", "stand_aside_reason", "key_risk", "thesis",
           "evidence_json", "dossier_hash", "llm_model", "atr_at_signal",
           "close_at_signal", "us_last_session_ret", "calendar_json", "judged_at",
           "session_plan")


def panel_assets():
    raw = (os.getenv("GTRADE_ANALYST_INTRADAY_PANEL") or "").strip()
    if raw:
        return [a.strip().upper() for a in raw.split(",") if a.strip()]
    try:
        with open(WATCHLIST, encoding="utf-8") as fh:
            names = json.load(fh).get("default") or []
    except (OSError, ValueError, AttributeError):
        names = []
    return [str(a).upper() for a in names] or list(PANEL)


def rewind_from():
    return (os.getenv("GTRADE_ANALYST_REWIND_FROM") or REWIND_FROM).strip()


def _connect(db_path=None):
    # store.DB_PATH read at call time, so a test that points the store at a
    # temporary database points this table there too.
    con = sqlite3.connect(db_path or store.DB_PATH)
    con.execute(DDL)
    try:                                   # tables made before session_plan
        con.execute("ALTER TABLE analyst_intraday_log ADD COLUMN session_plan TEXT")
    except sqlite3.OperationalError:
        pass
    return con


def us_last_session_ret(date, db_path=None):
    """SP500's close-to-close return on the bar dated `date`, or None.

    Only that date: a later SP500 bar would be the future for an Asian session
    judged the evening before, and an earlier one is a stale lead.
    """
    bars = ohlc_series(LEADER, days=60, db_path=db_path)
    for prev, bar in itertools.pairwise(bars):
        if bar["date"] == date and prev["close"]:
            return (bar["close"] - prev["close"]) / prev["close"]
    return None


def session_range_view(bars):
    """What the range question is judged against, from the bars up to the
    dossier's date (oldest first), in percent of price; None under 60 sessions.

    HAR forecasts the TRUE range, gap included; the question is the session's
    own high-low. Both are read over the same 60 sessions and HAR is scaled by
    the ratio of their medians, so its class comes from the same cut-offs the
    score uses for the session that follows.
    """
    from core.levels import _true_ranges, range_forecast

    hist = _ranges(bars)
    if len(hist) < HISTORY_SESSIONS:
        return None
    f = range_forecast(bars)
    tr = [t / b["close"] for t, b in zip(_true_ranges(bars), bars) if b["close"]]
    tr = tr[-HISTORY_SESSIONS:]
    if f is None or not tr or not statistics.median(tr):
        return None
    scale = statistics.median(hist) / statistics.median(tr)
    typical, wide = f["typical"] * scale, f["q90"] * scale
    q = statistics.quantiles(hist, n=3)

    def pct(x):
        return round(100 * x, 3)

    return {"har_typical_pct": pct(typical), "har_wide_pct": pct(wide),
            "har_class": _range_class(typical, hist),
            "usual_pct": pct(statistics.median(hist)),
            "calm_below_pct": pct(q[0]), "elevated_above_pct": pct(q[1]),
            "last_session_pct": pct(hist[-1])}


def build(asset, db_path=None, today=None):
    """The daily dossier plus the fields the intraday question needs."""
    d = dossier.build(asset, db_path=db_path, today=today)
    d["us_last_session_ret"] = (us_last_session_ret(d["date"], db_path)
                                if d.get("date") else None)
    bars = ([b for b in ohlc_series(asset, days=400, db_path=db_path)
             if b["date"] <= d["date"]] if d.get("date") else [])
    d["session_range"] = session_range_view(bars)
    return d


def write(row, db_path=None):
    with _connect(db_path) as con:
        con.execute("INSERT OR REPLACE INTO analyst_intraday_log (%s) VALUES (%s)"
                    % (",".join(_FIELDS), ",".join("?" * len(_FIELDS))),
                    [row.get(f) for f in _FIELDS])


def row_for(d, j, dossier_hash, llm_model):
    """The stored row for dossier `d` and parsed session judgment `j`."""
    calendar = {"macro_events": d.get("macro_events") or [],
                "next_earnings": d.get("next_earnings")}
    return {"date": d["date"], "asset": d["asset"],
            "direction": j["direction"], "gap": j["gap"],
            "conviction": j["conviction"], "vol_regime": j["vol_regime"],
            "stand_aside": int(j["stand_aside"]),
            "stand_aside_reason": j.get("stand_aside_reason"),
            "key_risk": j["key_risk"], "thesis": j["thesis"],
            "evidence_json": json.dumps(j["evidence"]),
            "dossier_hash": dossier_hash, "llm_model": llm_model,
            "atr_at_signal": d.get("atr"), "close_at_signal": d.get("close"),
            "us_last_session_ret": d.get("us_last_session_ret"),
            "calendar_json": json.dumps(calendar, ensure_ascii=False, default=str),
            "judged_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "session_plan": j.get("session_plan") or None}


def judged_with_hash(asset, dossier_hash, db_path=None):
    with _connect(db_path) as con:
        return con.execute(
            "SELECT 1 FROM analyst_intraday_log WHERE asset=? AND dossier_hash=? "
            "LIMIT 1", (asset, dossier_hash)).fetchone() is not None


def backfill(db_path=None, today=None):
    """Fill the session that followed each judgment. Returns rows filled.

    Only a bar dated BEFORE today counts. market.db has held partial daily bars
    since March 2026 (the fetcher stored sessions still in progress), and a
    session scored off a snapshot of itself would be scored on a price that
    never closed.
    """
    today = today or _dt.date.today().isoformat()
    filled = 0
    bars_by_asset = {}
    with _connect(db_path) as con:
        pending = con.execute(
            "SELECT date, asset FROM analyst_intraday_log "
            "WHERE session_date IS NULL ORDER BY asset, date").fetchall()
        for date, asset in pending:
            if asset not in bars_by_asset:
                bars_by_asset[asset] = ohlc_series(asset, days=6000, db_path=db_path)
            bars = bars_by_asset[asset]
            idx = next((i for i, b in enumerate(bars) if b["date"] == date), None)
            if idx is None or idx + 1 >= len(bars):
                continue
            nxt = bars[idx + 1]
            if nxt["date"] >= today:
                continue
            con.execute(
                "UPDATE analyst_intraday_log SET session_date=?, session_open=?, "
                "session_high=?, session_low=?, session_close=? "
                "WHERE date=? AND asset=?",
                (nxt["date"], nxt["open"], nxt["high"], nxt["low"], nxt["close"],
                 date, asset))
            filled += 1
    return filled


def pending_count(db_path=None):
    with _connect(db_path) as con:
        return con.execute("SELECT COUNT(*) FROM analyst_intraday_log "
                           "WHERE session_date IS NULL").fetchone()[0]


def scored_rows(db_path=None):
    with _connect(db_path) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(
            "SELECT * FROM analyst_intraday_log WHERE session_date IS NOT NULL "
            "ORDER BY date, asset")]


def recent(n=15, db_path=None):
    """The last n judgments across every asset, newest first, scored or not."""
    with _connect(db_path) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(
            "SELECT * FROM analyst_intraday_log ORDER BY date DESC, asset "
            "LIMIT ?", (int(n),))]


def latest(asset, db_path=None):
    """This asset's most recent session judgment, or None."""
    with _connect(db_path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM analyst_intraday_log WHERE asset=? "
                          "ORDER BY date DESC LIMIT 1", (asset,)).fetchone()
        return dict(row) if row else None


# ---- scoring ---------------------------------------------------------------

def _binom_greater(k, n, p=0.5):
    if not n:
        return None
    from scipy.stats import binomtest
    return round(float(binomtest(k, n, p, alternative="greater").pvalue), 4)


def _paired(agent_hits, base_hits):
    """Agent against a baseline on the SAME rows: a sign test on the rows where
    exactly one of them was right. The rows both got right, or both wrong, say
    nothing about which is better."""
    wins = sum(1 for a, b in zip(agent_hits, base_hits) if a and not b)
    losses = sum(1 for a, b in zip(agent_hits, base_hits) if b and not a)
    return {"wins": wins, "losses": losses,
            "p": _binom_greater(wins, wins + losses)}


def _rate(hits):
    return round(sum(hits) / len(hits), 4) if hits else None


def _session_range(o, h, low):
    return (h - low) / o if o else None


def _ranges(bars):
    """The last HISTORY_SESSIONS session ranges that are not zero. A vendor's
    flat bar (high == low, GOLD has two in its last 60) is not a quiet session,
    and counting it would leave the asset with too few to class against."""
    out = [_session_range(b["open"], b["high"], b["low"]) for b in bars]
    return [r for r in out if r][-HISTORY_SESSIONS:]


def _history(asset, session_date, db_path, cache):
    """Ranges of the HISTORY_SESSIONS sessions before `session_date`, or []."""
    if asset not in cache:
        cache[asset] = ohlc_series(asset, days=6000, db_path=db_path)
    return _ranges([b for b in cache[asset] if b["date"] < session_date])


def _range_class(r, history):
    q = statistics.quantiles(history, n=3)
    return "calm" if r < q[0] else "elevated" if r > q[1] else "normal"


def _calendar_hit(row):
    """Whether the calendar the judgment saw had an event ON the session date."""
    try:
        cal = json.loads(row.get("calendar_json") or "{}")
    except ValueError:
        return False
    dates = {str(e.get("date"))[:10] for e in cal.get("macro_events") or []
             if isinstance(e, dict)}
    earn = cal.get("next_earnings")
    if isinstance(earn, dict) and earn.get("date"):
        dates.add(str(earn["date"])[:10])
    return row["session_date"] in dates


def _lead_rule(rows, db_path):
    """{(date, asset): True for up / False for down} from "follow the last US
    session", with the sign read only from bars before each date."""
    import pandas as pd

    import lead_baseline

    out = {}
    con = sqlite3.connect(db_path or store.DB_PATH)
    try:
        lead = lead_baseline.load_closes(con, LEADER)
        if lead is None:
            return out
        lead_ret = lead.pct_change().dropna()
        series = {}
        for r in rows:
            a = r["asset"]
            if a not in series:
                s = lead_baseline.load_closes(con, a)
                series[a] = None if s is None else s.pct_change().dropna()
            if series[a] is None or r.get("us_last_session_ret") is None:
                continue
            sign, _ic = lead_baseline.lead_sign(lead_ret, series[a],
                                                pd.Timestamp(r["date"]))
            if sign is None:
                continue
            us_up = r["us_last_session_ret"] > 0
            out[(r["date"], a)] = us_up if sign > 0 else not us_up
    finally:
        con.close()
    return out


def score(db_path=None):
    """Every question's numbers on the scored rows. Pure reading."""
    from scipy.stats import mannwhitneyu

    from core.analyst.score import conviction_calibration

    rows = scored_rows(db_path)
    out = {"n": len(rows)}

    # D: direction, open -> close.
    d_hits, conv_rows = [], []
    for r in rows:
        o, c = r["session_open"], r["session_close"]
        if r["direction"] in ("up", "down") and o and c != o:
            d_hits.append((r["direction"] == "up") == (c > o))
            conv_rows.append({"direction": r["direction"],
                              "conviction": r["conviction"],
                              "realized_ret": (c - o) / o})
    out["direction"] = {"n": len(d_hits), "hit_rate": _rate(d_hits),
                        "p_vs_coin": _binom_greater(sum(d_hits), len(d_hits)),
                        "conviction": conviction_calibration(conv_rows)}

    # B: the gap, information only (see the module docstring), beside the
    # US-close rule on the same rows.
    lead_rows = [r for r in rows if r["gap"] in ("up", "down")
                 and r["session_open"] != r["close_at_signal"]]
    rule = _lead_rule(lead_rows, db_path)
    b_agent, b_rule = [], []
    for r in lead_rows:
        key = (r["date"], r["asset"])
        if key not in rule:
            continue
        up = r["session_open"] > r["close_at_signal"]
        b_agent.append((r["gap"] == "up") == up)
        b_rule.append(rule[key] == up)
    out["gap"] = {"n": len(b_agent), "hit_rate": _rate(b_agent),
                  "rule_hit_rate": _rate(b_rule), **_paired(b_agent, b_rule)}

    # C and A both need the asset's own recent sessions.
    cache, c_agent, c_base, c_har = {}, [], [], []
    stand, trade, cal_on, cal_off = [], [], [], []
    for r in rows:
        rng = _session_range(r["session_open"], r["session_high"], r["session_low"])
        hist = _history(r["asset"], r["session_date"], db_path, cache)
        if rng is None or len(hist) < HISTORY_SESSIONS:
            continue
        view = session_range_view([b for b in cache[r["asset"]] if b["date"] <= r["date"]])
        actual = _range_class(rng, hist)
        if view is not None:          # paired with HAR only where HAR can say
            c_agent.append(r["vol_regime"] == actual)
            c_base.append(actual == "normal")
            c_har.append(view["har_class"] == actual)
        med = statistics.median(hist)
        if med:
            surprise = rng / med
            (stand if r["stand_aside"] else trade).append(surprise)
            (cal_on if _calendar_hit(r) else cal_off).append(surprise)
    out["range"] = {"n": len(c_agent), "hit_rate": _rate(c_agent),
                    "har_hit_rate": _rate(c_har),
                    "always_normal_hit_rate": _rate(c_base),
                    **_paired(c_agent, c_har)}

    def split(on, off):
        mean = (lambda xs: round(statistics.fmean(xs), 3) if xs else None)
        p = (round(float(mannwhitneyu(on, off, alternative="greater").pvalue), 4)
             if on and off else None)
        return {"n_on": len(on), "n_off": len(off), "surprise_on": mean(on),
                "surprise_off": mean(off), "p": p}

    out["stand_aside"] = {**split(stand, trade), "calendar": split(cal_on, cal_off)}
    return out


def verdicts(s):
    """SHIP or HOLD per question, with what is missing. Same shape of answer as
    the daily score: a word alone would not say which condition failed."""
    def v(checks):
        missing = [want for ok, want in checks if not ok]
        return {"verdict": "HOLD" if missing else "SHIP", "missing": missing}

    def sig(p):
        return p is not None and p < 0.05

    d, r, a = s["direction"], s["range"], s["stand_aside"]
    floor = "%d scored sessions" % SHIP_FLOOR
    a_gap = (a["surprise_on"] or 0) - (a["surprise_off"] or 0)
    c = a["calendar"]
    c_gap = (c["surprise_on"] or 0) - (c["surprise_off"] or 0)
    return {
        "direction": v([(d["n"] >= SHIP_FLOOR, floor),
                        (sig(d["p_vs_coin"]), "better than a coin, p < 0.05")]),
        "gap": {"verdict": "INFO", "missing": [
            "not a tradable claim: the gap happens before the last close can be traded"]},
        "range": v([(r["n"] >= SHIP_FLOOR, floor),
                    (sig(r["p"]), "better than HAR, p < 0.05")]),
        "stand_aside": v([
            (a["n_on"] + a["n_off"] >= SHIP_FLOOR, floor),
            (a["n_on"] >= MIN_STAND_ASIDE,
             "%d stand-aside sessions" % MIN_STAND_ASIDE),
            (sig(a["p"]), "stand-aside sessions wilder than the rest, p < 0.05"),
            (a_gap > c_gap, "a bigger gap than the calendar rule's")]),
    }
