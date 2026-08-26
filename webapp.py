"""Web interface: signal radar, track record, models, risk.

Reads ready-made predictions from market.db (written by predict.py); it does not
load any models, so it starts instantly.

Run:
    uvicorn webapp:app --host 0.0.0.0 --port 8000
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import FULL_ASSET_MAP, RADAR_GROUPS, radar_category
from core import dashboard, timing_policy, track_record
from core import levels as levels_mod
from core import positions as positions_mod
from risk_manager import RISK_CONFIG, RiskManager, save_risk_config_override

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The web process reads the same .env every other entry point does. Without
# this it ran with GTRADE_TIMING_POLICY and GTRADE_TIMING_STAGE unset while
# predict.py had them set, so the served timing label and the watched-Q badge
# could never appear on a card no matter what the database held - the page was
# reporting the absence of a flag as the absence of a decision. load_dotenv
# does not override a variable already in the environment, so an explicit
# setting on the command line still wins.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"))
except Exception:
    pass
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
QUALITY_PATH = os.path.join(MODEL_DIR, "quality_report.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")
# train_payoff.py writes to the repo root beside levels_policy.json, not to
# MODEL_DIR like the two paths above.
PAYOFF_STATS_PATH = os.path.join(BASE_DIR, "payoff_stats.json")

app = FastAPI(title="Atratus")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _hms(seconds):
    """Seconds as a compact human duration; "unknown" when seconds is None or
    otherwise not a usable number. Total, because a filter that raises inside a
    render turns the status page into a 500."""
    if seconds is None:
        return "unknown"
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if total < 60:
        return "%ds" % total
    if total < 3600:
        return "%dm" % (total // 60)
    return "%dh %02dm" % (total // 3600, (total % 3600) // 60)


templates.env.filters["hms"] = _hms
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

LOOP_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loop_state.json")


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return default
    return default


_MISSING = object()


def _payoff_evidence(table, asset, asset_class, side):
    """Text naming what backs one side's number, for the card's sub-line.

    calibrate.forecast is always called here with cells={} (there is no
    analyst_log judgment history to shrink toward yet on this page), so its
    own "n" is always 0 and cannot tell the reader anything. The payoff
    table's own cells can: the asset's own cell for this side when the table
    has one, plus the class cell either way - the design's "the panel states
    its own sample size so the reader can discount it" promise, kept.
    """
    own = table.get("asset", {}).get(asset, {}).get(side)
    cls = table.get("class", {}).get(asset_class, {}).get(side)
    cls_n = cls["n"] if cls else 0
    if own:
        return f"{own['n']} on this asset + {cls_n} on class {asset_class}"
    return f"{cls_n} on class {asset_class}"


def _payoff_context(name, atr, close, table=_MISSING, analyst=None):
    """Both sides of the expected payoff for one asset, plus the analyst's read.

    `table` defaults to a sentinel rather than None so a caller can pass
    table=None to mean "no table" explicitly, without that also being the
    default that reads the real (gitignored) payoff_stats.json from disk. A
    missing table degrades the panel to None rather than raising: an install
    that never ran train_payoff.py must still serve the asset page.
    """
    from core.analyst import calibrate

    if table is _MISSING:
        table = _load_json(PAYOFF_STATS_PATH, None)
    if not table:
        return {"long": None, "short": None, "analyst": analyst,
                "coverage": None}

    cls = radar_category(name)
    sides = {}
    for label, direction, side in (("long", "up", "BUY"), ("short", "down", "SELL")):
        fc = calibrate.forecast(
            {"direction": direction, "conviction": 3, "vol_regime": "normal"},
            {}, name, cls, atr, close, table)
        fc["evidence"] = _payoff_evidence(table, name, cls, side)
        sides[label] = fc
    return {"long": sides["long"], "short": sides["short"],
            "analyst": analyst, "coverage": None}


def _risk_snapshot():
    rm = RiskManager()
    halted, halt_reason = rm.is_trading_halted()
    return {
        "state": {
            "current_capital": rm.current_capital,
            "peak_capital": rm.peak_capital,
            "initial_capital": rm.initial_capital,
            "open_positions": rm.open_positions,
        },
        "dd": rm.current_drawdown,
        "halted": halted,
        "halt_reason": halt_reason,
        "manual_halt": rm.manual_halt,
    }


def _latest_price(asset):
    series = track_record.price_series(asset, days=5)
    return series[-1]["close"] if series else None


def _portfolio_snapshot():
    """Portfolio view over the risk-manager positions (the same book as /risk):
    holdings + diversification / sector heat / held-asset correlation / per-
    position warnings (portfolio.py analytics on the current positions)."""
    rm = RiskManager()
    cap = rm.current_capital or 1.0
    pm = dashboard.portfolio_manager()
    positions = rm.open_positions
    # weights as a fraction of capital
    fractions = {a: (p.get("size_usd", 0) / cap) for a, p in positions.items()}
    held = list(positions.keys())

    holdings = []
    for a, p in positions.items():
        price = _latest_price(a)
        size = p.get("size_usd", 0)
        entry = p.get("entry_price")
        direction = p.get("direction", "BUY")
        pnl = None
        if price and entry:
            ret = (price - entry) / entry if direction == "BUY" else (entry - price) / entry
            pnl = ret * size
        corr_open = []
        if pm is not None:
            corr_open = [x for x in pm.get_correlated_assets(a, open_only=held) if x != a]
        holdings.append({
            "asset": a, "direction": direction, "size_usd": size,
            "weight": fractions[a], "sector": pm.get_sector(a) if pm else "OTHER",
            "entry": entry, "price": price, "pnl": pnl,
            "correlated_open": corr_open,
        })
    holdings.sort(key=lambda h: h["weight"], reverse=True)

    heat = []
    diversification = 100.0
    corr_rows = []
    if pm is not None and fractions:
        from portfolio import SECTOR_LIMITS
        diversification = pm.get_diversification_score(fractions)
        for sector, exp in pm.get_portfolio_heat(fractions).items():
            if exp <= 0:
                continue
            limit = SECTOR_LIMITS.get(sector, 0.20)
            heat.append({"sector": sector, "exposure": exp, "limit": limit,
                         "over": exp > limit})
        heat.sort(key=lambda h: h["exposure"], reverse=True)
        corr = pm.get_correlation_matrix()
        if held and not corr.empty:
            for a in held:
                vals = []
                for b in held:
                    c = None
                    if a in corr.index and b in corr.columns:
                        try:
                            c = float(corr.loc[a, b])
                        except Exception:
                            c = None
                    vals.append(c)
                corr_rows.append({"asset": a, "vals": vals})

    return {
        "holdings": holdings,
        "held": held,
        "total_exposure": sum(fractions.values()),
        "diversification": diversification,
        "heat": heat,
        "corr_rows": corr_rows,
        "capital": rm.current_capital,
    }


def _research_snapshot():
    """Findings-journal snapshot for the /research page: cumulative counters plus a
    flattened newest-first list of winners. Read-only.

    The whole journal, not a window on it: the page is the only place the run
    history can be read, and a cap there meant the older half of the campaign
    was invisible while the wiki still cited it. 63 records / 94 winner rows
    today, and the page scrolls them.
    """
    from core import ar_memory
    summary = ar_memory.findings_summary()
    recent = list(reversed(ar_memory.findings_all()))
    rows = []
    for rec in recent:
        for w in rec.get("winners", []):
            rows.append({
                "ts": rec.get("ts", ""), "mode": rec.get("mode", ""),
                "axis": w.get("axis", ""), "adoptable": bool(w.get("adoptable")),
                "replicated": bool(w.get("replicated")),
                "clears": w.get("clears") or 0, "neural_lift": w.get("neural_lift"),
                "tag": w.get("tag", ""),
            })
    from core import ar_progress
    # The unattended cycle's own stage. ar_progress reports what a TRAINING run
    # is doing; this reports which phase of search / A/B / adopt that training
    # belongs to, which is the question the page could not answer before.
    try:
        import auto_loop
        cycle = auto_loop.read_state()
    except Exception:
        cycle = {"current": None, "campaign": None, "history": []}
    # Who chose each cycle and what the bandit believes. Wrapped: the page is
    # read-only and must render even when the director module cannot load.
    try:
        from core import ar_director_rl
        arms = ar_director_rl.posteriors()
        director = {
            "mode": ar_director_rl.mode(),
            "arms": sorted(({"arm": a, "mean": m,
                             "hours": ar_director_rl.RECIPES[a]["hours"]}
                            for a, m in arms.items()),
                           key=lambda r: -r["mean"]),
            "recent": [{"ts": h.get("ts"), "cycle": h.get("cycle"),
                        "arm": ar_director_rl.arm_of(h.get("settings")),
                        "chosen_by": h.get("chosen_by")}
                       for h in (cycle.get("history") or [])[:10]],
        }
    except Exception:
        director = {"mode": "llm", "arms": [], "recent": []}
    # The policy layers and how they did on live signals. Read from the
    # snapshot policy_status.py writes, never recomputed here: the live sizing
    # arm reads every asset's bars, which is not something a page request does.
    policies = _load_json(os.path.join(BASE_DIR, "policy_status.json"), None)
    return {"summary": summary, "rows": rows, "runs": len(recent),
            "progress": ar_progress.snapshot(), "cycle": cycle,
            "director": director, "policies": policies}


def _experience_snapshot(sig=None):
    """What the research has learned, joined for the page. Read-only.

    Each section is wrapped on its own: the journals are written by a
    long-running agent while this page is read, and one unreadable file must
    cost its own panel, not the whole screen.
    """
    from core import experience

    out = {"funnel": None, "levers": [], "genomes": [], "generations": [],
           "selected": None, "similar": [], "errors": []}
    for key, call in (("funnel", experience.funnel),
                      ("levers", experience.levers),
                      ("genomes", experience.genomes)):
        try:
            out[key] = call()
        except Exception:
            out["errors"].append(key)
    try:
        out["generations"] = experience.generations()
    except Exception:
        out["errors"].append("generations")
    if sig:
        # Split so a broken similar() cannot be reported as a broken genome():
        # the page promises to name the source it could not read.
        try:
            out["selected"] = experience.genome(sig)
        except Exception:
            out["errors"].append("genome")
        try:
            out["similar"] = experience.similar(sig)
        except Exception:
            out["errors"].append("similar")
    return out


def _accuracy_panels():
    """The four evidence panels on the accuracy page. Read-only.

    Wrapped per panel for the same reason /experience is: one unreadable
    source must cost its own panel, not the page.

    The level-outcomes panel is exposed as "level_outcomes", not "levels":
    the /performance context already has a "levels" key from
    performance_tracker.level_summary() for the existing Trade levels panel,
    and a shared key would silently overwrite it.
    """
    import performance_tracker as pt
    from core import experience

    out = {"calibration": [], "by_asset": [], "generations": [],
           "level_outcomes": {"rows": []},
           "panel_errors": []}
    for key, call in (("calibration", pt.calibration),
                      ("by_asset", pt.accuracy_by_asset),
                      ("level_outcomes", pt.level_outcomes),
                      ("generations", experience.generations)):
        try:
            out[key] = call()
        except Exception:
            out["panel_errors"].append(key)
    return out


def _spark(closes, w=110, h=26):
    """Points for an svg sparkline from a list of closes."""
    if len(closes) < 2:
        return None
    lo, hi = min(closes), max(closes)
    span = (hi - lo) or 1.0
    step = w / (len(closes) - 1)
    pts = " ".join(
        f"{i * step:.1f},{h - 2 - (c - lo) / span * (h - 4):.1f}"
        for i, c in enumerate(closes)
    )
    chg = (closes[-1] - closes[0]) / closes[0] if closes[0] else 0.0
    return {"points": pts, "w": w, "h": h, "up": closes[-1] >= closes[0], "chg": chg}


def _summary(signals, stale):
    counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
    verified = correct = 0
    last_date = None
    for s in signals:
        counts[s["signal"]] = counts.get(s["signal"], 0) + 1
        verified += s["acc"]["n"]
        correct += s["acc"]["correct"]
        if last_date is None or s["date"] > last_date:
            last_date = s["date"]
    return {
        "total": len(signals),
        "counts": counts,
        "accuracy": (correct / verified) if verified else None,
        "verified": verified,
        "last_date": last_date,
        "stale": len(stale),
    }


def _top_signals(signals, n=5):
    actionable = [s for s in signals if s["signal"] in ("BUY", "SELL")]
    return sorted(actionable, key=lambda s: abs((s["probability"] or 0.5) - 0.5),
                  reverse=True)[:n]


def _timing_badge(row, show_timing):
    """Divergence-only badge string for a radar row, or None. `show_timing` is
    the reversibility guard (spec section 4.3), precomputed once per request."""
    if not show_timing:
        return None
    act = row.get("timing_action")
    if not act:
        return None
    text, is_div = timing_policy.display_label(act, row.get("timing_reason"))
    return text if is_div else None


def _grouped_signals(signals):
    show_timing = timing_policy.timing_on() and timing_policy.load_policy() is not None
    sigs = {s["asset"]: s for s in signals}
    groups = []
    for group, members in RADAR_GROUPS.items():
        rows = [sigs[a] for a in members if a in sigs]
        for r in rows:
            r["cat"] = radar_category(r["asset"])
            r["timing_badge"] = _timing_badge(r, show_timing)
        if rows:
            groups.append({"name": group, "rows": rows})
    return groups


@app.get("/", response_class=HTMLResponse)
def radar(request: Request):
    signals = track_record.latest_signals()
    taleb = dashboard.taleb_index()
    soft_cap, hard_cap = RISK_CONFIG["taleb_soft_cap"], RISK_CONFIG["taleb_risk_cap"]
    spark_series = track_record.price_series_many(
        [s["asset"] for s in signals], days=30)
    for s in signals:
        closes = [p["close"] for p in spark_series.get(s["asset"], [])]
        s["spark"] = _spark(closes)
        s["taleb"] = taleb.get(s["asset"])
        s["taleb_regime"] = dashboard.taleb_regime(s["taleb"], soft_cap, hard_cap)
    stale = track_record.stale_assets()
    regime = dashboard.global_regime()
    score = dashboard.regime_score(regime)
    sentiment = dashboard.market_sentiment()
    return templates.TemplateResponse(request, "radar.html", {
        "groups": _grouped_signals(signals),
        "summary": _summary(signals, stale),
        "top": _top_signals(signals),
        "stale": stale[:8],
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "score": score,
        "zone": dashboard.gauge_zone(score),
        "sentiment": sentiment,
        "sent_zone": dashboard.gauge_zone(sentiment["score"]),
        "breadth": dashboard.market_breadth(),
        "leaderboard": dashboard.top_leaderboard(limit=5),
        "config": RISK_CONFIG,
    })


@app.get("/asset/{name}", response_class=HTMLResponse)
def asset_page(request: Request, name: str):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    track = track_record.asset_track(name, limit=60)
    acc = track_record.asset_accuracy(name)
    # The watched challenger checked against what the bar then did, the same
    # correct/missed a signal's row gets. Its own definition of "right": in a
    # position, the bar went its way; flat while the signal wanted in, the trade
    # it skipped would not have paid. Oldest first, because a position is
    # rebuilt forward from the entries.
    from core.policy_report import live_timing_hits
    watched_hits = live_timing_hits(list(reversed(track)), "shadow_action")
    for row, verdict in zip(reversed(track), watched_hits["verdicts"]):
        row["watched_correct"] = verdict

    rets = [t["actual_next_ret"] for t in track if t["actual_next_ret"] is not None]
    wins = [t for t in track if t["correct"] == 1]
    losses = [t for t in track if t["correct"] == 0]
    stats = {
        "avg_ret": sum(rets) / len(rets) if rets else None,
        "wins": len(wins),
        "losses": len(losses),
        "outcomes": [t["correct"] for t in track[:15]][::-1],
    }

    reg = _load_json(REGISTRY_PATH, {}).get(name)
    thr = _load_json(THRESHOLDS_PATH, {}).get(name)
    if thr is None and reg:
        thr = {"buy": reg.get("buy_thr"), "sell": reg.get("sell_thr")}
    quality = next((q for q in _load_json(QUALITY_PATH, [])
                    if q.get("Asset") == name), None)
    group = next((g for g, m in RADAR_GROUPS.items() if name in m), None)

    # Collapse the per-bar signals into positions: enter/exit markers for the
    # chart, a state ribbon, a trade log and the current-position card.
    pos = positions_mod.build_positions(
        [{"date": t["date"], "signal": t["signal"], "ret": t["actual_next_ret"]}
         for t in reversed(track)])
    markers = pos["markers"]
    taleb = dashboard.taleb_for_asset(name)
    soft_cap, hard_cap = RISK_CONFIG["taleb_soft_cap"], RISK_CONFIG["taleb_risk_cap"]

    # asset_track doesn't carry the live-gate columns; pull the gated display
    # value + reason from latest_gated() so the chip matches the radar page.
    current = dict(track[0]) if track else None
    if current:
        gated = track_record.latest_gated(name)
        if gated:
            current["signal"] = gated["signal"]
            current["signal_raw"] = gated["signal_raw"]
            current["gate_reason"] = gated["gate_reason"]
        else:
            current["signal_raw"] = current.get("signal")
            current["gate_reason"] = None

        current["timing_label"] = current["watched_label"] = None
        if timing_policy.timing_on() and timing_policy.load_policy() is not None:
            act = current.get("timing_action") or (gated or {}).get("timing_action")
            if act:
                text, _div = timing_policy.display_label(
                    act, current.get("timing_reason")
                    or (gated or {}).get("timing_reason"))
                current["timing_label"] = text
            # The challenger, when it is only being watched. Shown ONLY where it
            # disagrees with what actually served: agreement is the common case
            # and carries nothing, and a chip on every card would be read as a
            # second instruction rather than as a comparison.
            from core import timing_fqi as _fq
            watched = (current.get("shadow_action")
                       or (gated or {}).get("shadow_action"))
            if _fq.stage_b_shadow_on() and watched and watched != act:
                current["watched_label"] = timing_policy.watched_label(watched)

    # Concrete prices beside the signal: where to enter, where to bail. The same
    # core.levels call the trade-levels sheet makes, on the same open segment, so
    # the card and the sheet can never quote different numbers for one asset on
    # one day. Sizing stays on the sheet: it needs an equity figure, and the card
    # answers "where", not "how much".
    segments = pos.get("segments") or []
    open_segment = segments[-1] if segments and segments[-1].get("open") else None
    taleb_hi, risky = dashboard.regime_flags(name, taleb=taleb)
    # The side the timing layer is actually on, the same one the journal
    # records and the fit is measured against. Showing the raw call here while
    # the badge below reports a policy that disagrees is two instructions on one
    # card; the badge is what says they differ, and that is its whole job.
    asset_levels = levels_mod.levels(
        track_record.ohlc_series(name, days=60),
        levels_mod.acting_side((current or {}).get("signal"), name,
                               (current or {}).get("timing_action")),
        segment=open_segment, taleb_hi=taleb_hi, risky=risky)

    return templates.TemplateResponse(request, "asset.html", {
        "asset": name,
        "ticker": FULL_ASSET_MAP[name],
        "levels": asset_levels,
        "levels_policy": levels_mod.policy_evidence(),
        "taleb": taleb,
        "taleb_regime": dashboard.taleb_regime(taleb, soft_cap, hard_cap),
        "taleb_soft_cap": soft_cap,
        "taleb_hard_cap": hard_cap,
        "group": group,
        "cat": radar_category(name),
        "track": track,
        "watched_acc": ({"n": watched_hits["decided"],
                         "correct": watched_hits["hits"],
                         "accuracy": watched_hits["accuracy"]}
                        if watched_hits["decided"] else None),
        "acc": acc,
        "stats": stats,
        "current": current,
        "position": pos["current"],
        "trades": pos["trades"],
        "segments": pos["segments"],
        "reg": reg,
        "thr": thr,
        "quality": quality,
        "markers_json": json.dumps(markers),
        "guru": dashboard.guru_for_asset(name),
        "payoff": _payoff_context(name, asset_levels.get("atr"),
                                  asset_levels.get("close")),
    })


@app.get("/models", response_class=HTMLResponse)
def models_page(request: Request):
    quality = _load_json(QUALITY_PATH, [])
    registry = _load_json(REGISTRY_PATH, {})
    sigs = {s["asset"]: s for s in track_record.latest_signals()}

    rows = []
    for q in quality:
        asset = q.get("Asset")
        reg = registry.get(asset, {})
        sig = sigs.get(asset)
        rows.append({
            "asset": asset,
            "score": q.get("Score"),
            "cb_acc": q.get("CB_Acc"),
            "lstm_acc": q.get("LSTM_Acc"),
            "profit": q.get("Profit"),
            "trades": q.get("Trades"),
            "status": q.get("Status"),
            "policy": q.get("Policy"),
            "mode": reg.get("ensemble_mode", "-"),
            "lookback": reg.get("lookback"),
            "updated": str(reg.get("updated_at", ""))[:10],
            "signal": sig["signal"] if sig else None,
            "live_acc": sig["acc"] if sig else None,
        })
    rows.sort(key=lambda r: r["score"] or 0, reverse=True)

    n = len(rows)
    stable = sum(1 for r in rows if r["status"] == "STABLE")
    last_train = max((r["updated"] for r in rows if r["updated"]), default=None)
    summary = {
        "total": n,
        "stable": stable,
        "unstable": n - stable,
        "avg_score": (sum(r["score"] or 0 for r in rows) / n) if n else None,
        "last_train": last_train,
    }
    return templates.TemplateResponse(request, "models.html", {
        "rows": rows, "summary": summary,
        "health": dashboard.models_health(),
        "stale": dashboard.models_stale(),
    })


def _taleb_top(limit=10):
    """Assets with the highest current Taleb tail-risk index, regime-tagged."""
    soft_cap, hard_cap = RISK_CONFIG["taleb_soft_cap"], RISK_CONFIG["taleb_risk_cap"]
    items = [(a, v) for a, v in dashboard.taleb_index().items() if v is not None]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [{"asset": a, "taleb": v,
             "regime": dashboard.taleb_regime(v, soft_cap, hard_cap)}
            for a, v in items[:limit]]


def _taleb_counts():
    """How many assets sit in each tail-risk band right now.

    The bands are risk_manager's own gates, so this is not a mood reading: the
    elevated count is how many assets are being sized down and the extreme count
    is how many cannot be bought at all. Per-asset that has always been visible;
    across the market it was not visible anywhere.
    """
    soft_cap, hard_cap = RISK_CONFIG["taleb_soft_cap"], RISK_CONFIG["taleb_risk_cap"]
    counts = {"normal": 0, "elevated": 0, "extreme": 0, "na": 0}
    for value in dashboard.taleb_index().values():
        counts[dashboard.taleb_regime(value, soft_cap, hard_cap)] += 1
    counts["total"] = counts["normal"] + counts["elevated"] + counts["extreme"]
    counts["gated"] = counts["elevated"] + counts["extreme"]
    counts["soft_cap"] = soft_cap
    counts["hard_cap"] = hard_cap
    return counts


@app.get("/risk", response_class=HTMLResponse)
def risk_page(request: Request):
    return templates.TemplateResponse(request, "risk.html", {
        **_risk_snapshot(), "config": RISK_CONFIG,
        "full_asset_map": sorted(FULL_ASSET_MAP),
        "taleb_top": _taleb_top(),
    })


@app.get("/levels", response_class=HTMLResponse)
def levels_page(request: Request):
    risk = _risk_snapshot()
    # Money only once the real account has been declared on /risk: without it
    # the book still holds whatever the paper experiments left behind, and
    # sizing a real trade off that number is exactly the mistake to avoid.
    equity = risk["state"]["current_capital"] if RISK_CONFIG["equity"] else 0.0
    return templates.TemplateResponse(request, "levels.html", {
        "rows": dashboard.levels_sheet(equity),
        "levels_policy": levels_mod.policy_evidence(),
        "config": RISK_CONFIG,
        "halted": risk["halted"],
        "halt_reason": risk["halt_reason"],
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_page(request: Request):
    return templates.TemplateResponse(request, "portfolio.html", {
        **_portfolio_snapshot(),
        "full_asset_map": sorted(FULL_ASSET_MAP),
    })


@app.get("/api/portfolio")
def api_portfolio():
    return _portfolio_snapshot()


@app.get("/api/ticker")
def api_ticker():
    return {"movers": dashboard.top_movers()}


@app.get("/api/health")
def api_health():
    return dashboard.health()


@app.get("/api/palette")
def api_palette():
    pages = [
        ["Radar", "/"], ["Market", "/market"], ["Sectors", "/sectors"],
        ["Correlations", "/correlations"], ["Accuracy", "/performance"],
        ["News", "/news"], ["Guru", "/guru"], ["Models", "/models"],
        ["Risk", "/risk"], ["Portfolio", "/portfolio"], ["What-If", "/whatif"],
        ["Research", "/research"], ["Analyst", "/analyst"],
        ["Experience", "/experience"],
    ]
    return {"pages": pages, "assets": sorted(FULL_ASSET_MAP)}


@app.get("/loop", response_class=HTMLResponse)
def loop_page(request: Request):
    from core import loop_state
    return templates.TemplateResponse(request, "loop.html", loop_state.load_state(LOOP_STATE_PATH))


@app.get("/api/loop")
def api_loop():
    from core import loop_state
    return loop_state.load_state(LOOP_STATE_PATH)


@app.post("/api/loop/approve")
async def api_loop_approve(request: Request):
    from core import loop_state
    try:
        body = await request.json()
    except Exception:
        body = {}
    assets = [str(a).upper() for a in (body.get("assets") or [])]
    return loop_state.approve(LOOP_STATE_PATH, assets)


@app.post("/api/loop/dismiss")
async def api_loop_dismiss(request: Request):
    from core import loop_state
    try:
        body = await request.json()
    except Exception:
        body = {}
    return loop_state.dismiss(LOOP_STATE_PATH, str(body.get("asset", "")).upper())


@app.get("/research", response_class=HTMLResponse)
def research_page(request: Request):
    from core import ar_wiki
    snap = _research_snapshot()
    # No max_chars: the 6000 default is the prompt budget, not a reading limit.
    snap["wiki"] = ar_wiki.wiki_summary(None) if ar_wiki.wiki_on() else ""
    return templates.TemplateResponse(request, "research.html", snap)


# One analyst pass at a time. A module global is the right size for the single
# uvicorn worker run_gtrade.bat starts; several workers would each keep their
# own and need a file lock instead.
# ponytail: per-process guard, swap for core.runlock if webapp ever runs multi-worker
_ANALYST_PROC = None


def _analyst_running():
    return _ANALYST_PROC is not None and _ANALYST_PROC.poll() is None


@app.get("/analyst", response_class=HTMLResponse)
def analyst_page(request: Request):
    """What the analyst has said, and whether it is worth believing yet.

    Deliberately shows the sample size next to every figure: until the log
    reaches the decision floor, nothing here is evidence of anything.
    """
    from core.analyst import score as analyst_score
    from core.analyst import store
    try:
        rows = store.scored_rows()
        pending = store.pending_count()
    except Exception:
        rows, pending = [], 0
    return templates.TemplateResponse(request, "analyst.html", {
        "scored": len(rows),
        "pending": pending,
        "coverage": analyst_score.coverage(rows),
        "recent": list(reversed(rows))[:15],
        "running": _analyst_running(),
        "floor": 500,
        "disabled": (os.getenv("GTRADE_ANALYST") or "1").strip() == "0",
    })


@app.post("/api/analyst/run")
def api_analyst_run():
    """Start one analyst pass in the background.

    Spends money: one LLM call per eligible asset. Refuses while a pass is
    already running rather than starting a second, and refuses outright when
    GTRADE_ANALYST=0, so the kill switch means the same thing here as on the
    command line.
    """
    global _ANALYST_PROC
    if (os.getenv("GTRADE_ANALYST") or "1").strip() == "0":
        return {"started": False, "reason": "GTRADE_ANALYST=0 disables the agent"}
    if _analyst_running():
        return {"started": False, "reason": "a pass is already running",
                "pid": _ANALYST_PROC.pid}
    _ANALYST_PROC = subprocess.Popen(
        [sys.executable, "analyst.py", "run"], cwd=BASE_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"started": True, "pid": _ANALYST_PROC.pid}


@app.get("/api/analyst/status")
def api_analyst_status():
    from core.analyst import store
    try:
        return {"running": _analyst_running(),
                "scored": len(store.scored_rows()),
                "pending": store.pending_count()}
    except Exception as exc:
        return {"running": _analyst_running(), "error": str(exc)[:140]}


@app.get("/api/research")
def api_research():
    return _research_snapshot()


@app.get("/experience", response_class=HTMLResponse)
def experience_page(request: Request, sig: str = ""):
    return templates.TemplateResponse(request, "experience.html",
                                      _experience_snapshot(sig or None))


@app.get("/api/experience")
def api_experience(sig: str = ""):
    return _experience_snapshot(sig or None)


@app.get("/whatif", response_class=HTMLResponse)
def whatif_page(request: Request):
    return templates.TemplateResponse(request, "whatif.html", {
        "full_asset_map": sorted(FULL_ASSET_MAP),
    })


@app.post("/api/whatif")
async def api_whatif(request: Request):
    """Run a hypothetical CatBoost-signal backtest. The simulation is CPU-bound,
    so it runs in a threadpool to avoid blocking the event loop. Assets are
    capped and days bounded to keep a single request responsive."""
    from starlette.concurrency import run_in_threadpool

    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        capital = max(1.0, float(body.get("capital") or 10000))
        days_back = max(10, min(365, int(body.get("days_back") or 90)))
    except (TypeError, ValueError):
        return {"error": "capital and days must be numbers"}
    strategy = body.get("strategy") if body.get("strategy") in ("equal", "kelly") else "equal"

    import whatif_simulator as wf
    try:
        if body.get("mode") == "top":
            n = max(1, min(12, int(body.get("top_n") or 5)))
            return await run_in_threadpool(
                wf.simulate_top_n, n=n, capital=capital, days_back=days_back)
        assets = [a for a in (body.get("assets") or []) if a in FULL_ASSET_MAP][:12]
        if not assets:
            return {"error": "No valid assets selected"}
        return await run_in_threadpool(
            wf.simulate, assets, capital=capital, days_back=days_back, strategy=strategy)
    except Exception as exc:
        return {"error": "Simulation failed: " + str(exc)[:140]}


@app.get("/market", response_class=HTMLResponse)
def market_page(request: Request):
    regime = dashboard.global_regime()
    score = dashboard.regime_score(regime)
    sentiment = dashboard.market_sentiment()
    # Everything below is already computed and cached for other pages; the market
    # page just had no view of it. Nothing here starts a fresh calculation.
    return templates.TemplateResponse(request, "market.html", {
        "regime": regime,
        "score": score,
        "zone": dashboard.gauge_zone(score),
        "sentiment": sentiment,
        "sent_zone": dashboard.gauge_zone(sentiment["score"]),
        "breadth": dashboard.market_breadth(),
        "taleb": _taleb_counts(),
        "taleb_top": _taleb_top(5),
        "stress": dashboard.correlation_stress(),
    })


@app.get("/news", response_class=HTMLResponse)
def news_page(request: Request, lang: str = "all", category: str = "all"):
    items = dashboard.news_digest(lang=lang, category=category)
    return templates.TemplateResponse(request, "news.html", {
        "items": items, "lang": lang, "category": category,
    })


@app.get("/sectors", response_class=HTMLResponse)
def sectors_page(request: Request):
    return templates.TemplateResponse(request, "sectors.html", {
        "momentum": dashboard.sector_momentum(),
        "heatmap": dashboard.sector_heatmap(),
    })


@app.get("/correlations", response_class=HTMLResponse)
def correlations_page(request: Request):
    return templates.TemplateResponse(request, "correlations.html", {
        "stress": dashboard.correlation_stress(),
        "heatmap": dashboard.correlation_heatmap(),
    })


@app.get("/performance", response_class=HTMLResponse)
def performance_page(request: Request):
    import performance_tracker
    try:
        meta_shadow = performance_tracker.meta_shadow_report(days=30)
    except Exception:
        meta_shadow = {"rows": 0}
    try:
        levels = performance_tracker.level_summary()
    except Exception:
        levels = {"issued": 0}
    context = {
        "series": dashboard.accuracy_timeseries(),
        "leaderboard": dashboard.top_leaderboard(limit=20),
        "version": dashboard.current_model_version(),
        "meta_shadow": meta_shadow,
        "levels": levels,
    }
    context.update(_accuracy_panels())
    return templates.TemplateResponse(request, "performance.html", context)


@app.get("/guru", response_class=HTMLResponse)
def guru_page(request: Request):
    verdicts = dashboard.guru_latest()
    # Overlay the ML signal so it is always visible next to the value verdict;
    # flag divergence (ML bullish vs guru bearish or vice versa) as advisory.
    ml = {s["asset"]: s for s in track_record.latest_signals()}
    for v in verdicts:
        sig = ml.get(v["asset"])
        v["ml_signal"] = sig["signal"] if sig else None
        v["ml_prob"] = sig.get("probability") if sig else None
        v["divergent"] = bool(sig and (
            (sig["signal"] == "BUY" and v["verdict"] == "AVOID") or
            (sig["signal"] == "SELL" and v["verdict"] == "BUY")))
    return templates.TemplateResponse(request, "guru.html", {
        "verdicts": verdicts,
        "accuracy": dashboard.guru_accuracy(),
    })


@app.get("/api/regime")
def api_regime():
    regime = dashboard.global_regime()
    return {**regime, "score": dashboard.regime_score(regime)}


@app.get("/api/sentiment")
def api_sentiment():
    return dashboard.market_sentiment()


@app.get("/api/sectors")
def api_sectors():
    return {"momentum": dashboard.sector_momentum(),
            "heatmap": dashboard.sector_heatmap()}


@app.get("/api/correlations")
def api_correlations():
    return {"stress": dashboard.correlation_stress(),
            "heatmap": dashboard.correlation_heatmap()}


@app.get("/api/performance")
def api_performance():
    return {"series": dashboard.accuracy_timeseries(),
            "leaderboard": dashboard.top_leaderboard(limit=20)}


@app.post("/api/reconcile")
async def api_reconcile():
    """Fill in actual outcomes for pending predictions (the loop's reconcile
    step, on demand). DB-bound, so it runs in a threadpool like /api/whatif."""
    from starlette.concurrency import run_in_threadpool

    import performance_tracker
    try:
        return await run_in_threadpool(performance_tracker.update_actuals)
    except Exception as exc:
        return {"error": "Reconcile failed: " + str(exc)[:140]}


@app.get("/api/guru")
def api_guru():
    return {"verdicts": dashboard.guru_latest(),
            "accuracy": dashboard.guru_accuracy()}


@app.post("/api/guru/{asset}/recalculate")
def api_guru_recalculate(asset: str):
    """Live re-score of one asset, persisted as the new latest guru_log verdict.

    Reuses guru_report.py's fundamentals resolution and core.guru's scoring
    engine - the same logic the console report and app.py use - so this never
    drifts into a second implementation of guru scoring.
    """
    asset = asset.upper()
    if asset not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {asset}")

    import guru_report
    import guru_tracker
    from core.guru import get_guru_analysis

    symbol = FULL_ASSET_MAP[asset]
    smartlab = guru_report.fetch_smartlab_data()
    fund = guru_report.resolve_fundamentals(asset, symbol, smartlab)
    tech = guru_report.technical_context(guru_report.get_technical(asset))
    analysis = get_guru_analysis(fund, tech)

    price = (fund or {}).get('price') or (tech['close'] if tech else 0)
    data_source = analysis['data_source']
    council = analysis['council']

    # Guru is a fundamentals-based value verdict. Without real fundamentals
    # (crypto/forex/indices/commodities, or a stock whose data failed to load)
    # the "verdict" would just be a shaved momentum read mislabeled as guru - so
    # report an honest N/A and do NOT pollute the accuracy track record. The ML
    # signal for this asset is shown separately and is unaffected.
    if data_source in ("technical", "backup"):
        return {
            "asset": asset, "verdict": "N/A", "no_fundamentals": True,
            "source": data_source, "date": datetime.now().strftime("%Y-%m-%d"),
        }

    guru_tracker.log_guru_verdict(
        asset,
        analysis['lynch']['_score'], analysis['buffett']['_score'],
        analysis['graham']['_score'], analysis['munger']['_score'],
        council['pct'], council['verdict'], data_source, price,
    )
    return {
        "asset": asset, "verdict": council['verdict'], "pct": council['pct'],
        "source": data_source, "date": datetime.now().strftime("%Y-%m-%d"),
        "lynch": analysis['lynch'], "buffett": analysis['buffett'],
        "graham": analysis['graham'], "munger": analysis['munger'],
    }


# Background state for the "recalculate all guru verdicts" batch. The batch
# scrapes Smart-Lab once and hits yfinance per US/EU stock, so it runs for
# minutes - too long for a blocking request. One batch runs at a time; the UI
# polls /status and reloads when it finishes.
_guru_recalc_lock = threading.Lock()
_guru_recalc = {"running": False, "done": 0, "total": 0, "updated": 0,
                "skipped": 0, "errors": 0, "error": None, "finished": None}


def _run_guru_recalc():
    import guru_report
    try:
        def _prog(done, total, _asset):
            with _guru_recalc_lock:
                _guru_recalc["done"] = done
                _guru_recalc["total"] = total
        res = guru_report.recalc_all_stocks(progress=_prog)
        with _guru_recalc_lock:
            _guru_recalc.update(updated=res["updated"], skipped=res["skipped"],
                                errors=res["errors"])
    except Exception as exc:
        with _guru_recalc_lock:
            _guru_recalc["error"] = str(exc)[:200]
    finally:
        with _guru_recalc_lock:
            _guru_recalc["running"] = False
            _guru_recalc["finished"] = datetime.now().strftime("%H:%M:%S")


@app.post("/api/guru/recalculate-all")
def api_guru_recalculate_all():
    """Kick off a background re-score of every stock (Smart-Lab once + yfinance
    per US/EU name). Returns immediately with the initial status; a second call
    while a batch is running is a no-op that returns the live status."""
    with _guru_recalc_lock:
        if _guru_recalc["running"]:
            return dict(_guru_recalc)
        _guru_recalc.update(running=True, done=0, total=0, updated=0,
                            skipped=0, errors=0, error=None, finished=None)
    threading.Thread(target=_run_guru_recalc, daemon=True).start()
    with _guru_recalc_lock:
        return dict(_guru_recalc)


@app.get("/api/guru/recalculate-all/status")
def api_guru_recalculate_status():
    with _guru_recalc_lock:
        return dict(_guru_recalc)


@app.get("/api/news")
def api_news(lang: str = "all", category: str = "all"):
    return dashboard.news_digest(lang=lang, category=category)


@app.get("/api/signals")
def api_signals():
    return track_record.latest_signals()


@app.get("/api/prices/{name}")
def api_prices(name: str, days: int = 90):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    days = 100000 if days <= 0 else max(10, min(days, 100000))
    return {"asset": name, "series": track_record.price_series(name, days=days)}


@app.get("/api/ohlc/{name}")
def api_ohlc(name: str, days: int = 120):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    days = 100000 if days <= 0 else max(10, min(days, 100000))
    return {"asset": name, "series": track_record.ohlc_series(name, days=days)}


@app.get("/api/track/{name}")
def api_track(name: str):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    return {
        "asset": name,
        "track": track_record.asset_track(name, limit=60),
        "accuracy": track_record.asset_accuracy(name),
    }


@app.get("/api/risk")
def api_risk():
    return {**_risk_snapshot(), "config": RISK_CONFIG}


# The Risk Alerts scan reads every asset's price history + RSI (a few seconds), so
# it is fetched lazily by the /risk panel and memoized briefly instead of blocking
# the page or the 20s poll.
_ALERTS_CACHE = {"ts": 0.0, "alerts": None}
_ALERTS_TTL = 180.0


@app.get("/api/risk/alerts")
async def api_risk_alerts(force: bool = False):
    """The HTML report's Risk Alerts (RSI overbought/oversold, fearful VIX, stale
    models), surfaced for the /risk panel. DB-heavy, so it runs in a threadpool and
    is cached for a few minutes (force=1 bypasses the cache for a manual refresh)."""
    import time

    from starlette.concurrency import run_in_threadpool

    now = time.time()
    if not force and _ALERTS_CACHE["alerts"] is not None and now - _ALERTS_CACHE["ts"] < _ALERTS_TTL:
        return {"alerts": _ALERTS_CACHE["alerts"], "cached": True}
    try:
        import performance_report
        alerts = await run_in_threadpool(performance_report.collect_risk_alerts)
        _ALERTS_CACHE.update(ts=now, alerts=alerts)
        return {"alerts": alerts, "cached": False}
    except Exception as exc:
        return {"error": "Risk alerts failed: " + str(exc)[:140], "alerts": []}


@app.post("/api/risk/position")
async def api_risk_open_position(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    asset = str(body.get("asset", "")).upper()
    direction = body.get("direction")
    size_usd = body.get("size_usd")
    entry_price = body.get("entry_price")

    if asset not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {asset}")
    if direction not in ("BUY", "SELL"):
        raise HTTPException(400, "direction must be BUY or SELL")
    try:
        size_usd = float(size_usd)
    except (TypeError, ValueError):
        raise HTTPException(400, "size_usd must be a number")
    if size_usd <= 0:
        raise HTTPException(400, "size_usd must be > 0")

    if entry_price is None:
        entry_price = _latest_price(asset)
    if entry_price is None:
        raise HTTPException(400, f"No price available for {asset} - supply entry_price")

    rm = RiskManager()
    rm.record_trade(asset, direction, size_usd, float(entry_price))
    return _risk_snapshot()


@app.post("/api/risk/position/{asset}/close")
async def api_risk_close_position(asset: str, request: Request):
    asset = asset.upper()
    try:
        body = await request.json()
    except Exception:
        body = {}
    exit_price = body.get("exit_price")

    rm = RiskManager()
    if asset not in rm.open_positions:
        raise HTTPException(404, f"No open position for {asset}")

    if exit_price is None:
        exit_price = _latest_price(asset)
    if exit_price is None:
        raise HTTPException(400, f"No price available for {asset} - supply exit_price")

    pnl = rm.close_trade(asset, float(exit_price))
    return {"pnl": pnl, **_risk_snapshot()}


@app.post("/api/risk/config")
async def api_risk_config(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        save_risk_config_override(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"config": RISK_CONFIG}


@app.post("/api/risk/halt")
def api_risk_halt():
    rm = RiskManager()
    rm.set_manual_halt(True)
    return _risk_snapshot()


@app.post("/api/risk/resume")
def api_risk_resume():
    rm = RiskManager()
    rm.set_manual_halt(False)
    return _risk_snapshot()
