"""The read-only dossier the analyst reasons over.

Assembled from modules that already exist. No new data source and no new
fetcher: every field here is something the project already computes for some
other panel.

What is NOT here is the design. The ensemble probability, the emitted signal,
the timing action and the sizing decision are excluded, because a second
opinion built on the first one's output is not a second opinion. FORBIDDEN_KEYS
and the test over it are what keep that true after this file stops being read.
"""

import hashlib
import json

from core.levels import ATR_PERIOD, atr_abs
from core.track_record import ohlc_series, volume_series

FORBIDDEN_KEYS = frozenset({
    "probability", "cb_prob", "lstm_prob", "meta_prob", "signal",
    "sig_shown", "timing_action", "timing_reason", "timing_stage",
    "shadow_action", "gate_reason", "model_version", "correct",
    # actual_next_ret is not an ensemble channel, it is the realized outcome.
    # A dossier carrying it would be look-ahead, which is worse than the
    # failure this set was written to catch.
    "actual_next_ret",
})

HISTORY_BARS = 120
RECENT_BARS = 20


def _pct(a, b):
    if not a or not b:
        return None
    return (b - a) / a


def _safe(fn, default=None):
    """A dossier field is worth less than the run that produces it."""
    try:
        return fn()
    except Exception:
        return default


def _context(asset):
    """The non-price half of the dossier: fundamentals, events, macro.

    Each source reaches a database or the network on its own, so each call is
    wrapped in `_safe`: one dead source must not stop the day's run, and it
    must not stop `build()` from returning a fixed-shape dossier either.
    """
    from config import FULL_ASSET_MAP
    from core import events  # imported here, not at module level: both reach
    from core.dashboard import guru_for_asset  # the network

    verdict = _safe(lambda: guru_for_asset(asset)) or {}
    earnings = _safe(
        lambda: events.earnings_for({asset: FULL_ASSET_MAP[asset]})) or {}
    return {
        "guru_verdict": verdict.get("verdict"),
        "guru_pct": verdict.get("pct"),
        "next_earnings": earnings.get(asset),
        "macro_events": _safe(lambda: [e["name"] for e in events.load_macro()],
                              default=[]),
    }


def _atr_distance(price, level, atr):
    """How far a level sits from the price, in ATR. None when either is absent.

    A percentage says nothing without knowing how much this asset normally
    moves; two ATR is a long way for a bond ETF and an ordinary Tuesday for a
    small-cap. The model needs the distance in the asset's own units.
    """
    if price is None or level is None or not atr:
        return None
    return (level - price) / atr


def _realized_vol(closes, window):
    """Standard deviation of daily returns over `window` bars, as a fraction."""
    if len(closes) < window + 1:
        return None
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(len(closes) - window, len(closes)) if closes[i - 1]]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return var ** 0.5


def _streak(closes):
    """Consecutive closes in one direction, signed. Positive is up."""
    if len(closes) < 2:
        return None
    step = 1 if closes[-1] >= closes[-2] else -1
    n = 0
    for i in range(len(closes) - 1, 0, -1):
        up = 1 if closes[i] >= closes[i - 1] else -1
        if up != step:
            break
        n += 1
    return step * n


def _own_record(asset, before=None):
    """What this analyst has already said about this asset, and how it went.

    `before` clips it to judgments made strictly BEFORE that date, which is
    what makes the block safe to put in a historical dossier: unclipped, a
    judgment dated in May would be told how its May calls turned out.

    The design named the agent's own track record as a dossier input and it was
    never wired in. Without it the analyst repeats a call it has already been
    wrong about and has no way to know. Scored rows only: an unresolved
    judgment is an opinion, not evidence.
    """
    try:
        from core.analyst import store

        rows = [r for r in store.scored_rows() if r.get("asset") == asset
                and (before is None or (r.get("date") or "") < before)]
    except Exception:
        return {"past_calls": 0, "past_hit_rate": None, "past_last_call": None,
                "past_last_outcome": None}
    if not rows:
        return {"past_calls": 0, "past_hit_rate": None, "past_last_call": None,
                "past_last_outcome": None}
    hits = 0
    for r in rows:
        realized = r.get("realized_ret")
        if realized is None:
            continue
        want_up = r.get("direction") == "up"
        want_down = r.get("direction") == "down"
        if (want_up and realized > 0) or (want_down and realized < 0):
            hits += 1
    last = rows[-1]
    return {
        "past_calls": len(rows),
        "past_hit_rate": round(hits / len(rows), 3) if rows else None,
        "past_last_call": last.get("direction"),
        "past_last_outcome": (round(last["realized_ret"], 5)
                              if last.get("realized_ret") is not None else None),
    }


# Which index an asset is naturally read against. Without this the analyst can
# see that a name fell six percent and has no way to tell whether the whole
# market fell with it, which is the first thing a person would ask.
BENCHMARK_BY_CLASS = {
    "ru": "IMOEX",
    "us": "SP500",
    "eu": "STOXX600",
    "crypto": "BTC",
    "commodity": "DBC",
}
FEAR_GAUGE = "VIX"


def _median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def _series_ret(asset, span, db_path, today):
    """Return of one asset over `span` bars, from the same local database."""
    bars = ohlc_series(asset, days=span + 5, db_path=db_path)
    if today is not None:
        bars = [b for b in bars if b["date"] <= today]
    closes = [b["close"] for b in bars]
    if len(closes) < span + 1 or not closes[-(span + 1)]:
        return None
    return (closes[-1] - closes[-(span + 1)]) / closes[-(span + 1)]


def _correlation(a_closes, b_closes):
    """Pearson correlation of daily returns over the overlapping tail."""
    n = min(len(a_closes), len(b_closes))
    if n < 21:
        return None
    a, b = a_closes[-n:], b_closes[-n:]
    ra = [(a[i] - a[i - 1]) / a[i - 1] for i in range(1, n) if a[i - 1]]
    rb = [(b[i] - b[i - 1]) / b[i - 1] for i in range(1, n) if b[i - 1]]
    m = min(len(ra), len(rb))
    if m < 20:
        return None
    ra, rb = ra[-m:], rb[-m:]
    ma, mb = sum(ra) / m, sum(rb) / m
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(m))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    if not va or not vb:
        return None
    return cov / (va * vb)


def _market_context(asset, bars, db_path, today):
    """Where the market stood while this asset did whatever it did.

    Local only: every one of these is already an asset in the map with its own
    table, so this costs a few sqlite reads and no network at all.
    """
    from config import radar_category

    bench = BENCHMARK_BY_CLASS.get(radar_category(asset))
    out = {"benchmark": bench, "benchmark_ret_1": None, "benchmark_ret_20": None,
           "corr_to_benchmark_60": None, "vix_level": None, "vix_chg_20": None}
    if bench and bench != asset:
        out["benchmark_ret_1"] = _series_ret(bench, 1, db_path, today)
        out["benchmark_ret_20"] = _series_ret(bench, 20, db_path, today)
        bb = ohlc_series(bench, days=70, db_path=db_path)
        if today is not None:
            bb = [b for b in bb if b["date"] <= today]
        out["corr_to_benchmark_60"] = _correlation(
            [b["close"] for b in bars], [b["close"] for b in bb])
    if asset != FEAR_GAUGE:
        vix = ohlc_series(FEAR_GAUGE, days=30, db_path=db_path)
        if today is not None:
            vix = [b for b in vix if b["date"] <= today]
        if vix:
            out["vix_level"] = vix[-1]["close"]
            out["vix_chg_20"] = _series_ret(FEAR_GAUGE, 20, db_path, today)
    return out


def _flow(asset, bars, atr, db_path, today):
    """Volume against its own norm, the opening gap, and today's range.

    A move on triple the usual volume and the same move on a third of it are
    different events, and the dossier could not tell them apart before.
    """
    out = {"volume_vs_20": None, "turnover": None,
           "gap_open": None, "range_atr": None}
    vols = volume_series(asset, days=60, db_path=db_path)
    if today is not None:
        vols = [v for v in vols if v["date"] <= today]
    if vols:
        today_vol = vols[-1]["volume"]
        norm = _median([v["volume"] for v in vols[-21:-1]])
        if today_vol and norm:
            out["volume_vs_20"] = today_vol / norm
        out["turnover"] = vols[-1].get("value")
    if len(bars) >= 2 and bars[-2]["close"]:
        out["gap_open"] = (bars[-1]["open"] - bars[-2]["close"]) / bars[-2]["close"]
    if bars and atr:
        out["range_atr"] = (bars[-1]["high"] - bars[-1]["low"]) / atr
    return out


_FUNDAMENTALS_BLANK = {"pe": None, "roe": None, "debt_ebitda": None,
                       "div_yield": None}

# The profile block's shape, in one place. It is a module constant so the test
# suite's no-network stub can BE this shape instead of restating it: a stub that
# restates a shape is a stub that silently lags the real one, and four new
# fields reached the dossier while every dossier test kept seeing seven.
PROFILE_BLANK = {"sector": None, "industry": None, "market_cap": None,
                 "float_shares": None, "short_ratio": None, "beta": None,
                 "ex_dividend_date": None, **_FUNDAMENTALS_BLANK}

# Smart-Lab publishes one table for the whole market, so this is a single
# request for every Russian name in a sweep, not one per asset. Cached for the
# life of the process; the numbers are quarterly, so a stale value inside one
# run is still a true value.
_SMARTLAB_CACHE = {}


def _smartlab_map():
    if "map" not in _SMARTLAB_CACHE:
        try:
            from guru_report import fetch_smartlab_data

            _SMARTLAB_CACHE["map"] = fetch_smartlab_data() or {}
        except Exception:
            _SMARTLAB_CACHE["map"] = {}
    return _SMARTLAB_CACHE["map"]


def _smartlab_fundamentals(asset):
    """P/E, ROE, debt/EBITDA and dividend yield for a Moscow-listed name.

    The key comes from guru_report.smartlab_ticker, which derives it from
    FULL_ASSET_MAP rather than from a hand-written remap: this function used to
    carry its own copy of that remap with two entries in it, so HH.ru, X5, Fix
    Price and MOEX itself silently had no fundamentals here either.
    """
    from guru_report import smartlab_ticker

    key = smartlab_ticker(asset)
    row = _smartlab_map().get(key) if key else None
    if not row:
        return dict(_FUNDAMENTALS_BLANK)
    # 99.0 is fetch_smartlab_data's "not reported" filler, not a P/E of 99.
    pe = row.get("pe")
    return {"pe": None if pe in (None, 99.0) else pe,
            "roe": row.get("roe") or None,
            "debt_ebitda": row.get("debt") or None,
            "div_yield": row.get("div") or None}


def _profile(asset):
    """Slow-moving facts about the instrument itself, not an opinion about it.

    Sector, size, float, short interest and beta describe what the thing IS.
    They change over months, so a stale value is still a true value, and the
    call is skipped entirely for symbols this source cannot resolve, which is
    the same guard the earnings scan uses and the reason Moscow-listed names
    stopped producing a 404 each.

    The ex-dividend date earns its place on its own: a dividend gap looks
    exactly like a fall and is not one, and an analyst that cannot see the date
    will read the drop as weakness every single time.
    """
    blank = dict(PROFILE_BLANK)
    try:
        from config import FULL_ASSET_MAP
        from core.events import can_have_earnings

        symbol = FULL_ASSET_MAP.get(asset)
        if not can_have_earnings(symbol, asset):
            # Yahoo cannot resolve a bare Moscow ticker, which is why the call
            # is skipped - but the project already fetches those fundamentals
            # from Smart-Lab for the guru report, so the whole block used to be
            # blank for every Russian name for want of a lookup. SBER read as
            # an instrument with no sector, no size and no dividend at all.
            return {**blank, **_smartlab_fundamentals(asset)}
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
    except Exception:
        return blank
    ex = info.get("exDividendDate")
    if ex:
        try:
            import datetime as _dt

            ex = _dt.datetime.fromtimestamp(int(ex), _dt.UTC).date().isoformat()
        except Exception:
            ex = None
    return {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "float_shares": info.get("floatShares"),
        "short_ratio": info.get("shortRatio"),
        "beta": info.get("beta"),
        "ex_dividend_date": ex,
        "pe": info.get("trailingPE"),
        "roe": info.get("returnOnEquity"),
        "debt_ebitda": None,          # not in Yahoo's info payload
        "div_yield": info.get("dividendYield"),
    }


_REGIME_BLANK = {"regime_trend": None, "regime_vol": None,
                 "regime_momentum": None, "rsi_14": None, "atr_vs_90d": None}
_MARKET_BLANK = {"breadth_above_sma50_pct": None,
                 "breadth_positive_20d_pct": None,
                 "cross_asset_corr": None, "cross_asset_corr_label": None}
_SECTOR_BLANK = {"sector_group": None, "sector_momentum": None,
                 "sector_trend": None}

# get_market_breadth reads every asset table in the database: 12.7s measured on
# 844 of them, against 0.1s for the per-asset regime. Market-wide answers are
# the same for every asset in a run, so they are computed once and held for the
# life of the process, the way _SMARTLAB_CACHE is. Inside the sweep loop this
# would have been 12.7s per asset instead of 12.7s per run.
_MARKET_CACHE = {}


def _regime(asset):
    """This asset's own regime: trend, volatility band, momentum, RSI.

    From regime_detector, which classifies off SMA, ATR and RSI over the price
    table and reaches no model, so nothing here is a channel of the ensemble's
    opinion. RSI is new to the dossier outright; atr_vs_90d measures today's
    ATR against a 90-bar norm, a much longer memory than vol_20_vs_60's.
    """
    try:
        import regime_detector

        r = regime_detector.get_asset_regime(asset) or {}
    except Exception:
        return dict(_REGIME_BLANK)
    if not r:
        return dict(_REGIME_BLANK)
    return {"regime_trend": r.get("trend"), "regime_vol": r.get("volatility"),
            "regime_momentum": r.get("momentum"), "rsi_14": r.get("rsi"),
            "atr_vs_90d": r.get("atr_ratio")}


def _market_state():
    """Breadth and cross-asset correlation, the same for every asset today.

    Breadth answers whether a fall was this name's or everyone's, one level
    above the single benchmark the dossier already carries. Average pairwise
    correlation is the other half: a market where everything moves together is
    a different place to be short than one where it does not.
    """
    if "state" not in _MARKET_CACHE:
        out = dict(_MARKET_BLANK)
        try:
            import regime_detector

            b = regime_detector.get_market_breadth() or {}
            out["breadth_above_sma50_pct"] = b.get("above_sma50_pct")
            out["breadth_positive_20d_pct"] = b.get("positive_20d_pct")
        except Exception:
            pass
        try:
            import correlation_alert

            c = correlation_alert.get_stress_indicator() or {}
            if c.get("label") != "N/A":
                out["cross_asset_corr"] = c.get("avg_corr")
                out["cross_asset_corr_label"] = c.get("label")
        except Exception:
            pass
        _MARKET_CACHE["state"] = out
    return dict(_MARKET_CACHE["state"])


def _sector_momentum_table():
    """{sector: (momentum_score, trend)}, computed once per process."""
    if "sectors" not in _MARKET_CACHE:
        table = {}
        try:
            import sector_rotation

            df = sector_rotation.get_sector_momentum()
            for _i, row in df.iterrows():
                table[row["Sector"]] = (float(row["Momentum_Score"]),
                                        row["Trend"])
        except Exception:
            table = {}
        _MARKET_CACHE["sectors"] = table
    return _MARKET_CACHE["sectors"]


def _sector_state(asset):
    """Whether this asset's sector is being bought or sold.

    The group comes from config.SECTOR_MAP, the project's canonical asset-to-
    sector map, NOT from the Yahoo `sector` field - so Moscow-listed names get
    a group (Russia) where their Yahoo sector is blank. SECTOR_MAP is curated
    rather than exhaustive, so an asset outside it keeps the blank shape: no
    sector is a true answer, not a missing one.
    """
    try:
        from config import SECTOR_MAP
    except Exception:
        return dict(_SECTOR_BLANK)
    group = next((g for g, names in SECTOR_MAP.items() if asset in names), None)
    if group is None:
        return dict(_SECTOR_BLANK)
    score, trend = _sector_momentum_table().get(group, (None, None))
    return {"sector_group": group, "sector_momentum": score,
            "sector_trend": trend}


def _headlines(asset, limit=6):
    """Raw headlines, deliberately without the sentiment score computed on them.

    A headline is material the analyst can read; a sentiment number is somebody
    else's reading of it, and the whole point of this agent is that it forms
    its own. news_analyzer caches internally, so repeated calls in one run are
    cheap.
    """
    try:
        import news_analyzer

        items = news_analyzer.fetch_news(asset, max_articles=limit) or []
    except Exception:
        return {"headlines": []}
    out = []
    for it in items[:limit]:
        title = (it.get("title") or "").strip() if isinstance(it, dict) else ""
        if title:
            out.append(title[:180])
    return {"headlines": out}


# Which dossier blocks come off the network rather than out of market.db.
# Ordered, because they are printed as a run summary. macro is NOT here: it is
# read from a local calendar file, so an empty one says nothing about the
# connection and gets its own line.
NETWORK_BLOCKS = ("headlines", "guru", "fundamentals", "earnings")


def _applicable(asset):
    """(fundamentals_possible, earnings_possible) for this asset.

    The two are not the same question, which is why the first version of this
    summary told a person to check the network because SBER had no earnings
    date. can_have_earnings documents both of its reasons: called WITHOUT the
    asset it answers "is this a company at all", which is what decides whether
    fundamentals can exist; called WITH it, it also excludes Moscow-listed
    names, whose earnings this source genuinely cannot serve while their
    fundamentals come from Smart-Lab and are unaffected.
    """
    try:
        from config import FULL_ASSET_MAP
        from core.events import can_have_earnings

        symbol = FULL_ASSET_MAP.get(asset)
        return bool(can_have_earnings(symbol)), bool(
            can_have_earnings(symbol, asset))
    except Exception:
        return True, True


def filled_blocks(dossier):
    """Which network blocks arrived: 1 yes, 0 expected but empty, None n/a here.

    Per-block, not per-field: `sector` is legitimately absent for a Moscow name,
    so a field-level count would cry wolf on every sweep. A block counts as
    arrived when ANY of its fields did. None is the third state the first
    version lacked: an index has no earnings and gold has no P/E, and counting
    those as misses is what turned a correct run into a connection warning.
    """
    funda_ok, earn_ok = _applicable(dossier.get("asset"))
    return {
        "headlines": int(bool(dossier.get("headlines"))),
        "guru": int(dossier.get("guru_verdict") is not None),
        "fundamentals": int(any(dossier.get(k) is not None for k in
                                ("sector", "market_cap", "beta", "pe", "roe",
                                 "div_yield"))) if funda_ok else None,
        "earnings": (int(dossier.get("next_earnings") is not None)
                     if earn_ok else None),
    }


def macro_status():
    """Why macro_events is empty, in the words of the actual cause.

    Returns None when there is nothing to report. The calendar is a local file
    that ships only as an example, so the honest message names the file rather
    than blaming the network for it.
    """
    import os

    from core import events

    if not os.path.exists(events.MACRO_PATH):
        return ("%s does not exist (only the .example), so no macro event "
                "reached any judgment" % os.path.basename(events.MACRO_PATH))
    if not events.load_macro():
        return "%s carries no usable events" % os.path.basename(events.MACRO_PATH)
    return None


def _as_of(asset, today):
    """The blocks whose value depends on WHEN you ask, resolved for `today`.

    today=None is the live path: everything is fetched exactly as before.

    today set means a historical judgment, and then most of this cannot be had
    honestly. Yahoo will not tell you last month's P/E, the news feeds will not
    give you last month's front page, and regime_detector, correlation_alert
    and sector_rotation all classify off the WHOLE price table with no date
    argument, so on a past date they answer using bars from AFTER it. Feeding
    any of that to a backfilled judgment is look-ahead, and look-ahead does not
    produce a weak measurement, it produces a flattering one - which is worse,
    because it reads as success.

    The agent's own record is the one block here that genuinely rewinds, and it
    is also the most dangerous one left unrewound: unfiltered it would tell a
    judgment dated in May how its May calls turned out.

    So a historical dossier keeps what is truly rewound - every price field,
    flow, market context, and the own record as of that date - and blanks the
    rest. It is thinner than a live one and must be read as such. The
    alternative was a backfill whose numbers could not be trusted at all.
    """
    if today is None:
        return {**_profile(asset), **_regime(asset), **_market_state(),
                **_sector_state(asset), **_headlines(asset), **_context(asset),
                **_own_record(asset)}
    return {**PROFILE_BLANK, **_REGIME_BLANK, **_MARKET_BLANK, **_SECTOR_BLANK,
            "headlines": [], "guru_verdict": None, "guru_pct": None,
            "next_earnings": None, "macro_events": [],
            **_own_record(asset, before=today)}


def build(asset, db_path=None, today=None):
    """One asset's dossier. Missing pieces are None, never absent, so that the
    prompt and the hash have a fixed shape regardless of what was available.

    `today` rewinds it: see _as_of for what can be rewound and what is blanked
    rather than faked.
    """
    bars = ohlc_series(asset, days=HISTORY_BARS, db_path=db_path)
    if today is not None:
        bars = [b for b in bars if b["date"] <= today]

    close = bars[-1]["close"] if bars else None
    atr = atr_abs(bars) if len(bars) >= ATR_PERIOD else None
    closes = [b["close"] for b in bars]

    high_20 = max(closes[-RECENT_BARS:]) if closes else None
    low_20 = min(closes[-RECENT_BARS:]) if closes else None
    high_60 = max(closes[-60:]) if closes else None
    vol_now = _realized_vol(closes, 20)
    vol_ref = _realized_vol(closes, 60)

    return {
        "asset": asset,
        "date": bars[-1]["date"] if bars else None,
        "close": close,
        "atr": atr,
        "atr_pct": (atr / close) if (atr and close) else None,
        "ret_1": _pct(closes[-2], closes[-1]) if len(closes) >= 2 else None,
        "ret_5": _pct(closes[-6], closes[-1]) if len(closes) >= 6 else None,
        "ret_20": _pct(closes[-21], closes[-1]) if len(closes) >= 21 else None,
        "ret_60": _pct(closes[-61], closes[-1]) if len(closes) >= 61 else None,
        "high_20": high_20,
        "low_20": low_20,
        # The same two levels in the asset's own units of movement, which is
        # what makes "close to the low" mean something comparable across a
        # bond ETF and a small-cap.
        "atr_to_high_20": _atr_distance(close, high_20, atr),
        "atr_to_low_20": _atr_distance(close, low_20, atr),
        "drawdown_60": _pct(high_60, close) if high_60 else None,
        # Volatility measured against this asset's own recent norm rather than
        # an absolute threshold: 2 percent daily is calm for one name and a
        # crisis for another.
        "vol_20": vol_now,
        "vol_20_vs_60": (vol_now / vol_ref) if (vol_now and vol_ref) else None,
        "streak_days": _streak(closes),
        "bars_available": len(bars),
        **_flow(asset, bars, atr, db_path, today),
        **_market_context(asset, bars, db_path, today),
        **_as_of(asset, today),
    }


def dossier_hash(dossier):
    """A stable 16-hex digest, used as the LLM cache key.

    Rounded before hashing: an unchanged dossier must hash the same across
    runs, and raw floats out of sqlite do not reliably do that.
    """
    rounded = {k: (round(v, 8) if isinstance(v, float) else v)
               for k, v in sorted(dossier.items())}
    blob = json.dumps(rounded, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
