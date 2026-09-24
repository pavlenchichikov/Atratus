"""Sources the analyst can ASK for, rather than ones it is always handed.

The dossier is a fixed shape decided before the model sees anything. That is
what makes a judgment reproducible: the same dossier hashes the same, the cache
knows it was judged, and `--back` can rebuild a past one. Tools break that if
they are done carelessly, so three rules hold everything together:

  Every call and every result is RECORDED with the judgment. A tool call the
  log does not carry is a judgment nobody can replay, which is the same defect
  as an unscored one.

  Every tool declares whether it REWINDS. A rewound run (`--as-of`, `--back`)
  may only use tools that honour the date, or backfilling May would hand the
  model September's filings - the exact trap core/analyst/dossier.py's `_as_of`
  exists to close, reopened from a new direction.

  The registry is an ALLOW-LIST of sources, never a fetch-any-URL. A model that
  can be told which page to read is a model an attacker can steer through a
  headline, and every one of these results goes straight into a prompt.

  A tool returns MATERIAL, never somebody's conclusion. Sell-side consensus,
  price targets, ratings and buy/sell calls are all off the list by decision of
  the owner, 2026-09-03: the point of this agent is its own reading, and a
  second opinion assembled from other people's opinions is not one. It is the
  same rule FORBIDDEN_KEYS applies to the ensemble's own output in
  core/analyst/dossier.py. `register` enforces it rather than
  trusting a reader to remember, because the tempting sources are exactly the
  easy ones to wire: Yahoo hands out recommendationMean in the same payload
  this project already fetches for P/E.

  Extended 2026-09-24 to the project's own guru council. News is material
  and stays, but always spread across publishers (dossier.diverse_headlines),
  so no single outlet or the company's own channel can carry a judgment.

Budget matters more here than anywhere else in the project: a local 26b model
answers in 9 to 25 minutes, so every extra round trip is another quarter hour.
So a reply may ask for SEVERAL tools at once: MAX_CALLS bounds the sources,
MAX_ROUNDS the round trips, and the operator can set either to zero.
"""

import datetime
import json
import os

MAX_CALLS = 6
MAX_ROUNDS = 2
TIMEOUT = 12
# Some public APIs (FRED, Wikimedia) refuse a request that names no client.
PLAIN_UA = {"User-Agent": "AtratusResearch/1.0 (personal research tool)"}

# SEC's published access policy requires a User-Agent naming a real contact,
# and returns 403 to anything else. That contact is the operator's, not the
# project's, so it is read from the environment and never committed: the tool
# says what to set rather than shipping somebody's address to a third party.
UA_ENV = "GTRADE_SEC_CONTACT"
UA_HELP = ("SEC requires a contact address in the User-Agent and answers 403 "
           "without one. Set %s to an email you are willing to give them, "
           "for example %s=you@example.com" % (UA_ENV, UA_ENV))


def user_agent():
    contact = (os.getenv(UA_ENV) or "").strip()
    return "Atratus research %s" % contact if contact else None

_REGISTRY = {}


# Words that name somebody else's conclusion rather than material. Matched
# against a tool's name and description at registration, so the refusal lands
# on whoever is adding the tool instead of on a judgment months later.
OPINION_WORDS = ("consensus", "price target", "price_target", "analyst rating",
                 "rating", "recommendation", "upgrade", "downgrade",
                 "buy/sell call", "estimate revision", "guru")


class OpinionSource(Exception):
    """Raised when a tool would hand the model a conclusion to borrow."""


class Tool:
    def __init__(self, name, args, rewinds, describe, run, applies=None):
        self.name = name
        self.args = args           # {arg: "what it is"}
        self.rewinds = rewinds     # honours `today`, so a backfill may use it
        self.describe = describe
        self.run = run
        # asset -> bool. The menu shows only what can answer for this asset:
        # offered SEC filings and US options for SBER, gemma spent two of its
        # six requests on empty replies (2026-09-24).
        self.applies = applies


def us_listed(asset):
    """A name with SEC filings and listed US options: a plain US ticker."""
    import re

    from config import FULL_ASSET_MAP, MOEX_ASSETS

    sym = FULL_ASSET_MAP.get(asset) or asset
    return asset not in MOEX_ASSETS and bool(re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", sym))


def is_crypto(asset):
    from config import ASSET_TYPES

    return asset in ASSET_TYPES.get("CRYPTO", [])


def register(tool):
    text = ("%s %s" % (tool.name, tool.describe)).lower()
    hit = next((w for w in OPINION_WORDS if w in text), None)
    if hit:
        raise OpinionSource(
            "%r looks like a source of other people's conclusions (%r). The "
            "analyst is meant to form its own; consensus and ratings are "
            "deliberately out. If this really returns material rather than a "
            "verdict, reword the description." % (tool.name, hit))
    _REGISTRY[tool.name] = tool
    return tool


def available(today=None, asset=None):
    """The tools this run may use. A rewound run keeps only the dated ones,
    and with an asset named, only the ones that can answer for it."""
    return [t for t in _REGISTRY.values()
            if (t.rewinds or today is None)
            and (asset is None or t.applies is None or t.applies(asset))]


def spec_lines(today=None, asset=None):
    """The tool menu as it appears in the prompt, or "" when there is none."""
    tools = available(today, asset)
    if not tools:
        return ""
    lines = [
        "",
        (("Your FIRST reply must be a request for more raw evidence, not a "
          "judgment: pick the sources below that bear on this asset and "
          "horizon, up to %d at once, in this form:" if require_first() else
          "You may ask for MORE raw evidence before deciding. To do that, "
          "return this instead of a judgment, with up to %d requests at once:")
         % max_calls()),
        '{"tools": [{"tool": "<name>", "args": {...}}, ...]}',
        ("You get every result together and are asked again. Each round "
         "costs real time, so ask for everything you need in ONE reply, and "
         "never twice for the same thing. Available:"),
    ]
    for t in sorted(tools, key=lambda x: x.name):
        args = ", ".join('"%s": <%s>' % (k, v) for k, v in t.args.items())
        lines.append('  %s {%s}  -  %s' % (t.name, args, t.describe))
    return "\n".join(lines)


def parse_request(data):
    """A validated {tool, args} request, or None if this is not one.

    Returns None rather than raising for anything unrecognised: the same reply
    is also checked against the judgment schema, and only one of the two is
    supposed to match.
    """
    if not isinstance(data, dict):
        return None
    name = data.get("tool")
    if not isinstance(name, str) or name not in _REGISTRY:
        return None
    args = data.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    allowed = set(_REGISTRY[name].args)
    return {"tool": name, "args": {k: v for k, v in args.items() if k in allowed}}


def parse_requests(data):
    """Every valid request in a reply: the batch form {"tools": [...]} or the
    single {"tool": ...} form. [] when the reply is not a request at all."""
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        return [r for r in (parse_request(d) for d in data["tools"]) if r]
    one = parse_request(data)
    return [one] if one else []


def call(request, asset, today=None):
    """Run one validated request. Never raises; a dead source is a result too.

    The returned dict is what gets recorded AND what goes back into the prompt,
    so it carries the request beside the answer: a log entry that says what came
    back but not what was asked is not a replay.
    """
    tool = _REGISTRY.get(request["tool"])
    entry = {"tool": request["tool"], "args": request["args"],
             "at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    if tool is None:
        return {**entry, "error": "no such tool"}
    if tool.applies is not None and asset and not tool.applies(asset):
        return {**entry, "error": "this tool does not cover %s" % asset}
    if today is not None and not tool.rewinds:
        # Not an error the model caused, so it is told plainly rather than
        # being left to wonder why the answer was empty.
        return {**entry, "error": "this tool cannot answer for a past date"}
    try:
        return {**entry, "result": tool.run(asset=asset, today=today,
                                            **request["args"])}
    except Exception as exc:
        return {**entry, "error": "%s: %s" % (type(exc).__name__, exc)[:200]}


def _get(url, headers=None):
    """Through net.http_get, which owns the VPN routing.

    urllib would take whatever route the OS happens to have, and this project
    has two: MOEX answers only over a Russian exit and the rest work better
    over a foreign one. net.py learns which is which per host and fails over,
    so a tool that bypasses it works on the operator's machine and nowhere
    else.
    """
    import net

    resp = net.http_get(url, headers=headers or PLAIN_UA, timeout=TIMEOUT)
    if resp is None:
        raise OSError("no route answered for %s" % url)
    resp.raise_for_status()
    return resp.text


# --------------------------------------------------------------------------
# insider_filings: what the people who run the company DISCLOSED doing with
# their own shares. Public filings, on the schedule the regulator sets. This is
# not, and must not be confused with, material non-public information.
# --------------------------------------------------------------------------

def _sec_cik(symbol, agent=None):
    """The SEC's zero-padded CIK for a ticker, from its public mapping file."""
    raw = _get("https://www.sec.gov/files/company_tickers.json",
               headers={"User-Agent": agent})
    for row in json.loads(raw).values():
        if str(row.get("ticker", "")).upper() == symbol.upper():
            return str(row["cik_str"]).zfill(10)
    return None


def _insider_filings(asset, today=None, limit=8):
    """Recent Form 4 filings for a US name, newest first.

    Form 4 is the disclosure an officer, director or 10% holder must file
    within two business days of trading their own company's stock. It is
    public by construction and published by the SEC itself.
    """
    from config import FULL_ASSET_MAP, MOEX_ASSETS

    if asset in MOEX_ASSETS:
        return {"note": "SEC filings do not cover Moscow-listed names; "
                        "Russian disclosure is published at e-disclosure.ru "
                        "and is not wired into this tool yet.",
                "filings": []}
    agent = user_agent()
    if not agent:
        return {"note": UA_HELP, "filings": []}
    symbol = (FULL_ASSET_MAP.get(asset) or asset).split("-")[0].split(".")[0]
    cik = _sec_cik(symbol, agent=agent)
    if not cik:
        return {"note": "no SEC registrant matches %s" % symbol, "filings": []}
    raw = _get("https://data.sec.gov/submissions/CIK%s.json" % cik,
               headers={"User-Agent": agent})
    recent = (json.loads(raw).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    out = []
    for i, form in enumerate(forms):
        if str(form).strip() != "4" or i >= len(dates):
            continue
        # The date bound is what makes this tool rewindable at all.
        if today is not None and str(dates[i]) > str(today):
            continue
        out.append({"filed": dates[i], "form": form,
                    "reporter": (recent.get("primaryDocDescription") or
                                 [None] * len(forms))[i]})
        if len(out) >= limit:
            break
    return {"symbol": symbol, "cik": cik, "filings": out,
            "note": "Form 4 is a DISCLOSED trade by an insider, filed within "
                    "two business days. Counts and dates only; read it as "
                    "activity, not as a recommendation."}


register(Tool(
    name="insider_filings",
    args={},
    rewinds=True,
    describe="recent disclosed insider trades (SEC Form 4) for this asset",
    run=_insider_filings, applies=us_listed))


# --------------------------------------------------------------------------
# news_search: many publishers on a topic the model picks. Live only: an RSS
# feed has no archive, so a rewound run would silently receive today's news.
# --------------------------------------------------------------------------

def _news_search(asset, today=None, query=""):
    import news_analyzer
    from core.analyst.dossier import NEWS_LIMIT, diverse_headlines

    term = str(query or asset).strip()[:80]
    items = news_analyzer.fetch_news(term, max_articles=NEWS_LIMIT * 5) or []
    return {"query": term, **diverse_headlines(items, asset, query=term)}


register(Tool(
    name="news_search",
    args={"query": "what to search for, a few words"},
    rewinds=False,
    describe="titles on a topic from many publishers, at most two per outlet",
    run=_news_search))


# --------------------------------------------------------------------------
# macro_series: official statistics from FRED (St. Louis Fed). Dated, so a
# rewind is honoured by cutting at `today`; later revisions are the one thing
# that leaks, and they move a CPI print by a tenth, not a direction.
# --------------------------------------------------------------------------

FRED_SERIES = {
    "us_10y_yield": "DGS10", "us_2y_yield": "DGS2", "us_curve_10y_2y": "T10Y2Y",
    "fed_funds": "DFF", "us_cpi": "CPIAUCSL", "us_unemployment": "UNRATE",
    "us_breakeven_10y": "T10YIE", "high_yield_spread": "BAMLH0A0HYM2",
    "brent": "DCOILBRENTEU", "usd_broad_index": "DTWEXBGS",
}


def _fred_observations(raw):
    obs = []
    for line in raw.splitlines()[1:]:
        day, _, val = line.partition(",")
        try:
            obs.append((day, float(val)))
        except ValueError:
            continue          # FRED writes "." or "" for a holiday
    return obs


def _macro_series(asset, today=None, series=""):
    sid = FRED_SERIES.get(str(series).strip())
    if not sid:
        return {"note": "unknown series; one of: %s" % ", ".join(sorted(FRED_SERIES))}
    end = (datetime.date.fromisoformat(str(today)[:10]) if today
           else datetime.date.today())
    start = end - datetime.timedelta(days=400)
    obs = _fred_observations(_get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s&cosd=%s&coed=%s"
        % (sid, start.isoformat(), end.isoformat())))
    if not obs:
        return {"series": series, "note": "no observations"}

    def back(days):
        cut = (datetime.date.fromisoformat(obs[-1][0])
               - datetime.timedelta(days=days)).isoformat()
        prior = [v for d, v in obs if d <= cut]
        return round(obs[-1][1] - prior[-1], 4) if prior else None

    return {"series": series, "fred_id": sid, "last": obs[-1],
            "change_1m": back(30), "change_3m": back(91), "change_1y": back(365),
            "recent": obs[-8:]}


register(Tool(
    name="macro_series",
    args={"series": "one of " + ", ".join(sorted(FRED_SERIES))},
    rewinds=True,
    describe="official US macro and market statistics from FRED, with changes",
    run=_macro_series))


# --------------------------------------------------------------------------
# company_financials: the numbers a US company FILED with the SEC (XBRL), not
# anybody's model of them. Rewinds on the filing date.
# --------------------------------------------------------------------------

SEC_TAGS = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet"),
    "net_income": ("NetIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "shares_outstanding": ("CommonStockSharesOutstanding",),
}


def _filed_series(facts, tags, today=None, periods=6):
    """The last `periods` filed values of the first tag this filer uses, one
    per period end, the latest filing for that period winning."""
    for tag in tags:
        units = (facts.get(tag) or {}).get("units") or {}
        rows = next(iter(units.values()), [])
        rows = [r for r in rows if r.get("form") in ("10-Q", "10-K")
                and (today is None or str(r.get("filed", "")) <= str(today))]
        if not rows:
            continue
        by_end = {}
        for r in sorted(rows, key=lambda r: r.get("filed", "")):
            by_end[r["end"]] = {"period_end": r["end"], "fp": r.get("fp"),
                                "value": r.get("val"), "filed": r.get("filed")}
        return sorted(by_end.values(), key=lambda r: r["period_end"])[-int(periods):]
    return None


def _company_financials(asset, today=None, quarters=6):
    from config import FULL_ASSET_MAP, MOEX_ASSETS

    if asset in MOEX_ASSETS:
        return {"note": "SEC filings do not cover Moscow-listed names; their "
                        "reported ratios are already in the dossier from Smart-Lab."}
    agent = user_agent()
    if not agent:
        return {"note": UA_HELP}
    symbol = (FULL_ASSET_MAP.get(asset) or asset).split("-")[0].split(".")[0]
    cik = _sec_cik(symbol, agent=agent)
    if not cik:
        return {"note": "no SEC registrant matches %s" % symbol}
    facts = json.loads(_get("https://data.sec.gov/api/xbrl/companyfacts/CIK%s.json"
                            % cik, headers={"User-Agent": agent})).get("facts", {})
    merged = {**facts.get("us-gaap", {}), **facts.get("dei", {})}
    out = {"symbol": symbol}
    for label, tags in SEC_TAGS.items():
        out[label] = _filed_series(merged, tags, today=today, periods=quarters)
    out["note"] = ("As filed. fp FY rows are full years; some filers report "
                   "year-to-date figures in their Q rows.")
    return out


register(Tool(
    name="company_financials",
    args={"quarters": "how many recent periods, default 6"},
    rewinds=True,
    describe="revenue, net income, cash flow, EPS and shares as filed with the SEC",
    run=_company_financials, applies=us_listed))


# --------------------------------------------------------------------------
# options_positioning: what the listed options market holds right now. Open
# interest is positions, not views. Live. Implied volatility is left out:
# Yahoo reports it off an empty book outside the session (NVDA read 2 percent
# at a true 40 on 2026-09-24).
# --------------------------------------------------------------------------

def _options_positioning(asset, today=None):
    import yfinance as yf

    from config import FULL_ASSET_MAP

    tk = yf.Ticker(FULL_ASSET_MAP.get(asset) or asset)
    # Skip the expiries inside five days: their implied volatility is mostly
    # noise and Yahoo reports it as 0 when the book is empty.
    soon = (datetime.date.today() + datetime.timedelta(days=5)).isoformat()
    expiries = [e for e in (tk.options or []) if e >= soon][:2]
    if not expiries:
        return {"note": "no listed options for this asset"}
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        spot = None
    out = []
    for exp in expiries:
        ch = tk.option_chain(exp)
        c_oi = float(ch.calls["openInterest"].fillna(0).sum())
        p_oi = float(ch.puts["openInterest"].fillna(0).sum())
        c_vol = float(ch.calls["volume"].fillna(0).sum())
        p_vol = float(ch.puts["volume"].fillna(0).sum())
        row = {"expiry": exp, "call_oi": c_oi, "put_oi": p_oi,
               "put_call_oi": round(p_oi / c_oi, 3) if c_oi else None,
               "put_call_volume": round(p_vol / c_vol, 3) if c_vol else None}
        out.append(row)
    return {"spot": spot, "expiries": out}


register(Tool(
    name="options_positioning",
    args={},
    rewinds=False,
    describe="listed options: put/call open interest and volume for the next two expiries",
    run=_options_positioning, applies=us_listed))


# --------------------------------------------------------------------------
# crypto_derivatives: perpetual futures on Binance. Funding is what longs pay
# shorts, open interest how much is at stake, the account ratio how traders are
# positioned. All three are positions and prices. Live.
# --------------------------------------------------------------------------

# Coins priced below a cent trade as a 1000-unit contract.
_FUTURES_PREFIX = {"PEPE": "1000", "SHIB": "1000"}


def _crypto_derivatives(asset, today=None):
    from config import ASSET_TYPES

    if asset not in ASSET_TYPES.get("CRYPTO", []):
        return {"note": "not a crypto asset"}
    sym = "%s%sUSDT" % (_FUTURES_PREFIX.get(asset, ""), asset)
    base = "https://fapi.binance.com"
    funding = json.loads(_get("%s/fapi/v1/fundingRate?symbol=%s&limit=9" % (base, sym)))
    if not isinstance(funding, list) or not funding:
        return {"note": "no Binance perpetual for %s" % sym}
    oi = json.loads(_get("%s/futures/data/openInterestHist?symbol=%s&period=1d&limit=30"
                         % (base, sym)))
    ls = json.loads(_get("%s/futures/data/globalLongShortAccountRatio?symbol=%s"
                         "&period=1d&limit=7" % (base, sym)))
    oi_vals = [float(r["sumOpenInterestValue"]) for r in oi if isinstance(r, dict)]
    return {
        "contract": sym,
        "funding_last_3d": [float(r["fundingRate"]) for r in funding],
        "open_interest_usd": oi_vals[-1] if oi_vals else None,
        "open_interest_chg_7d": (round(oi_vals[-1] / oi_vals[-8] - 1, 4)
                                 if len(oi_vals) >= 8 and oi_vals[-8] else None),
        "open_interest_chg_30d": (round(oi_vals[-1] / oi_vals[0] - 1, 4)
                                  if len(oi_vals) >= 2 and oi_vals[0] else None),
        "long_short_account_ratio_7d": [float(r["longShortRatio"]) for r in ls
                                        if isinstance(r, dict)],
        "note": "funding > 0: longs pay shorts, i.e. leverage leans long.",
    }


register(Tool(
    name="crypto_derivatives",
    args={},
    rewinds=False,
    describe="Binance perpetuals: funding, open interest and its change, long/short account ratio",
    run=_crypto_derivatives, applies=is_crypto))


# --------------------------------------------------------------------------
# attention: daily Wikipedia page views of the company's article. How many
# people went looking, which is a count, not a view. Dated, so it rewinds.
# --------------------------------------------------------------------------

def _attention(asset, today=None, article=""):
    import news_analyzer

    title = str(article).strip() or news_analyzer.SEARCH_NAMES.get(asset, asset)
    # The pageviews API does not follow redirects, and a redirect page is a
    # few dozen stray visits: "NVIDIA" read 48 a day, the article 40 thousand.
    q = json.loads(_get("https://en.wikipedia.org/w/api.php?action=query&redirects=1"
                        "&format=json&titles=" + title.replace(" ", "_")))
    pages = list(((q.get("query") or {}).get("pages") or {}).values())
    if not pages or "missing" in pages[0]:
        return {"article": title, "note": "no English Wikipedia article by that title"}
    title = pages[0]["title"].replace(" ", "_")
    end = (datetime.date.fromisoformat(str(today)[:10]) if today
           else datetime.date.today())
    start = end - datetime.timedelta(days=90)
    raw = _get("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               "en.wikipedia/all-access/user/%s/daily/%s/%s"
               % (title, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")))
    views = [int(i["views"]) for i in json.loads(raw).get("items", [])]
    if len(views) < 14:
        return {"article": title, "note": "too little history"}
    base = sum(views[:-7]) / len(views[:-7])
    last7 = sum(views[-7:]) / 7
    return {"article": title, "views_last_7d_avg": round(last7),
            "views_prior_avg": round(base),
            "last_7d_vs_prior": round(last7 / base, 3) if base else None,
            "max_day_vs_prior": round(max(views[-7:]) / base, 3) if base else None}


register(Tool(
    name="attention",
    args={"article": "English Wikipedia article title, optional"},
    rewinds=True,
    describe="daily Wikipedia page views of the company, last week against the prior quarter",
    run=_attention))


def max_calls():
    """The per-judgment budget. 0 disables tools without touching the code."""
    try:
        return max(0, int(os.getenv("GTRADE_ANALYST_TOOL_CALLS", MAX_CALLS)))
    except ValueError:
        return MAX_CALLS


def require_first():
    """Whether the first reply must ask for evidence. A 26b model left to
    choose asked for nothing on SBER (2026-09-24) and judged from the dossier
    alone; the owner wants an agent that goes and looks."""
    return (os.getenv("GTRADE_ANALYST_REQUIRE_TOOL") or "1").strip() != "0"


def max_rounds():
    """Round trips spent asking. Each is another full model call."""
    try:
        return max(0, int(os.getenv("GTRADE_ANALYST_TOOL_ROUNDS", MAX_ROUNDS)))
    except ValueError:
        return MAX_ROUNDS
