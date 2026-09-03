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
  core/analyst/dossier.py, and the same reason _headlines carries raw titles
  and drops news_analyzer's sentiment score. `register` enforces it rather than
  trusting a reader to remember, because the tempting sources are exactly the
  easy ones to wire: Yahoo hands out recommendationMean in the same payload
  this project already fetches for P/E.

  The one standing exception is `guru_verdict`, which predates this and stays:
  it is the project's OWN council over fundamentals it holds, scored in
  guru_log, not an outside house's rating. The line is whether the project can
  check the opinion against outcomes, not whether it is an opinion.

Budget matters more here than anywhere else in the project: a local 26b model
answers in 9 to 25 minutes, so every extra round trip is another quarter hour.
MAX_CALLS is deliberately small and the operator can set it to zero.
"""

import datetime
import json
import os

MAX_CALLS = 2
TIMEOUT = 12

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
                 "buy/sell call", "estimate revision")


class OpinionSource(Exception):
    """Raised when a tool would hand the model a conclusion to borrow."""


class Tool:
    def __init__(self, name, args, rewinds, describe, run):
        self.name = name
        self.args = args           # {arg: "what it is"}
        self.rewinds = rewinds     # honours `today`, so a backfill may use it
        self.describe = describe
        self.run = run


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


def available(today=None):
    """The tools this run may use. A rewound run keeps only the dated ones."""
    return [t for t in _REGISTRY.values() if t.rewinds or today is None]


def spec_lines(today=None):
    """The tool menu as it appears in the prompt, or "" when there is none."""
    tools = available(today)
    if not tools:
        return ""
    lines = [
        "",
        ("You may ask for MORE evidence before deciding. To do that, "
         "return this instead of a judgment:"),
        '{"tool": "<name>", "args": {...}}',
        ("You get the result and are asked again. Each request costs real "
         "time, so ask only when the answer would change your direction, "
         "and never twice for the same thing. Available:"),
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

    resp = net.http_get(url, headers=headers or {}, timeout=TIMEOUT)
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
    run=_insider_filings))


# --------------------------------------------------------------------------
# news_search: the same feeds the dossier already reads, on a query the model
# chooses. Live only: an RSS feed has no archive, so a rewound run asking for
# a past date would silently receive today's news.
# --------------------------------------------------------------------------

def _news_search(asset, today=None, query="", limit=6):
    import news_analyzer

    term = str(query or asset).strip()[:80]
    items = news_analyzer.fetch_news(term, max_articles=limit * 2) or []
    out = []
    for it in items[:limit]:
        if isinstance(it, dict) and it.get("title"):
            out.append({"title": it["title"][:180],
                        "source": (it.get("source") or "unknown")[:40],
                        "credibility": it.get("credibility")})
    return {"query": term, "headlines": out}


register(Tool(
    name="news_search",
    args={"query": "what to search for, a few words"},
    rewinds=False,
    describe="search the project's news feeds for something specific",
    run=_news_search))


def max_calls():
    """The per-judgment budget. 0 disables tools without touching the code."""
    try:
        return max(0, int(os.getenv("GTRADE_ANALYST_TOOL_CALLS", MAX_CALLS)))
    except ValueError:
        return MAX_CALLS
