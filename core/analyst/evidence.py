"""What counts as raw evidence. Code decides, not the model.

A judgment may cite a dossier field by name (checked in agent.parse_judgment) or
an outside item {source, kind, value, asof}: a URL it read or a tool it called.
Outside items are kept only when they are material (data, a filing, a
statistic, a price, or news within a publisher spread), never somebody else's
conclusion: rating and forecast sites are refused by domain, verdict wording by
OPINION_WORDS, and one outlet cannot fill the news block.
"""

import datetime
import re
from urllib.parse import urlparse

KINDS = ("data", "filing", "statistic", "price", "news")
NEWS_PER_PUBLISHER = 2
NEWS_MAX_DAYS = 90

# Sites whose product is a conclusion: ratings, targets, forecasts, ideas.
# A subdomain of any of these is the same site.
OPINION_DOMAINS = (
    "tipranks.com", "marketbeat.com", "seekingalpha.com", "zacks.com",
    "simplywall.st", "walletinvestor.com", "gov.capital", "stockinvest.us",
    "macroaxis.com", "wallstreetzen.com", "fool.com", "benzinga.com",
    "coincodex.com", "longforecast.com", "30rates.com", "stockanalysis.com/forecast",
    "tradingview.com/ideas", "investing.com/analysis", "bcs-express.ru",
    "finam.ru/analysis", "smart-lab.ru/blog", "pulse.tbank.ru",
)


def _host_path(url):
    u = urlparse(url if "//" in url else "//" + url)
    host = (u.hostname or "").lower()
    for prefix in ("www.", "m."):
        host = host.removeprefix(prefix)
    return host, (u.path or "").lower()


def blocked(url):
    """True when the URL is on (or under) an opinion site."""
    host, path = _host_path(str(url))
    if not host:
        return False
    for entry in OPINION_DOMAINS:
        dom, _, sub = entry.partition("/")
        if (host == dom or host.endswith("." + dom)) and (
                not sub or path.startswith("/" + sub)):
            return True
    return False


def opinion_word(text):
    """The first OPINION_WORDS entry present as a whole word, or None."""
    from core.analyst.tools import OPINION_WORDS

    low = str(text).lower()
    return next((w for w in OPINION_WORDS
                 if re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(w), low)), None)


def _days_old(asof, today):
    try:
        a = datetime.date.fromisoformat(str(asof)[:10])
        t = datetime.date.fromisoformat(str(today)[:10])
    except ValueError:
        return None
    return (t - a).days


def check(items, called=(), today=None):
    """(kept, dropped) for outside evidence items; dropped entries carry `why`."""
    today = today or datetime.date.today().isoformat()
    kept, dropped, per_pub = [], [], {}

    def drop(item, why):
        dropped.append({**item, "why": why} if isinstance(item, dict)
                       else {"item": item, "why": why})

    for item in items:
        if not isinstance(item, dict):
            drop(item, "not an object")
            continue
        source = str(item.get("source") or "").strip()
        kind = item.get("kind")
        if kind not in KINDS:
            drop(item, "kind %r is not one of %s" % (kind, ", ".join(KINDS)))
            continue
        is_url = "." in source and "/" in source
        if not is_url and source not in called:
            drop(item, "source %r is neither a URL nor a tool you called" % source)
            continue
        if is_url and blocked(source):
            drop(item, "opinion site")
            continue
        word = opinion_word("%s %s" % (source, item.get("value") or ""))
        if word:
            drop(item, "names a conclusion (%r)" % word)
            continue
        if kind == "news":
            old = _days_old(item.get("asof"), today) if item.get("asof") else None
            if old is not None and old > NEWS_MAX_DAYS:
                drop(item, "news older than %d days" % NEWS_MAX_DAYS)
                continue
            pub = _host_path(source)[0] if is_url else source
            if per_pub.get(pub, 0) >= NEWS_PER_PUBLISHER:
                drop(item, "more than %d stories from %s" % (NEWS_PER_PUBLISHER, pub))
                continue
            per_pub[pub] = per_pub.get(pub, 0) + 1
        kept.append({k: item.get(k) for k in ("source", "kind", "value", "asof")
                     if item.get(k) is not None})
    return kept, dropped


def label(item):
    """A short hashable name for any evidence entry: a field name as it is, an
    outside item as kind:host (or kind:tool)."""
    if not isinstance(item, dict):
        return str(item)
    source = str(item.get("source") or "")
    host = _host_path(source)[0] if "/" in source else source
    return "%s:%s" % (item.get("kind") or "?", host or "?")
