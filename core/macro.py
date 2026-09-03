"""Fetch the macro calendar instead of asking somebody to type it.

`macro_calendar.json` shipped only as an example, so `macro_events` was blank
for every asset on every run since the analyst started - cited five times as
grounding while empty, which is what made the empty-evidence check necessary in
core/analyst/agent.py. The reason it stayed blank is that nobody should invent
central bank meeting dates, and a hand-maintained file is a chore that gets
skipped.

Both banks publish their own schedule, so nothing here is invented:

    CBR   https://www.cbr.ru/dkp/cal_mp/            (Russian, key rate)
    FOMC  federalreserve.gov/monetarypolicy/...     (US, rate decision)

Two rules make this safe to run repeatedly:

  A source that fails contributes NOTHING rather than a guess. `fetch` returns
  what it got and names what it did not, and the CLI prints both. A calendar
  half-filled by a dead connection must not look like a calendar.

  Hand-written entries SURVIVE. The file was designed to be maintained by hand
  and may hold events these two sources do not carry - an OPEC meeting, an
  election. `merge` keeps anything it did not fetch, so running this is never
  destructive.

Requests go through net.http_get, which owns the VPN routing: cbr.ru answers
over a Russian exit and the Fed over a foreign one, and which is which changes
when the VPN does.
"""

import datetime
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE_DIR, "macro_calendar.json")

CBR_URL = "https://www.cbr.ru/dkp/cal_mp/"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

_RU_MONTHS = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5,
              "июн": 6, "июл": 7, "август": 8, "сентябр": 9, "октябр": 10,
              "ноябр": 11, "декабр": 12}
_EN_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def _get(url):
    import net

    resp = net.http_get(url, timeout=20)
    if resp is None or resp.status_code != 200:
        raise OSError("%s answered %s" % (
            url, "nothing" if resp is None else resp.status_code))
    return resp.text


def _strip(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def cbr_events(html):
    """Key-rate decisions and minutes from the Bank of Russia calendar.

    The two are separated because they are not the same event: a decision
    moves the rate, a `Резюме обсуждения` explains one taken six weeks ago.
    Folded together, an analyst reads a minutes publication as a rate risk.
    """
    text = html.replace("&nbsp;", " ")
    out = []
    for block in re.split(r'class="date[^"]*">', text)[1:]:
        head = re.match(r"\s*(\d{1,2})\s+([А-Яа-яё]+)\s+(20\d\d)", block)
        if not head:
            continue
        month = next((v for k, v in _RU_MONTHS.items()
                      if head.group(2).lower().startswith(k)), None)
        if not month:
            continue
        body = _strip(block[:1200])
        decision = "Заседание Совета директоров" in body
        if not decision and "Резюме обсуждения" not in body:
            continue
        out.append({
            "date": "%s-%02d-%02d" % (head.group(3), month, int(head.group(1))),
            "name": ("CBR key rate decision" if decision
                     else "CBR key rate minutes"),
            "importance": "high" if decision else "medium",
            "region": "RU", "source": "cbr.ru"})
    return out


def fomc_events(html):
    """FOMC meetings. A two-day meeting is dated on the day it DECIDES.

    The page writes the span ("27-28") and the statement lands on the second
    day, so filing it under the first would put the event a day early - inside
    a one-day horizon that is the whole event.

    Scanned in document order rather than split into blocks. Splitting looked
    tidier and silently kept one meeting in three: a chunk that happened to
    contain two rows was searched once, and 27 of the page's 57 meetings came
    out. The year comes from the section heading the row falls under, which is
    why order matters and a per-row regex alone would not do.
    """
    pattern = re.compile(
        r'(?P<year>20\d\d) FOMC Meetings'
        r'|fomc-meeting__month[^>]*>\s*<strong>(?P<month>[A-Za-z]+)'
        r'|fomc-meeting__date[^>]*>\s*(?P<days>[\d\-/ ]+)')
    out, year, month = [], None, None
    for m in pattern.finditer(html):
        if m.group("year"):
            year, month = m.group("year"), None
            continue
        if m.group("month"):
            month = _EN_MONTHS.get(m.group("month").strip().lower())
            continue
        days = re.findall(r"\d{1,2}", m.group("days") or "")
        if not (year and month and days):
            continue
        # A meeting spanning a month boundary ("29-1") decides in the NEXT
        # month, so a falling day number rolls the month forward.
        last, first = int(days[-1]), int(days[0])
        stamp_month, stamp_year = month, int(year)
        if last < first:
            stamp_month = 1 if month == 12 else month + 1
            stamp_year += 1 if month == 12 else 0
        out.append({"date": "%d-%02d-%02d" % (stamp_year, stamp_month, last),
                    "name": "FOMC rate decision", "importance": "high",
                    "region": "US", "source": "federalreserve.gov"})
        month = None
    return out


SOURCES = (("CBR", CBR_URL, cbr_events), ("FOMC", FOMC_URL, fomc_events))


def fetch(sources=SOURCES):
    """({events}, {source: error}). A dead source contributes nothing."""
    events, failed = [], {}
    for name, url, parse in sources:
        try:
            got = parse(_get(url))
        except Exception as exc:
            failed[name] = "%s: %s" % (type(exc).__name__, exc)
            continue
        if not got:
            failed[name] = "reachable but nothing parsed; the page layout "\
                           "probably changed"
            continue
        events.extend(got)
    return events, failed


def _key(event):
    return (str(event.get("date"))[:10], str(event.get("name", "")).strip())


def merge(existing, fetched):
    """Fetched events on top of hand-written ones, newest data winning a tie.

    Anything in the file that the sources did not return is KEPT: the calendar
    was designed to be hand-maintained and may carry an OPEC meeting or an
    election that neither central bank publishes.
    """
    out = {_key(e): e for e in existing if _key(e)[0] != "None"}
    for event in fetched:
        out[_key(event)] = event
    return sorted(out.values(), key=lambda e: (e["date"], e.get("name", "")))


def load(path=None):
    try:
        with open(path or PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save(events, path=None):
    with open(path or PATH, "w", encoding="utf-8") as fh:
        json.dump(events, fh, ensure_ascii=False, indent=2)
    return len(events)


def upcoming(events, today=None, days=90):
    today = today or datetime.date.today().isoformat()
    until = (datetime.date.fromisoformat(str(today)[:10])
             + datetime.timedelta(days=days)).isoformat()
    return [e for e in events if today <= str(e.get("date", ""))[:10] <= until]
