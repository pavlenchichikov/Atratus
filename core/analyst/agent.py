"""The LLM half of the analyst: dossier in, discrete judgment out.

The model never returns a percentage. It returns a cell of a small grid, and
core/analyst/calibrate.py turns that cell into a number using what the cell has
historically been worth. That is the only arrangement in which the analyst can
be corrected rather than merely scored: a free-form percentage is a different
answer every time and no bucket ever accumulates enough history to recalibrate.

Provider handling is deliberately absent. The caller injects `call`, exactly as
core/llm_proposer.py keeps its SDK imports inside its call functions, so this
module imports cleanly with no SDK present and every test runs offline.
"""

import json
import re

DIRECTIONS = ("up", "down", "flat")
CONVICTIONS = (1, 2, 3, 4, 5)
VOL_REGIMES = ("calm", "normal", "elevated")

MAX_ATTEMPTS = 2


def _first_json_object(text):
    """The first complete JSON object in a reply, or None.

    Not a regex. A greedy `\\{.*\\}` spans from the first brace to the last,
    so a model that appends an aside containing a brace, or returns two
    objects, produces one unparseable blob and its judgment is thrown away.
    raw_decode stops at the end of the first object and does not care what
    follows it.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


BRIEF_TAIL = (
    "Do NOT return a price, a target, or a percentage. Conviction 1 means "
    "barely a lean; 5 means you would stake the account on it. In the thesis, "
    "quote the numbers you actually used and say what would change your mind."
)


def _span(horizon):
    return ("the next trading day" if int(horizon) == 1
            else "the next %d trading days" % int(horizon))


def prompt_for(dossier, depth="full", horizon=1, tool_menu=""):
    """The judgment prompt. Carries the dossier and nothing else.

    `depth` buys substance with time, and on a local model the exchange rate is
    steep: measured on gemma4:12b over one asset, the brief form answered in
    637s with a 383-character thesis and the full form in 2189s with a
    1044-character one. That is 3.4 times the wall clock for 2.7 times the
    reasoning, which is worth it for one asset a person asked about and not
    worth it for a sweep of the whole watchlist.

    The numbered checklist is not decoration, it is what the model actually
    reads. Over the first 35 judgments, 16 of the 21 fields the checklist NAMES
    were cited as evidence, against 9 of the other 39 - and those nine mostly
    once each. The dossier grew by news, breadth, cross-asset correlation,
    regime, RSI, sector and the whole flow block in the 2026-08-31 session and
    this checklist did not, so 33 fields were being fetched over the network,
    rendered into the prompt and never read. A field nobody is told to use is a
    field nobody uses.

    Note what is absent: the ensemble's probability, signal, timing/shadow
    action - no channel of the model's own opinion reaches this prompt (see
    core/analyst/dossier.py's FORBIDDEN_KEYS, which is what actually
    enforces that). The words BUY and SELL DO appear, by design, whenever the
    dossier carries a guru_verdict: that is the guru council's own
    fundamentals opinion, a second, independent source, not the ensemble's.
    """
    return (
        "You are an independent market analyst. Below is everything known "
        "about one asset. Form your own view of " + _span(horizon) + ".\n\n"
        + json.dumps(dossier, indent=2, ensure_ascii=True)
        + "\n\nReturn STRICT JSON, no prose, with exactly these keys:\n"
          '{"direction": "up|down|flat", "conviction": 1-5, '
          '"vol_regime": "calm|normal|elevated", '
          '"key_risk": "the one thing most likely to make this wrong", '
          '"thesis": "six to ten sentences", '
          '"evidence": ["names of the fields above you actually used"]}\n'
        + (BRIEF_TAIL if depth == "brief" else
          "Do NOT return a price, a target, or a percentage. Conviction 1 "
          "means barely a lean; 5 means you would stake the account on it.\n"
          "The thesis is the part a person can argue with, so make it worth "
          "arguing with. Work through these in order, in prose rather than as "
          "a list:\n"
          "1. Where the price sits. Use atr_to_high_20 and atr_to_low_20, "
          "which are distances in this asset's own units of movement, and "
          "drawdown_60. Say whether the price is stretched or mid-range.\n"
          "2. What the move has been. Compare ret_5, ret_20 and ret_60 and say "
          "whether they agree. A short-term bounce inside a long decline is a "
          "different situation from a steady trend, and streak_days tells you "
          "which.\n"
          "3. What the volatility says. vol_20_vs_60 is current volatility "
          "against this asset's own recent norm, so a value near 1 means "
          "ordinary conditions for THIS asset regardless of the absolute "
          "number.\n"
          + ("4. What the fundamentals and the calendar add. guru_verdict "
             "is a value-investing council and pe, roe, div_yield, "
             "debt_ebitda, market_cap, beta and short_ratio are slow facts. "
             "Over %s they are part of the case rather than decoration, so "
             "weigh them instead of waving them off. ex_dividend_date, "
             "next_earnings and macro_events matter if they fall inside the "
             "window.\n" % _span(horizon)
             if int(horizon) > 1 else
             "4. What the fundamentals and the calendar add, if anything. "
             "guru_verdict is a value-investing council and pe, roe, "
             "div_yield, debt_ebitda, market_cap, beta and short_ratio are "
             "slow facts; say plainly when they are irrelevant to a one-day "
             "view rather than citing them for the sake of it. Two are NOT "
             "slow: ex_dividend_date, because a dividend gap looks exactly "
             "like a fall and is not one, and next_earnings. macro_events "
             "matters only if it is close.\n") +
          "5. The news. headlines is the actual wire, not a sentiment score. "
          "Read it. If it says something the numbers do not, say so; if it is "
          "stale, generic, or about a different part of the business, say "
          "that instead of passing over it in silence.\n"
          "6. Whether it moved alone. benchmark_ret_1 and benchmark_ret_20 "
          "are its index, corr_to_benchmark_60 how tightly it usually "
          "follows, breadth_above_sma50_pct and breadth_positive_20d_pct how "
          "much of the whole market is holding up, cross_asset_corr whether "
          "everything is moving together, and vix_level with vix_chg_20 the "
          "price of fear. A fall everything shared is a different fact from a "
          "fall this asset had on its own.\n"
          "7. Its regime and its sector. regime_trend, regime_vol and "
          "regime_momentum are this asset classified from its own prices; "
          "rsi_14 and atr_vs_90d put today against a longer norm than "
          "vol_20_vs_60 does; sector_momentum and sector_trend say whether "
          "its neighbours are doing the same thing.\n"
          "8. The year behind it. ret_1y, ret_ytd and vol_1y are the "
          "scale ret_60 stops short of, max_dd_1y is the worst it has been "
          "down over that year, off_52w_high where it sits against its own "
          "high, and excess_vs_benchmark_1y with beta_1y say whether it beat "
          "its index and how much of its move was the index in the first "
          "place. A name down 30% inside an index down 25% is a different "
          "case from one down 30% on its own.\n"
          "9. What actually traded. volume_vs_20 and turnover say whether "
          "anyone was there for the move, gap_open how much of it happened "
          "before the session opened, range_atr how much of its own daily "
          "range the day used. A move on no volume is a different move.\n"
          "10. Your own record here. past_calls, past_hit_rate and "
          "past_last_call are YOUR previous judgments on this asset and how "
          "they resolved. If you were recently wrong in the direction you are "
          "about to choose again, say so and justify repeating it.\n"
          "11. What would change your mind, concretely: a level, a move, or "
          "an event, not a vague condition.\n"
          "Quote the numbers you use. Name what you are deliberately NOT "
          "leaning on where a reader might assume you did. If the evidence is "
          "thin, say the case is thin and pick conviction 1 or 2 rather than "
          "dressing up a guess.")
        # Last, so the schema above is what the model reads immediately before
        # answering, and the menu is an aside rather than the instruction.
        + tool_menu
    )


# Escapes rather than the characters themselves: a non-breaking space is
# invisible in source, and a reviewer cannot check a table they cannot see.
_TYPOGRAPHY = {
    "\u2018": "'", "\u2019": "'",    # curly single quotes
    "\u201c": '"', "\u201d": '"',    # curly double quotes
    "\u00a0": " ",                   # non-breaking space
    "\u2026": "...",                 # ellipsis
}
_DASH = re.compile(r"\s*[\u2014\u2013]\s*")   # em dash, en dash


def plain(text):
    """Model prose with the typography this project does not ship.

    Em and en dashes become " - ", curly quotes become straight ones. Guillemets
    are left alone: they are Russian punctuation, not decoration.

    Applied to the free text the model writes rather than at the database or at
    the console, because there are three consumers - the run's own printout, the
    web card and the store - and sanitising at each of them is how one of them
    ends up forgotten.
    """
    if not text:
        return text
    text = _DASH.sub(" - ", text)
    for bad, good in _TYPOGRAPHY.items():
        text = text.replace(bad, good)
    return text.strip()


def _no(why, reason):
    """Record why an answer was thrown away, and throw it away.

    Falls off the end rather than returning None explicitly, which ruff asks
    for and which is the same thing: every caller writes `return _no(...)` and
    propagates that None as the rejection.
    """
    if why is not None:
        why.append(reason)


def parse_judgment(text, allowed=None, empty=(), why=None):
    """A validated judgment, or None. Never raises.

    `why`, when a list is passed, collects one short line naming the check that
    failed. Nothing here changes for a caller that does not pass it; the reason
    exists because a local model can fail this parse three times in an hour and
    the fix depends entirely on WHICH check it failed.

    `allowed`, when given, is the set of dossier field names the model may cite
    as evidence; `empty` is the subset of those that carried no value. An
    invented field name means the answer was not grounded in what it was shown.
    An empty evidence list is also rejected: a model that cannot name one field
    it used did not ground its answer either, and this is the only check that
    tells those two cases apart.

    A cited field that was present but EMPTY is dropped from the evidence rather
    than failing the whole judgment. The check used to be against the key set
    alone, so macro_events - blank for every asset on every run because the
    calendar file does not exist - was cited five times in the first 33
    judgments and recorded as grounding. One filler citation is not a reason to
    throw away an otherwise reasoned call; citing nothing else is, and that
    falls through to the empty-evidence rejection above.
    """
    if not text:
        return _no(why, "the reply was empty")
    data = _first_json_object(text)
    if not isinstance(data, dict):
        return _no(why, "no JSON object in the reply")

    if data.get("direction") not in DIRECTIONS:
        return _no(why, "direction=%r is not up/down/flat" % data.get("direction"))
    if data.get("conviction") not in CONVICTIONS:
        return _no(why, "conviction=%r is not an integer 1-5"
                   % data.get("conviction"))
    if data.get("vol_regime") not in VOL_REGIMES:
        return _no(why, "vol_regime=%r is not calm/normal/elevated"
                   % data.get("vol_regime"))
    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(
            isinstance(e, str) for e in evidence):
        return _no(why, "evidence is not a non-empty list of field names")
    if allowed is not None:
        invented = sorted(set(evidence) - set(allowed) - set(empty))
        if invented:                         # a field name that does not exist
            return _no(why, "evidence cites fields the dossier does not "
                       "have: %s" % ", ".join(invented[:4]))
        evidence = [e for e in evidence if e not in set(empty)]
        if not evidence:                     # cited only fields that were empty
            return _no(why, "evidence cited only fields that were blank")

    return {"direction": data["direction"],
            "conviction": int(data["conviction"]),
            "vol_regime": data["vol_regime"],
            "key_risk": plain(str(data.get("key_risk") or ""))[:400],
            "thesis": plain(str(data.get("thesis") or ""))[:2500],
            "evidence": evidence}


def judge(dossier, call=None, depth="full", horizon=1, on_reject=None,
          today=None, tool_calls=None):
    """One judgment for one dossier, or None when the model will not produce one.

    Returning None rather than a default is the point: a fabricated neutral
    judgment would enter the log, get scored, and dilute every cell it touched.

    `on_reject`, when given, is called once per discarded attempt with a short
    reason. A retry is otherwise invisible: on 2026-09-03 a run of two assets
    made three calls and finished saying `written=2 skipped=0 refused=0`, with
    sixteen minutes of local inference thrown away and nothing anywhere saying
    so. A retry that succeeds still cost the money.

    `tool_calls`, when a list is passed, collects every source the model asked
    for and what came back, so the judgment stays replayable. `today` is the
    rewind date and reaches core/analyst/tools.py, which drops any tool that
    cannot honour it - otherwise a backfill of May would be answered with
    September's filings.

    The tool budget is per JUDGMENT, not per attempt, and it is spent before
    the parse retries: a model that asks for two things and then returns
    unparseable JSON has cost four calls, and on a local 26b that is an hour.
    """
    from core.analyst import tools

    budget = tools.max_calls() if tool_calls is not None else 0
    extra = ""
    if call is None:
        raise ValueError("judge() needs an injected call; see analyst.py")
    prompt = prompt_for(dossier, depth=depth, horizon=horizon,
                        tool_menu=tools.spec_lines(today) if budget else "")
    allowed = set(dossier)
    # A field the dossier carries as None or [] was shown to the model as empty,
    # so citing it is not evidence of anything.
    empty = {k for k, v in dossier.items() if v is None or v == []}
    from core.llm_proposer import ProviderUnavailable

    for _ in range(MAX_ATTEMPTS + budget):
        try:
            answer = call(prompt + extra)
        except ProviderUnavailable:
            # Not a refusal and not worth a second attempt: the SDK will be
            # just as absent next time. Let it out so the caller can say what
            # is missing instead of reporting a model that would not answer.
            raise
        except Exception as exc:
            if on_reject is not None:
                on_reject("the call itself failed: %s" % exc)
            continue
        # A request for evidence is not a failed judgment, so it is checked
        # first and does not spend a parse attempt.
        if budget > 0:
            request = tools.parse_request(_first_json_object(answer))
            if request is not None:
                budget -= 1
                entry = tools.call(request, asset=dossier.get("asset"),
                                   today=today)
                tool_calls.append(entry)
                extra += ("\n\nYou asked for %s and received:\n%s\n"
                          "Now return the judgment JSON."
                          % (request["tool"],
                             json.dumps(entry.get("result", entry.get("error")),
                                        ensure_ascii=True)[:1800]))
                continue

        why = []
        parsed = parse_judgment(answer, allowed=allowed, empty=empty, why=why)
        if parsed is not None:
            return parsed
        if on_reject is not None:
            on_reject(why[0] if why else "unparseable")
    return None
