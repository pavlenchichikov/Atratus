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


def prompt_for(dossier, depth="full", horizon=1):
    """The judgment prompt. Carries the dossier and nothing else.

    `depth` buys substance with time, and on a local model the exchange rate is
    steep: measured on gemma4:12b over one asset, the brief form answered in
    637s with a 383-character thesis and the full form in 2189s with a
    1044-character one. That is 3.4 times the wall clock for 2.7 times the
    reasoning, which is worth it for one asset a person asked about and not
    worth it for a sweep of the whole watchlist.

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
             "is a value-investing council and pe, roe and div_yield are "
             "slow facts. Over %s they are part of the case rather than "
             "decoration, so weigh them instead of waving them off. "
             "next_earnings and macro_events matter if they fall inside "
             "the window.\n" % _span(horizon)
             if int(horizon) > 1 else
             "4. What the fundamentals and the calendar add, if anything. "
             "guru_verdict is a value-investing council, a slow signal; "
             "say plainly when it is irrelevant to a one-day view rather "
             "than citing it for the sake of it. next_earnings and "
             "macro_events matter only if they are close.\n") +
          "5. Your own record here. past_calls, past_hit_rate and "
          "past_last_call are YOUR previous judgments on this asset and how "
          "they resolved. If you were recently wrong in the direction you are "
          "about to choose again, say so and justify repeating it.\n"
          "6. What would change your mind, concretely: a level, a move, or an "
          "event, not a vague condition.\n"
          "Quote the numbers you use. Name what you are deliberately NOT "
          "leaning on where a reader might assume you did. If the evidence is "
          "thin, say the case is thin and pick conviction 1 or 2 rather than "
          "dressing up a guess.")
    )


def parse_judgment(text, allowed=None, empty=()):
    """A validated judgment, or None. Never raises.

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
        return None
    data = _first_json_object(text)
    if not isinstance(data, dict):
        return None

    if data.get("direction") not in DIRECTIONS:
        return None
    if data.get("conviction") not in CONVICTIONS:
        return None
    if data.get("vol_regime") not in VOL_REGIMES:
        return None
    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(
            isinstance(e, str) for e in evidence):
        return None
    if allowed is not None:
        if not set(evidence) <= set(allowed) | set(empty):
            return None                      # a field name that does not exist
        evidence = [e for e in evidence if e not in set(empty)]
        if not evidence:
            return None                      # cited only fields that were empty

    return {"direction": data["direction"],
            "conviction": int(data["conviction"]),
            "vol_regime": data["vol_regime"],
            "key_risk": str(data.get("key_risk") or "")[:400],
            "thesis": str(data.get("thesis") or "")[:2500],
            "evidence": evidence}


def judge(dossier, call=None, depth="full", horizon=1):
    """One judgment for one dossier, or None when the model will not produce one.

    Returning None rather than a default is the point: a fabricated neutral
    judgment would enter the log, get scored, and dilute every cell it touched.
    """
    if call is None:
        raise ValueError("judge() needs an injected call; see analyst.py")
    prompt = prompt_for(dossier, depth=depth, horizon=horizon)
    allowed = set(dossier)
    # A field the dossier carries as None or [] was shown to the model as empty,
    # so citing it is not evidence of anything.
    empty = {k for k, v in dossier.items() if v is None or v == []}
    for _ in range(MAX_ATTEMPTS):
        try:
            answer = call(prompt)
        except Exception:
            continue
        parsed = parse_judgment(answer, allowed=allowed, empty=empty)
        if parsed is not None:
            return parsed
    return None
