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


def prompt_for(dossier):
    """The judgment prompt. Carries the dossier and nothing else.

    Note what is absent: the ensemble's probability, signal, timing/shadow
    action - no channel of the model's own opinion reaches this prompt (see
    core/analyst/dossier.py's FORBIDDEN_KEYS, which is what actually
    enforces that). The words BUY and SELL DO appear, by design, whenever the
    dossier carries a guru_verdict: that is the guru council's own
    fundamentals opinion, a second, independent source, not the ensemble's.
    """
    return (
        "You are an independent market analyst. Below is everything known "
        "about one asset. Form your own view of the next trading day.\n\n"
        + json.dumps(dossier, indent=2, ensure_ascii=True)
        + "\n\nReturn STRICT JSON, no prose, with exactly these keys:\n"
          '{"direction": "up|down|flat", "conviction": 1-5, '
          '"vol_regime": "calm|normal|elevated", '
          '"key_risk": "one clause", "thesis": "two or three sentences", '
          '"evidence": ["names of the fields above you actually used"]}\n'
          "Do NOT return a price, a target, or a percentage. Conviction 1 "
          "means barely a lean; 5 means you would stake the account on it."
    )


def parse_judgment(text, allowed=None):
    """A validated judgment, or None. Never raises.

    `allowed`, when given, is the set of dossier field names the model may cite
    as evidence. An invented field name means the answer was not grounded in
    what it was shown. An empty evidence list is also rejected: a model that
    cannot name one field it used did not ground its answer either, and this
    is the only check that tells those two cases apart.
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
    if allowed is not None and not set(evidence) <= set(allowed):
        return None

    return {"direction": data["direction"],
            "conviction": int(data["conviction"]),
            "vol_regime": data["vol_regime"],
            "key_risk": str(data.get("key_risk") or "")[:200],
            "thesis": str(data.get("thesis") or "")[:600],
            "evidence": evidence}


def judge(dossier, call=None):
    """One judgment for one dossier, or None when the model will not produce one.

    Returning None rather than a default is the point: a fabricated neutral
    judgment would enter the log, get scored, and dilute every cell it touched.
    """
    if call is None:
        raise ValueError("judge() needs an injected call; see analyst.py")
    prompt = prompt_for(dossier)
    allowed = set(dossier)
    for _ in range(MAX_ATTEMPTS):
        try:
            answer = call(prompt)
        except Exception:
            continue
        parsed = parse_judgment(answer, allowed=allowed)
        if parsed is not None:
            return parsed
    return None
