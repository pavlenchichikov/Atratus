"""core.analyst.agent: the judgment schema and its parser.

No test here reaches a provider. `judge` takes an injected call, which is also
how core/llm_proposer.py is exercised.
"""

import json

import pytest

from core.analyst import agent


def _dossier():
    # The real 15-key shape core/analyst/dossier.py's build() produces -
    # see test_the_dossier_shape_is_declared_and_any_new_field_must_be_too
    # in test_analyst_dossier.py, which pins it from the other side.
    return {"asset": "SBER", "date": "2026-01-20", "close": 100.0, "atr": 2.0,
            "atr_pct": 0.02, "ret_1": 0.001, "ret_5": 0.01, "ret_20": -0.03,
            "high_20": 104.0, "low_20": 96.0, "bars_available": 120,
            "guru_verdict": "BUY", "guru_pct": 62.0,
            "next_earnings": {"date": "2026-02-01", "confirmed": True},
            "macro_events": ["CPI"]}


def _valid():
    return {"direction": "up", "conviction": 3, "vol_regime": "elevated",
            "key_risk": "earnings in two days", "thesis": "Tape is firm.",
            "evidence": ["ret_20", "atr_pct"]}


def test_a_well_formed_answer_parses():
    assert agent.parse_judgment(json.dumps(_valid())) == _valid()


def test_json_wrapped_in_prose_still_parses():
    # Models add a preamble no matter how the prompt is worded. Failing on that
    # would throw away an otherwise usable judgment.
    text = "Here is my read:\n```json\n" + json.dumps(_valid()) + "\n```\nHope that helps."
    assert agent.parse_judgment(text) == _valid()


def test_an_unknown_direction_is_rejected():
    assert agent.parse_judgment(json.dumps({**_valid(), "direction": "sideways"})) is None


def test_a_conviction_outside_the_scale_is_rejected():
    assert agent.parse_judgment(json.dumps({**_valid(), "conviction": 9})) is None


def test_a_free_form_percentage_is_not_accepted_as_a_judgment():
    # The whole calibration design rests on the LLM emitting a discrete cell.
    # A percentage would be measurable and uncorrectable.
    assert agent.parse_judgment(json.dumps({"forecast_pct": 2.3})) is None


def test_garbage_returns_none_rather_than_raising():
    assert agent.parse_judgment("the market will go up, probably") is None
    assert agent.parse_judgment("") is None


def test_evidence_must_name_real_dossier_fields():
    bad = {**_valid(), "evidence": ["insider_flow"]}
    assert agent.parse_judgment(json.dumps(bad), allowed=set(_dossier())) is None


def test_empty_evidence_is_rejected():
    # Citing nothing must not be indistinguishable from citing real fields.
    bad = {**_valid(), "evidence": []}
    assert agent.parse_judgment(json.dumps(bad)) is None


def test_an_extraneous_key_is_dropped_not_rejected():
    # Deliberate: the return dict is an explicit key allow-list. Refusing an
    # otherwise-usable judgment because the model volunteered an extra field
    # would discard signal for cosmetics.
    extra = {**_valid(), "forecast_pct": 2.3}
    parsed = agent.parse_judgment(json.dumps(extra))
    assert parsed == _valid()
    assert "forecast_pct" not in parsed


def test_judge_retries_once_then_gives_up():
    calls = []

    def failing(prompt):
        calls.append(prompt)
        return "no json here"

    assert agent.judge(_dossier(), call=failing) is None
    assert len(calls) == 2


def test_judge_retries_after_call_raises():
    # A raised exception is the case retrying should help most (timeout,
    # transient hiccup); it must consume an attempt, not burn the budget.
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return json.dumps(_valid())

    assert agent.judge(_dossier(), call=flaky) == _valid()
    assert len(calls) == 2


def test_judge_returns_the_first_valid_answer():
    def flaky(prompt):
        return "junk" if not flaky.hit else json.dumps(_valid())
    flaky.hit = 0

    def call(prompt):
        out = flaky(prompt)
        flaky.hit += 1
        return out

    assert agent.judge(_dossier(), call=call) == _valid()


def test_prose_with_a_brace_after_the_json_does_not_break_parsing():
    # A greedy first-brace-to-last-brace match would swallow the aside below
    # as part of the JSON blob and fail to parse at all.
    text = json.dumps(_valid()) + "\nNote: not the {legacy format}."
    assert agent.parse_judgment(text) == _valid()


def test_two_json_objects_returns_the_first():
    # A greedy match spans both objects into one unparseable blob; the first
    # complete object is the one that should win.
    text = json.dumps(_valid()) + "\n" + json.dumps({"other": "object"})
    assert agent.parse_judgment(text) == _valid()


def test_the_prompt_never_contains_the_ensembles_opinion():
    # The dossier's guru_verdict is legitimately "BUY" here - the guru
    # council's fundamentals opinion is included by design (dossier.py's own
    # docstring says so). What must never reach the prompt is the ENSEMBLE's
    # channels: its probability and its emitted signal/timing/shadow actions.
    text = agent.prompt_for(_dossier()).lower()
    for banned in ("probability", "cb_prob", "lstm_prob", "meta_prob",
                   "timing_action", "shadow_action", "sig_shown"):
        assert banned not in text


def test_citing_an_empty_field_is_not_grounding():
    """macro_events was cited five times in the first 33 judgments while blank
    for every asset on every run, and the validator recorded it as evidence: it
    checked the field NAME against the dossier's keys, never the value.

    Dropped, not fatal, so one filler citation does not throw away a reasoned
    call. Citing nothing else IS fatal, and the last case is the positive
    control for that."""
    answer = ('{"direction": "down", "conviction": 2, "vol_regime": "calm",'
              ' "key_risk": "r", "thesis": "t",'
              ' "evidence": ["ret_20", "macro_events"]}')
    allowed = {"ret_20", "macro_events", "close"}
    j = agent.parse_judgment(answer, allowed=allowed, empty={"macro_events"})
    assert j["evidence"] == ["ret_20"]

    # a name that is in neither set was invented, and still fails outright
    bad = answer.replace("macro_events", "confidence")
    assert agent.parse_judgment(bad, allowed=allowed,
                                empty={"macro_events"}) is None

    only_empty = ('{"direction": "down", "conviction": 2, "vol_regime": "calm",'
                  ' "key_risk": "r", "thesis": "t",'
                  ' "evidence": ["macro_events"]}')
    assert agent.parse_judgment(only_empty, allowed=allowed,
                                empty={"macro_events"}) is None


def test_judge_marks_the_empty_dossier_fields_for_the_validator():
    """The seam between the two: judge knows the VALUES, parse_judgment only
    ever sees names. A None or [] field must reach it as empty."""
    seen = {}

    def fake_call(prompt):
        return ('{"direction": "up", "conviction": 3, "vol_regime": "calm",'
                ' "key_risk": "r", "thesis": "t",'
                ' "evidence": ["ret_20", "headlines", "next_earnings"]}')

    d = {"asset": "SBER", "ret_20": -0.05, "headlines": [],
         "next_earnings": None, "close": 268.9}
    j = agent.judge(d, call=fake_call, depth="brief")
    assert j["evidence"] == ["ret_20"], seen


def test_a_missing_provider_is_not_a_refusal():
    """The 2026-08-31 run: provider anthropic, the SDK absent, two identical
    attempts, and a report of refused=1. A refusal means the model would not
    answer; here nothing was ever asked, and the two readings deserve opposite
    handling.

    The second half is the positive control: an ordinary failure must still be
    retried and still end as a refusal, or this would have turned every hiccup
    into a hard stop."""
    from core.llm_proposer import ProviderUnavailable

    tries = []

    def missing(_prompt):
        tries.append(1)
        raise ProviderUnavailable("the anthropic provider needs the anthropic "
                                  "package: pip install anthropic")

    with pytest.raises(ProviderUnavailable):
        agent.judge({"asset": "SBER", "close": 1.0}, call=missing, depth="brief")
    assert len(tries) == 1, "a missing package was retried"

    flaky = []

    def broken(_prompt):
        flaky.append(1)
        raise RuntimeError("the model returned nothing")

    assert agent.judge({"asset": "SBER", "close": 1.0}, call=broken,
                       depth="brief") is None
    assert len(flaky) == agent.MAX_ATTEMPTS


def test_a_rejection_says_which_check_failed():
    """A count of discarded calls does not tell you what to fix; the reason
    does. A local model failing on conviction is a prompt problem, failing on
    evidence is a dossier problem, and they cost the same nine minutes."""
    bad = _valid() | {"conviction": 2.5}
    why = []
    assert agent.parse_judgment(json.dumps(bad), why=why) is None
    assert "conviction" in why[0]

    why = []
    assert agent.parse_judgment("no json here at all", why=why) is None
    assert "JSON" in why[0]

    why = []
    bad = _valid() | {"evidence": ["moon_phase"]}
    assert agent.parse_judgment(json.dumps(bad), allowed=set(_dossier()),
                                why=why) is None
    assert "moon_phase" in why[0]

    # positive control: a good answer appends nothing
    why = []
    assert agent.parse_judgment(json.dumps(_valid()), why=why) is not None
    assert why == []


def test_a_retry_that_succeeds_is_still_reported():
    """The run that prompted this made three calls for two assets and finished
    saying `written=2 skipped=0 refused=0`. Sixteen minutes of local inference
    were discarded with nothing anywhere recording it."""
    replies = ["{not json}", json.dumps(_valid())]
    seen = []
    out = agent.judge(_dossier(), call=lambda p: replies.pop(0),
                      on_reject=seen.append)
    assert out["direction"] == "up", "the second attempt still wins"
    assert len(seen) == 1 and "JSON" in seen[0]


def test_a_provider_that_raises_is_reported_as_such():
    def boom(prompt):
        raise RuntimeError("connection reset")

    seen = []
    assert agent.judge(_dossier(), call=boom, on_reject=seen.append) is None
    assert len(seen) == agent.MAX_ATTEMPTS
    assert "connection reset" in seen[0]


def test_model_typography_is_normalised_out_of_the_prose():
    """The thesis reaches three consumers - the console, the store and the web
    card - so it is cleaned where it is parsed rather than at each of them."""
    j = _valid() | {
        "thesis": "I move away from ‘down’—the tape turned…",
        "key_risk": "A break below “support”"}
    out = agent.parse_judgment(json.dumps(j))
    assert out["thesis"] == "I move away from 'down' - the tape turned..."
    assert out["key_risk"] == 'A break below "support"'


def test_the_sanitiser_leaves_russian_quotation_marks_alone():
    # Guillemets are punctuation in Russian, not typographic decoration.
    assert agent.plain("«ГАЗП»") == "«ГАЗП»"


def test_the_checklist_names_the_fields_it_expects_to_be_read():
    """The measured driver of what the model reads.

    Over the first 35 judgments, 16 of the 21 fields the checklist NAMED were
    cited, against 9 of the other 39 - mostly once each. News, breadth,
    cross-asset correlation, regime, sector and the whole flow block reached
    the prompt in the 2026-08-31 session and the checklist was never extended,
    so they were fetched, rendered and never read. This pins the blocks back
    to the instruction that earns them."""
    from core.analyst import dossier

    d = {k: 1.0 for block in dossier.BLOCKS.values() for k in block}
    for horizon in (1, 5):
        instructions = agent.prompt_for(
            d, depth="full", horizon=horizon).split("in order")[1]
        for field in ("headlines", "breadth_above_sma50_pct", "cross_asset_corr",
                      "vix_level", "regime_trend", "rsi_14", "sector_momentum",
                      "volume_vs_20", "turnover", "gap_open", "range_atr",
                      "ex_dividend_date", "market_cap"):
            assert field in instructions, "%s unnamed at horizon %d" % (
                field, horizon)


def test_the_brief_form_stays_brief():
    """The checklist is what makes the full form cost 3.4x the wall clock.
    Growing it must not quietly grow the sweep form too."""
    from core.analyst import dossier

    d = {k: 1.0 for block in dossier.BLOCKS.values() for k in block}
    brief = agent.prompt_for(d, depth="brief", horizon=1)
    full = agent.prompt_for(d, depth="full", horizon=1)
    assert len(brief) < len(full) / 2
