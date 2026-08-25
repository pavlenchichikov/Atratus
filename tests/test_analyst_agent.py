"""core.analyst.agent: the judgment schema and its parser.

No test here reaches a provider. `judge` takes an injected call, which is also
how core/llm_proposer.py is exercised.
"""

import json

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
