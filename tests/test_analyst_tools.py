"""core/analyst/tools.py: sources the model asks for rather than is handed.

The dossier's whole guarantee is that a judgment can be rebuilt from what it
saw. Tools are the one thing that can break it, so most of what is pinned here
is a refusal rather than a feature.
"""
import json

import pytest

from core.analyst import agent, tools


def _dossier():
    return {"asset": "NVDA", "date": "2026-09-03", "close": 100.0, "atr": 2.0,
            "atr_pct": 0.02, "ret_20": -0.03}


def _judgment(**over):
    return {"direction": "up", "conviction": 3, "vol_regime": "calm",
            "key_risk": "earnings", "thesis": "Tape is firm.",
            "evidence": ["ret_20"], **over}


@pytest.fixture()
def budget(monkeypatch):
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "2")


def test_only_a_registered_tool_is_ever_called():
    """The registry is an allow-list, not a fetch-any-URL. Every result goes
    straight into a prompt, so a model that can be told which page to read is
    a model an attacker can steer through a headline."""
    assert tools.parse_request({"tool": "http_get",
                                "args": {"url": "http://evil/"}}) is None
    assert tools.parse_request({"tool": "news_search", "args": {}}) is not None
    # and an unknown argument is dropped rather than forwarded
    got = tools.parse_request({"tool": "news_search",
                               "args": {"query": "x", "url": "http://evil/"}})
    assert got["args"] == {"query": "x"}


def test_a_judgment_is_not_mistaken_for_a_request():
    assert tools.parse_request(_judgment()) is None
    assert tools.parse_request(None) is None
    assert tools.parse_request({"tool": "news_search", "args": "not a dict"}) is None


def test_a_rewound_run_can_only_use_a_tool_that_honours_the_date():
    """Otherwise backfilling May is answered with September's filings, which
    is the trap dossier._as_of exists to close, reopened from a new side."""
    live = {t.name for t in tools.available(today=None)}
    past = {t.name for t in tools.available(today="2026-05-01")}
    assert "news_search" in live, "an RSS feed has no archive"
    assert "news_search" not in past
    assert past, "at least one tool must survive, or a backfill has none"
    assert past <= live


def test_a_live_only_tool_asked_for_a_past_date_is_refused_not_answered():
    entry = tools.call({"tool": "news_search", "args": {"query": "x"}},
                       asset="NVDA", today="2026-05-01")
    assert "cannot answer for a past date" in entry["error"]
    assert "result" not in entry


def test_a_dead_source_is_a_recorded_result_rather_than_a_crash(monkeypatch):
    def boom(**kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(tools._REGISTRY["news_search"], "run", boom)
    entry = tools.call({"tool": "news_search", "args": {}}, asset="NVDA")
    assert "connection reset" in entry["error"]
    assert entry["tool"] == "news_search" and "at" in entry


def test_the_call_is_recorded_with_what_was_asked_not_only_what_came_back(
        monkeypatch, budget):
    """A log entry that says what arrived but not what was requested is not a
    replay, which is the same defect as a judgment nobody scored."""
    monkeypatch.setattr(tools._REGISTRY["news_search"], "run",
                        lambda **kw: {"headlines": ["a recall"]})
    replies = [json.dumps({"tool": "news_search", "args": {"query": "recall"}}),
               json.dumps(_judgment())]
    calls = []
    out = agent.judge(_dossier(), call=lambda p: replies.pop(0),
                      tool_calls=calls)
    assert out["direction"] == "up"
    assert len(calls) == 1
    assert calls[0]["args"] == {"query": "recall"}
    assert calls[0]["result"] == {"headlines": ["a recall"]}


def test_the_result_is_fed_back_so_the_second_answer_can_use_it(
        monkeypatch, budget):
    monkeypatch.setattr(tools._REGISTRY["news_search"], "run",
                        lambda **kw: {"headlines": ["a product recall"]})
    seen = []

    def call(prompt):
        seen.append(prompt)
        return (json.dumps({"tool": "news_search", "args": {"query": "x"}})
                if len(seen) == 1 else json.dumps(_judgment()))

    agent.judge(_dossier(), call=call, tool_calls=[])
    assert "product recall" in seen[1], "the answer never reached the model"
    assert "product recall" not in seen[0]


def test_the_budget_is_per_judgment_and_stops_an_endless_asking(
        monkeypatch, budget):
    """A model that keeps asking would run forever, and on a local 26b each
    round is another nine to twenty-five minutes."""
    monkeypatch.setattr(tools._REGISTRY["news_search"], "run",
                        lambda **kw: {"headlines": []})
    calls = []
    out = agent.judge(
        _dossier(),
        call=lambda p: json.dumps({"tool": "news_search", "args": {}}),
        tool_calls=calls)
    assert out is None, "it never produced a judgment"
    assert len(calls) == 2, "and it stopped at the budget"


def test_tools_are_off_unless_the_caller_asks_for_them(monkeypatch, budget):
    """A caller that passes no list gets the old behaviour exactly: no menu in
    the prompt and no round trips."""
    seen = []

    def call(prompt):
        seen.append(prompt)
        return json.dumps(_judgment())

    agent.judge(_dossier(), call=call)
    assert "You may ask for MORE evidence" not in seen[0]


def test_the_operator_can_switch_them_off_without_touching_the_code(
        monkeypatch):
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "0")
    assert tools.max_calls() == 0
    seen = []

    def call(prompt):
        seen.append(prompt)
        return json.dumps(_judgment())

    agent.judge(_dossier(), call=call, tool_calls=[])
    assert "You may ask for MORE evidence" not in seen[0]
    monkeypatch.setenv("GTRADE_ANALYST_TOOL_CALLS", "not a number")
    assert tools.max_calls() == tools.MAX_CALLS


def test_a_moscow_name_is_told_sec_does_not_cover_it_rather_than_getting_nothing():
    """An empty list reads as "no insider bought anything", which is a claim.
    "This source does not cover this market" is the truth."""
    out = tools._insider_filings("SBER")
    assert out["filings"] == []
    assert "do not cover Moscow-listed names" in out["note"]


def test_a_source_of_other_peoples_conclusions_is_refused_at_registration():
    """The owner's rule, 2026-09-03: the analyst is for its own reading, and
    consensus is something he can look up himself. Enforced at registration
    rather than trusted to a reader, because the tempting sources are the easy
    ones - Yahoo hands out recommendationMean in the very payload this project
    already fetches for P/E."""
    for name, describe in (("analyst_consensus", "what the street thinks"),
                           ("targets", "sell-side price target for this name"),
                           ("ratings", "broker upgrade and downgrade history")):
        with pytest.raises(tools.OpinionSource):
            tools.register(tools.Tool(name, {}, True, describe, lambda **k: {}))
    assert "analyst_consensus" not in tools._REGISTRY

    # positive control: a tool that returns material still registers, and the
    # check is on meaning rather than on the word "analyst" appearing anywhere
    probe = tools.Tool("dividend_history", {}, True,
                       "declared dividends and their record dates",
                       lambda **k: {})
    try:
        tools.register(probe)
        assert "dividend_history" in tools._REGISTRY
    finally:
        tools._REGISTRY.pop("dividend_history", None)


def test_the_shipped_tools_return_material_rather_than_a_verdict():
    """Both of them, restated as an assertion so a later edit to a description
    that turns one into an opinion source fails here."""
    for tool in tools._REGISTRY.values():
        text = ("%s %s" % (tool.name, tool.describe)).lower()
        assert not any(w in text for w in tools.OPINION_WORDS)
