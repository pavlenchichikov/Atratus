import json

from core.analyst import agent
from core.analyst import evidence as ev


def test_opinion_domain_and_subdomain_are_dropped():
    kept, dropped = ev.check([
        {"source": "https://www.tipranks.com/x", "kind": "data", "value": "1"},
        {"source": "https://uk.marketbeat.com/y", "kind": "data", "value": "1"},
        {"source": "https://fred.stlouisfed.org/z", "kind": "data", "value": "4.1"}])
    assert [k["source"] for k in kept] == ["https://fred.stlouisfed.org/z"]
    assert len(dropped) == 2 and all(d["why"] for d in dropped)


def test_unknown_kind_opinion_words_and_uncalled_tool_drop():
    kept, dropped = ev.check([
        {"source": "raw_source", "kind": "data", "value": "x"},
        {"source": "https://sec.gov/a", "kind": "rumour", "value": "x"},
        {"source": "https://sec.gov/b", "kind": "data", "value": "price target 200"}],
        called=("web_fetch",))
    assert kept == [] and len(dropped) == 3


def test_a_called_tool_is_a_source_and_operating_is_not_rating():
    kept, _ = ev.check([{"source": "raw_source", "kind": "data",
                         "value": "operating income 4.2bn"}], called=("raw_source",))
    assert len(kept) == 1


def test_news_capped_two_per_publisher_and_90_days():
    items = [{"source": "https://reuters.com/%d" % i, "kind": "news", "value": "v",
              "asof": "2026-09-2%d" % i} for i in range(3)]
    items.append({"source": "https://apnews.com/a", "kind": "news", "value": "v",
                  "asof": "2026-01-01"})
    kept, dropped = ev.check(items, today="2026-09-29")
    assert len(kept) == 2 and len(dropped) == 2


def test_judgment_keeps_fields_and_raw_items_and_reports_the_dropped():
    reply = json.dumps({"direction": "up", "conviction": 3, "vol_regime": "normal",
                        "thesis": "t", "key_risk": "k", "evidence": [
                            "close",
                            {"source": "https://fred.stlouisfed.org/a", "kind": "data",
                             "value": "4.1", "asof": "2026-09-28"},
                            {"source": "https://www.zacks.com/b", "kind": "data", "value": "1"}]})
    j = agent.parse_judgment(reply, allowed={"close"})
    assert j["evidence"][0] == "close" and len(j["evidence"]) == 2
    assert len(j["dropped"]) == 1


def test_judgment_with_only_dropped_items_is_rejected():
    reply = json.dumps({"direction": "up", "conviction": 3, "vol_regime": "normal",
                        "evidence": [{"source": "https://www.zacks.com/b", "kind": "data",
                                      "value": "1"}]})
    why = []
    assert agent.parse_judgment(reply, allowed={"close"}, why=why) is None
    assert why


def test_label_turns_raw_items_into_hashable_text():
    assert ev.label("close") == "close"
    assert ev.label({"source": "https://www.fred.stlouisfed.org/x", "kind": "data"}) \
        == "data:fred.stlouisfed.org"
    assert ev.label({"source": "raw_source", "kind": "statistic"}) == "statistic:raw_source"


def _reply(evidence):
    return json.dumps({"direction": "up", "conviction": 3, "vol_regime": "normal",
                       "thesis": "t", "key_risk": "k", "evidence": evidence})


def test_a_tool_that_was_called_is_valid_evidence_by_name():
    j = agent.parse_judgment(_reply(["close", "compare", "macro_series:brent",
                                     "news_search (Sberbank)"]),
                             allowed={"close"},
                             called={"compare", "macro_series", "news_search"})
    assert j is not None
    assert j["evidence"] == ["close", "compare", "macro_series:brent", "news_search (Sberbank)"]


def test_a_tool_that_was_not_called_is_still_an_invented_field():
    why = []
    assert agent.parse_judgment(_reply(["close", "price_history"]), allowed={"close"},
                                called={"compare"}, why=why) is None
    assert "price_history" in why[0]


def test_the_critic_accepts_the_tools_the_judgment_used():
    dossier = {"asset": "SBER", "close": 1.0}
    j = {"direction": "up", "conviction": 3, "vol_regime": "normal", "thesis": "t",
         "key_risk": "k", "evidence": ["close"]}
    out = agent.critique(dossier, j, lambda p: _reply(["compare", "close"]),
                         called={"compare"})
    assert out is not None and "compare" in out["evidence"]
