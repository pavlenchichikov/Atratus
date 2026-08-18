"""What the campaign director is allowed to call a result.

It reads the findings journal and picks the next search settings. Given only
that journal it chased the one axis that kept getting flagged: the labeling
winner was gate-flagged eight times between 2026-08-17 and 08-18, the director
wrote "the labeling axis is currently the only one yielding adoptable results",
and the A/B that finally tested it lost on 10 of 14 held-out assets.

A search-gate flag says "worth an A/B", measured on the search basis against a
bare base. An A/B outcome says "beat what is running". Only the second is a
result, and the director now sees both, told apart.
"""

import json

import adopt_genome
from core import ar_director as director


def _finding(axis="labeling", adoptable=True):
    return {"ts": "2026-08-18T04:14:26", "mode": "axes", "basis": "net_auc",
            "axes": [axis], "budget": 30,
            "winners": [{"axis": axis, "value": 0.0918, "adoptable": adoptable,
                         "tag": "mean dScore 0.09", "clears": 8}]}


def test_a_gate_flag_is_reported_as_a_gate_flag(monkeypatch):
    """Called "adoptable" it read as a verdict. It is a shortlist entry."""
    rows = director.compact_findings([_finding()])
    assert rows[0]["gate_flagged"] == 1
    assert "adoptable" not in rows[0]


def test_a_winner_the_gate_turned_down_is_not_counted():
    rows = director.compact_findings([_finding(adoptable=False)])
    assert rows[0]["gate_flagged"] == 0


def test_the_prompt_separates_worth_testing_from_beat_production():
    ctx = {"basis": "net_auc", "decision": "raw", "objective": "mean",
           "findings": director.compact_findings([_finding()]),
           "adoptions": [{"candidate": "axis:labeling", "verdict": "FAILED",
                          "would_promote": 3, "would_demote": 10}],
           "archive_n": 6, "cycles": 7}
    text = director._prompt(ctx)
    assert "search_basis" in text and "decision_basis" in text
    assert "WORTH TESTING" in text
    assert "ACTUALLY RUNNING" in text
    assert "would_demote" in text
    assert "axis is exhausted" in text          # the rule that ends the loop
    assert '"verdict": "FAILED"' in text        # the outcome itself is in there


def test_a_campaign_with_no_finished_ab_says_so_rather_than_showing_nothing():
    ctx = {"basis": "net_auc", "decision": "raw", "objective": "mean",
           "findings": [], "adoptions": [], "archive_n": 0, "cycles": 1}
    assert "no A/B has finished" in director._prompt(ctx)


def test_the_decision_basis_falls_back_to_the_search_basis(monkeypatch):
    """An older campaign froze only one basis; the director must still be told
    what an adoption is judged on."""
    seen = {}
    monkeypatch.setattr(director, "_prompt", lambda ctx: seen.update(ctx) or "{}")
    monkeypatch.setattr(director.llm_proposer, "_backend",
                        lambda _role: (lambda _p: '{"reason": "x"}'))
    director.propose([], {"GTRADE_AR_SCORE_BASIS": "net_auc",
                          "GTRADE_AR_OBJECTIVE": "mean"})
    assert seen["decision"] == "net_auc"


# --- the outcomes themselves ------------------------------------------------

def _ab_file(tmp_path, name, results, reference="adopted:A"):
    (tmp_path / name).write_text(json.dumps({
        "holdout": "A,B", "objective": "mean", "floor": 0.5, "alpha": 0.05,
        "reference": reference, "results": results}), encoding="utf-8")


def test_an_outcome_that_demotes_more_than_it_promotes_reads_as_failed(tmp_path):
    _ab_file(tmp_path, "_ab_genomes_20260818-1222.json",
             {"axis:labeling": {"value_raw": 0.036, "p_raw": 0.0067, "n_raw": 14,
                                "promoted": 3, "demoted": 10}})
    out = adopt_genome.ab_outcomes(str(tmp_path))
    assert out[0]["verdict"] == "FAILED"
    assert out[0]["measured_against"] == "adopted:A"
    assert (out[0]["would_promote"], out[0]["would_demote"]) == (3, 10)


def test_a_real_win_reads_as_passed(tmp_path):
    _ab_file(tmp_path, "_ab_genomes_20260818-1222.json",
             {"cand": {"value_raw": 1.63, "p_raw": 0.0067, "n_raw": 14,
                       "promoted": 9, "demoted": 4}})
    assert adopt_genome.ab_outcomes(str(tmp_path))[0]["verdict"] == "PASSED"


def test_outcomes_come_newest_first_and_are_capped(tmp_path):
    for stamp in ("20260810-0000", "20260818-1222", "20260812-0000"):
        _ab_file(tmp_path, "_ab_genomes_%s.json" % stamp,
                 {"c": {"value_raw": 1.0, "p_raw": 0.01, "n_raw": 14}})
    out = adopt_genome.ab_outcomes(str(tmp_path), limit=2)
    assert [o["ts"] for o in out] == ["20260818", "20260812"]


def test_no_ab_files_is_an_empty_list_not_an_error(tmp_path):
    assert adopt_genome.ab_outcomes(str(tmp_path)) == []
