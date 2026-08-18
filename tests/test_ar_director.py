"""What the campaign director is and is not allowed to return."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import ar_director

CAMPAIGN = {"GTRADE_AR_AXES": "qd", "GTRADE_AR_SCORE_BASIS": "net_auc",
            "GTRADE_AR_OBJECTIVE": "mean", "GTRADE_LABEL_MODE": "direction",
            "GTRADE_LABEL_HORIZON": "1", "GTRADE_AR_PROPOSER": "evolutionary",
            "AR_BUDGET": "15"}


def _reply(**kw):
    out = {"axes": "hyper", "label_mode": "direction", "label_horizon": 1,
           "budget": 20, "proposer": "llm", "reason": "hyper is unexplored"}
    out.update(kw)
    return out


def test_a_well_formed_reply_is_accepted():
    # Positive control: every rejection test below is only meaningful because
    # a correct reply really does come back clean.
    settings, problems = ar_director.validate(_reply(), CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_AR_AXES"] == "hyper"
    assert settings["AR_BUDGET"] == "20"


def test_the_director_cannot_set_the_objective_directly():
    settings, problems = ar_director.validate(
        _reply(objective="cvar"), CAMPAIGN)
    assert settings is None
    assert any("whitelist" in p for p in problems)


def test_the_director_cannot_set_the_score_basis_directly():
    settings, problems = ar_director.validate(
        _reply(score_basis="raw"), CAMPAIGN)
    assert settings is None
    assert any("whitelist" in p for p in problems)


def test_an_unknown_axis_is_refused_rather_than_guessed():
    settings, problems = ar_director.validate(_reply(axes="vibes"), CAMPAIGN)
    assert settings is None
    assert any("axes" in p for p in problems)


def test_a_budget_outside_the_range_is_refused_rather_than_clamped():
    for bad in (0, 5000, "lots"):
        settings, problems = ar_director.validate(_reply(budget=bad), CAMPAIGN)
        assert settings is None, bad
        assert any("budget" in p for p in problems)


def test_the_weighting_axis_is_refused_under_a_next_bar_label():
    settings, problems = ar_director.validate(
        _reply(axes="weighting", label_mode="direction"), CAMPAIGN)
    assert settings is None
    assert any("no-op" in p for p in problems)
    # and it is allowed once the label spans more than one bar
    settings, problems = ar_director.validate(
        _reply(axes="weighting", label_mode="triple_barrier", label_horizon=20),
        CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_LABEL_MODE"] == "triple_barrier"


def test_a_new_campaign_needs_a_written_reason():
    settings, problems = ar_director.validate(
        _reply(new_campaign={"basis": "ens_auc", "objective": "mean",
                             "reason": "no"}), CAMPAIGN)
    assert settings is None
    assert any("reason" in p for p in problems)


def test_a_new_campaign_with_a_reason_carries_the_gate_constants():
    settings, problems = ar_director.validate(
        _reply(new_campaign={"basis": "ens_auc", "objective": "median",
                             "reason": "four runs on net_auc, nothing adoptable"}),
        CAMPAIGN)
    assert problems == []
    assert settings["new_campaign"]["GTRADE_AR_SCORE_BASIS"] == "ens_auc"
    assert settings["new_campaign"]["GTRADE_AR_OBJECTIVE"] == "median"


def test_a_new_campaign_cannot_invent_a_basis():
    settings, problems = ar_director.validate(
        _reply(new_campaign={"basis": "money", "objective": "mean",
                             "reason": "it would be better this way"}), CAMPAIGN)
    assert settings is None
    assert any("basis" in p for p in problems)


def test_garbage_is_refused():
    for junk in (None, [], "just some prose", 7):
        settings, problems = ar_director.validate(junk, CAMPAIGN)
        assert settings is None and problems


def test_a_partly_understood_reply_is_never_half_applied():
    # Valid axis, impossible budget: the good half must not survive.
    settings, _ = ar_director.validate(_reply(axes="nets", budget=999), CAMPAIGN)
    assert settings is None


def test_compact_findings_drops_the_genome_bodies():
    findings = [{"ts": "2026-08-16T10:00:00", "mode": "qd", "basis": "net_auc",
                 "axes": ["qd"], "winners": [
                     {"value": 0.02, "adoptable": True, "genome": {"drops": ["x"] * 40}},
                     {"value": 0.01, "adoptable": False, "genome": {}}]}]
    out = ar_director.compact_findings(findings)
    assert out == [{"ts": "2026-08-16", "mode": "qd", "basis": "net_auc",
                    "axes": ["qd"], "tried": 2, "gate_flagged": 1, "best": 0.02}]
    assert "genome" not in repr(out)
