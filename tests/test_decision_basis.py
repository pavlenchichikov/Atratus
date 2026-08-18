"""An adoption may be judged on a different basis than the search optimised.

The search basis is chosen for signal-to-noise, the decision basis for
relevance. Holding both in one constant made a search convenience into the
adoption criterion: on 2026-08-18, over the same 14 held-out assets and the same
two trainings, the mean Net_AUC delta was +0.036 while the mean Score delta was
-1.85, with rank correlation -0.24. The A/B passed on the first number and the
retrain it authorised then kept the champion on 23 of the first 29 assets.
"""

import auto_loop
import auto_research as ar


def test_the_decision_basis_defaults_to_the_search_basis(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "net_auc")
    monkeypatch.delenv("GTRADE_AR_DECISION_BASIS", raising=False)
    assert ar.decision_basis() == "net_auc"


def test_a_named_decision_basis_moves_the_gate_and_not_the_search(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "net_auc")
    monkeypatch.setenv("GTRADE_AR_DECISION_BASIS", "raw")
    assert ar._score_basis() == "net_auc"
    assert ar.decision_basis() == "raw"


def test_the_floor_follows_the_basis_it_is_given(monkeypatch):
    """A floor of 0.005 against a Score, or 0.5 against an AUC, passes or
    rejects everything regardless of the result."""
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "net_auc")
    assert ar._adopt_floor("mean") == 0.005
    assert ar._adopt_floor("mean", basis="raw") == ar.ADOPT_MEAN_SCORE_DELTA


def test_an_unknown_decision_basis_falls_back_rather_than_inventing_one(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "net_auc")
    monkeypatch.setenv("GTRADE_AR_DECISION_BASIS", "profit")
    assert ar.decision_basis() == "net_auc"


def test_rows_rekey_onto_the_basis_they_are_handed(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "raw")
    rows = [{"Asset": "A", "Score": 3.0, "Net_AUC": 0.61}]
    assert ar.rekey_rows(rows)[0]["Score"] == 3.0
    assert ar.rekey_rows(rows, basis="net_auc")[0]["Score"] == 0.61


def test_the_decision_basis_is_frozen_with_the_rest_of_the_campaign():
    assert "GTRADE_AR_DECISION_BASIS" in auto_loop.FROZEN
    assert auto_loop.CAMPAIGN["GTRADE_AR_DECISION_BASIS"] == ""


def test_a_campaign_frozen_before_the_constant_existed_still_runs():
    """Absent and empty are the same thing, or every older campaign would read
    as having moved a constant nobody touched."""
    old = {"GTRADE_AR_SCORE_BASIS": "net_auc", "GTRADE_AR_OBJECTIVE": "mean"}
    env = dict(auto_loop.CAMPAIGN)
    assert auto_loop.freeze_problems(old, env) == []


def test_a_real_move_is_still_refused():
    old = {"GTRADE_AR_SCORE_BASIS": "net_auc", "GTRADE_AR_OBJECTIVE": "mean",
           "GTRADE_AR_DECISION_BASIS": "raw"}
    env = dict(auto_loop.CAMPAIGN, GTRADE_AR_DECISION_BASIS="ens_auc")
    problems = auto_loop.freeze_problems(old, env)
    assert problems and "GTRADE_AR_DECISION_BASIS" in problems[0]


def test_naming_a_decision_basis_does_not_throw_the_search_archive_away(tmp_path, monkeypatch):
    """The archive ranks elites by SEARCH fitness. Changing how a result is
    ADOPTED says nothing about that ranking, and discarding six good elites for
    it would repeat the 2026-08-17 wipe the guard exists to prevent."""
    import json
    archive = tmp_path / "_qd_archive.json"
    archive.write_text(json.dumps({"2_4_5": {"fitness": 0.065, "genome": {}}}),
                       encoding="utf-8")
    monkeypatch.setattr(auto_loop, "ARCHIVE_PATH", str(archive))
    state = {"campaign": {"GTRADE_AR_SCORE_BASIS": "net_auc",
                          "GTRADE_AR_OBJECTIVE": "mean",
                          "GTRADE_AR_DECISION_BASIS": ""}}
    env = dict(auto_loop.CAMPAIGN, GTRADE_AR_DECISION_BASIS="raw")
    auto_loop.start_campaign(state, env, "split the bases")
    assert archive.exists(), "the archive was set aside for a decision-basis change"


def test_moving_the_search_basis_still_sets_the_archive_aside(tmp_path, monkeypatch):
    """Score-scale fitness runs 1.5 to 8.9 and AUC-scale about 0.01, so one
    survivor of the other scale outranks every elite of this one."""
    import json
    archive = tmp_path / "_qd_archive.json"
    archive.write_text(json.dumps({"2_4_5": {"fitness": 0.065, "genome": {}}}),
                       encoding="utf-8")
    monkeypatch.setattr(auto_loop, "ARCHIVE_PATH", str(archive))
    state = {"campaign": {"GTRADE_AR_SCORE_BASIS": "net_auc",
                          "GTRADE_AR_OBJECTIVE": "mean"}}
    env = dict(auto_loop.CAMPAIGN, GTRADE_AR_SCORE_BASIS="raw")
    auto_loop.start_campaign(state, env, "different basis")
    assert not archive.exists()
    assert (tmp_path / "_qd_archive.json.bak").exists()
