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


def test_several_axes_may_share_one_cycle():
    settings, problems = ar_director.validate(
        _reply(axes="hyper,nets", budget=20), CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_AR_AXES"] == "hyper,nets"   # the runner's own format
    # a JSON list is the other shape a model reaches for
    settings, problems = ar_director.validate(
        _reply(axes=["hyper", "nets", "thresholds"], budget=20), CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_AR_AXES"] == "hyper,nets,thresholds"
    # a repeat is one axis, not a doubled bill
    settings, problems = ar_director.validate(
        _reply(axes="hyper, hyper", budget=60), CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_AR_AXES"] == "hyper"


def test_a_mixed_cycle_is_capped_in_width_and_in_total_cost():
    # width
    settings, problems = ar_director.validate(
        _reply(axes="hyper,nets,thresholds,regime", budget=10), CAMPAIGN)
    assert settings is None
    assert any("at most" in p for p in problems)
    # total cost: the budget is spent per axis, so 3 x 40 is not 40
    settings, problems = ar_director.validate(
        _reply(axes="hyper,nets,thresholds", budget=40), CAMPAIGN)
    assert settings is None
    assert any("per axis" in p for p in problems)


def test_qd_cannot_share_a_cycle_because_the_runner_would_ignore_the_rest():
    settings, problems = ar_director.validate(
        _reply(axes="qd,features", budget=20), CAMPAIGN)
    assert settings is None
    assert any("qd" in p for p in problems)


def test_one_bad_name_refuses_the_whole_mix():
    # Silently dropping it would run a cycle nobody chose, and the reason the
    # director gave would describe a search that never happened.
    settings, problems = ar_director.validate(
        _reply(axes="hyper,vibes", budget=20), CAMPAIGN)
    assert settings is None
    assert any("vibes" in p for p in problems)
    # and the no-op rule still bites inside a mix
    settings, problems = ar_director.validate(
        _reply(axes="hyper,weighting", label_mode="direction", budget=20), CAMPAIGN)
    assert settings is None
    assert any("no-op" in p for p in problems)


def test_the_cheap_gates_and_the_search_set_are_directable():
    settings, problems = ar_director.validate(
        _reply(screen=False, tier=True, search_assets="fast", hours=6), CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_AR_SCREEN"] == "0"     # the runner reads 1/0, not JSON true
    assert settings["GTRADE_AR_TIER"] == "1"
    assert settings["GTRADE_AR_SELECTION"] == "fast"
    assert settings["GTRADE_AR_TIME_BUDGET_H"] == "6.0"
    # and the enumerations are closed
    assert ar_director.validate(_reply(search_assets="just BTC"), CAMPAIGN)[0] is None
    assert ar_director.validate(_reply(screen="maybe"), CAMPAIGN)[0] is None
    assert ar_director.validate(_reply(hours=99), CAMPAIGN)[0] is None


def test_a_regate_cycle_re_tests_instead_of_searching():
    # Not _reply(): its axes/proposer defaults are exactly what a regate cycle
    # may not carry, so the fixture would test the refusal instead.
    settings, problems = ar_director.validate(
        {"mode": "regate", "regate_k": 12, "reason": "flagged runs ahead of replicated"},
        CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_AR_MODE"] == "regate"
    assert settings["GTRADE_AR_REGATE_K"] == "12"
    # k outside the range is refused, and the key is meaningless in a search
    assert ar_director.validate({"mode": "regate", "regate_k": 99}, CAMPAIGN)[0] is None
    settings, problems = ar_director.validate(_reply(regate_k=5), CAMPAIGN)
    assert settings is None
    assert any("mode=regate" in p for p in problems)


def test_a_lever_that_lands_on_nothing_is_refused_not_ignored():
    """A setting nobody reads still gets journalled as part of the run, so a later
    reader would credit the result to a knob that was never applied."""
    # qd levers on a non-qd cycle
    settings, problems = ar_director.validate(_reply(axes="hyper", qd_llm_p=0.9), CAMPAIGN)
    assert settings is None and any("qd search" in p for p in problems)
    # illum on a non-qd cycle
    settings, problems = ar_director.validate(_reply(axes="nets", illum="full"), CAMPAIGN)
    assert settings is None and any("qd archive" in p for p in problems)
    # a search proposer inside a regate cycle
    settings, problems = ar_director.validate(
        {"mode": "regate", "proposer": "llm"}, CAMPAIGN)
    assert settings is None and any("no search" in p for p in problems)
    # barrier settings under a next-bar label
    settings, problems = ar_director.validate(_reply(vol_window=40), CAMPAIGN)
    assert settings is None and any("triple_barrier" in p for p in problems)


def test_the_qd_and_barrier_levers_pass_when_they_apply():
    settings, problems = ar_director.validate(
        _reply(axes="qd", proposer="llm", qd_init=12, qd_final=5, max_misses=20,
               qd_llm_p=0.6, illum="full"), CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_AR_QD_INIT"] == "12"
    assert settings["GTRADE_AR_QD_FINAL"] == "5"
    assert settings["GTRADE_AR_QD_MAX_MISSES"] == "20"
    assert settings["GTRADE_AR_QD_LLM_P"] == "0.6"
    assert settings["GTRADE_AR_ILLUM"] == "full"
    settings, problems = ar_director.validate(
        _reply(axes="labeling", label_mode="triple_barrier", label_horizon=20,
               barrier_k=1.5, vol_window=40), CAMPAIGN)
    assert problems == []
    assert settings["GTRADE_LABEL_BARRIER_K"] == "1.5"
    assert settings["GTRADE_LABEL_VOL_WINDOW"] == "40"


def test_the_sticky_levers_are_written_on_every_reply():
    """They are env vars on a long-lived loop. Omitting one would carry the
    previous cycle's choice into a cycle whose reason never mentions it."""
    settings, _ = ar_director.validate(_reply(), CAMPAIGN)
    for key in ("GTRADE_AR_MODE", "GTRADE_AR_ILLUM", "GTRADE_AR_SELECTION"):
        assert key in settings, key
    assert settings["GTRADE_AR_MODE"] == "search"
    assert settings["GTRADE_AR_SELECTION"] == "full"


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


def test_axis_yield_totals_the_cycles_an_axis_produced_nothing_in():
    """The record-per-cycle view leaves the adding up to the model, and a local
    model does not do it: five consecutive empty `hyper` cycles read as five
    unrelated records. The totals must say it in one line."""
    findings = [{"axes": ["hyper"], "winners": []} for _ in range(5)]
    findings.append({"axes": ["qd"], "winners": [{"axis": "qd", "adoptable": True}]})
    out = {c["axis"]: c for c in ar_director.axis_yield(findings)}
    assert out["hyper"] == {"axis": "hyper", "cycles": 5, "winners": 0,
                            "gate_flagged": 0}
    assert out["qd"] == {"axis": "qd", "cycles": 1, "winners": 1,
                         "gate_flagged": 1}
    # the busiest axis first, so the empty one cannot hide at the bottom
    assert ar_director.axis_yield(findings)[0]["axis"] == "hyper"


def test_a_mixed_cycle_credits_the_axis_that_produced_the_winner():
    """Both axes spent a cycle, but only one of them found anything. Splitting
    the winner across the pair would exonerate the axis that found nothing."""
    findings = [{"axes": ["hyper", "features"],
                 "winners": [{"axis": "features", "adoptable": False}]}]
    out = {c["axis"]: c for c in ar_director.axis_yield(findings)}
    assert out["hyper"]["cycles"] == 1 and out["hyper"]["winners"] == 0
    assert out["features"]["cycles"] == 1 and out["features"]["winners"] == 1


def test_illum_full_is_refused_on_a_basis_that_cannot_read_the_nets():
    """It pays about 12x per genome to train real nets during illumination, and
    the raw Score cannot resolve them (retraining noise 0.45-1.52 Score), so the
    archive would rank the noise it just bought."""
    raw = dict(CAMPAIGN, GTRADE_AR_SCORE_BASIS="raw")
    settings, problems = ar_director.validate(_reply(axes="qd", illum="full"), raw)
    assert settings is None and any("raw Score basis" in p for p in problems)
    # Positive control: the same reply is legal wherever the nets are readable.
    for basis in ("net_gain", "net_auc", "ens_auc"):
        camp = dict(CAMPAIGN, GTRADE_AR_SCORE_BASIS=basis)
        settings, problems = ar_director.validate(
            _reply(axes="qd", illum="full"), camp)
        assert problems == [], basis
        assert settings["GTRADE_AR_ILLUM"] == "full"
    # And the cheap illumination is legal on raw, which is what it is for.
    _s, problems = ar_director.validate(_reply(axes="qd", illum="cb"), raw)
    assert problems == []


def test_a_campaign_that_names_no_basis_is_not_second_guessed():
    """The recipe shape-check validates against a fixture with no basis; making
    that fail would take the whole loop down over one arm."""
    bare = {k: v for k, v in CAMPAIGN.items() if k != "GTRADE_AR_SCORE_BASIS"}
    _s, problems = ar_director.validate(_reply(axes="qd", illum="full"), bare)
    assert problems == []
