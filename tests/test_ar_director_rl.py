"""The campaign director's arms and what they are allowed to be."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import ar_director
from core import ar_director_rl as rl

CAMPAIGN = {"GTRADE_AR_AXES": "qd", "GTRADE_AR_SCORE_BASIS": "net_auc",
            "GTRADE_AR_OBJECTIVE": "mean", "GTRADE_LABEL_MODE": "direction",
            "GTRADE_LABEL_HORIZON": "1", "GTRADE_AR_PROPOSER": "evolutionary",
            "AR_BUDGET": "15"}


def test_every_recipe_is_a_legal_cycle():
    """The free positive control over the whole set: the validator already
    refuses levers that land on nothing, so a recipe that passes is one the
    runner can actually execute. A failure names the recipe."""
    for name in rl.ARMS:
        settings, problems = ar_director.validate(rl.recipe_reply(name), CAMPAIGN)
        assert problems == [], "%s: %s" % (name, problems)
        assert settings["GTRADE_AR_AXES"]


def test_the_arm_set_covers_the_moves_the_campaign_needs():
    # the replication arm, the neural arm and the cheap-information arm are the
    # three the design argues cannot be dropped
    assert rl.recipe_reply("regate")["mode"] == "regate"
    assert rl.recipe_reply("qd_neural")["illum"] == "full"
    assert rl.recipe_reply("fast_scout")["search_assets"] == "fast"
    assert len(rl.ARMS) == len(set(rl.ARMS)) >= 10


def test_a_recipe_carries_a_cost_prior():
    for name in rl.ARMS:
        assert rl.RECIPES[name]["hours"] > 0
    # the neural arm is the expensive one and the design leans on that
    assert rl.RECIPES["qd_neural"]["hours"] > rl.RECIPES["qd_cheap"]["hours"]
    assert rl.RECIPES["regate"]["hours"] < rl.RECIPES["qd_cheap"]["hours"]


# --- the reward -------------------------------------------------------------

def test_a_flag_alone_can_never_outscore_a_pass():
    """gate_flagged means "worth testing", measured on the search basis against
    a bare base. The LLM director read it as a result and picked one axis eight
    times for a winner whose A/B then lost on 10 of 14 held-out assets."""
    flag = rl.cycle_reward(found=True, flagged=True, hours=1.0)
    passed = rl.cycle_reward(found=True, flagged=True, passed=True, hours=1.0)
    assert 0 < flag < passed
    # and it is capped once per cycle: two flagged winners are not two flags
    assert rl.cycle_reward(found=True, flagged=True, hours=1.0) == flag


def test_a_cycle_that_found_nothing_is_zero_and_that_is_available_at_once():
    assert rl.cycle_reward(found=False, hours=4.0) == 0.0


def test_replication_outscores_a_bare_pass():
    p = rl.cycle_reward(found=True, flagged=True, passed=True, hours=1.0)
    r = rl.cycle_reward(found=True, flagged=True, passed=True, replicated=True,
                        hours=1.0)
    assert r > p


def test_the_reward_is_per_hour_so_a_cheap_arm_can_win():
    """Without the normalisation the bandit systematically buys the most
    expensive arm. regate at 2h beating qd_neural at 12h is the whole point."""
    cheap = rl.cycle_reward(found=True, flagged=True, passed=True, hours=2.0)
    dear = rl.cycle_reward(found=True, flagged=True, passed=True, hours=12.0)
    assert cheap > dear
    # a zero stays zero however cheap the cycle was
    assert rl.cycle_reward(found=False, hours=0.1) == 0.0


def test_a_nonsense_duration_cannot_divide_by_zero():
    assert rl.cycle_reward(found=True, passed=True, hours=0.0) > 0
    assert rl.cycle_reward(found=True, passed=True, hours=-5.0) > 0


# --- credit assignment ------------------------------------------------------

def _hist(ts, arm, seconds=3600.0, action="search", rc=0):
    return {"ts": ts, "action": action, "rc": rc, "cycle": 1,
            "seconds": seconds, "chosen_by": "rl",
            "settings": rl.settings_of(arm)}


def _finding(ts, sig, adoptable=True):
    return {"ts": ts, "mode": "axes", "winners": [
        {"axis": "hyper", "adoptable": adoptable, "genome": {"sig": sig}}]}


def test_an_outcome_is_credited_to_the_cycle_that_produced_the_genome():
    """Not to the cycle that ran the A/B. That cycle chose an ab_run, which is
    not a research move and has no arm."""
    hist = [_hist("2026-08-19T02:00:00", "qd_cheap"),
            {"ts": "2026-08-19T09:00:00", "action": "ab_run", "rc": 0,
             "cycle": 2, "seconds": 900.0, "settings": None, "chosen_by": None}]
    findings = [_finding("2026-08-19T05:00:00", "SIG1")]
    outcomes = [{"sig": "SIG1", "verdict": "PASSED"}]
    got = rl.settle(hist, findings, outcomes, set(), set(),
                    sig_of=lambda g: g.get("sig"))
    assert len(got) == 1
    assert got[0]["arm"] == "qd_cheap"
    assert got[0]["reward"] > 0


def test_crediting_the_same_outcome_twice_is_a_no_op():
    hist = [_hist("2026-08-19T02:00:00", "qd_cheap")]
    findings = [_finding("2026-08-19T05:00:00", "SIG1")]
    outcomes = [{"sig": "SIG1", "verdict": "PASSED"}]
    first = rl.settle(hist, findings, outcomes, set(), set(),
                      sig_of=lambda g: g.get("sig"))
    credited = {r["key"] for r in first}
    again = rl.settle(hist, findings, outcomes, set(), credited,
                      sig_of=lambda g: g.get("sig"))
    assert again == []


def test_an_untraceable_outcome_is_dropped_rather_than_guessed_at():
    """A miscredited reward teaches the wrong arm, which is worse than a
    missing one."""
    hist = [_hist("2026-08-19T02:00:00", "qd_cheap")]
    findings = [_finding("2026-08-19T05:00:00", "SIG1")]
    outcomes = [{"sig": "NOTHING_KNOWS_THIS", "verdict": "PASSED"}]
    assert rl.settle(hist, findings, outcomes, set(), set(),
                     sig_of=lambda g: g.get("sig")) == []


def test_a_cycle_that_found_nothing_is_credited_zero_immediately():
    hist = [_hist("2026-08-19T02:00:00", "hyper_nets")]
    findings = [{"ts": "2026-08-19T05:00:00", "mode": "axes", "winners": []}]
    got = rl.settle(hist, findings, [], set(), set(),
                    sig_of=lambda g: g.get("sig"))
    assert len(got) == 1 and got[0]["arm"] == "hyper_nets"
    assert got[0]["reward"] == 0.0


def test_a_crashed_cycle_earns_neither_a_reward_nor_a_zero():
    """Its arm was not given a fair trial."""
    hist = [_hist("2026-08-19T02:00:00", "qd_neural", rc=1)]
    findings = [{"ts": "2026-08-19T05:00:00", "mode": "axes", "winners": []}]
    assert rl.settle(hist, findings, [], set(), set(),
                     sig_of=lambda g: g.get("sig")) == []


def test_a_history_entry_from_before_the_settings_existed_is_skipped():
    hist = [{"ts": "2026-08-10T02:00:00", "action": "search", "rc": 0,
             "cycle": 1}]
    findings = [{"ts": "2026-08-10T05:00:00", "mode": "axes", "winners": []}]
    assert rl.settle(hist, findings, [], set(), set(),
                     sig_of=lambda g: g.get("sig")) == []


# --- choosing, routing, state -----------------------------------------------

def test_the_replication_debt_forces_the_regate_arm():
    """Arithmetic, not something worth spending weeks of GPU time learning."""
    assert rl.forced_arm(13) == "regate"
    assert rl.forced_arm(1) is None


def test_the_debt_counts_genomes_not_journal_events():
    """A regate cycle re-flags a genome it already replicated. Counted as
    events that reads as fresh debt; counted as genomes it reads as nothing."""
    rec = {"winners": [{"adoptable": True, "genome": {"sig": "a"},
                        "replicated": True}]}
    sig_of = lambda g: g.get("sig")
    assert rl.replication_debt([rec] * 50, {"a"}, sig_of) == 0
    assert rl.replication_debt([rec] * 50, set(), sig_of) == 1


def test_the_forced_regate_never_runs_twice_in_a_row():
    """The 2026-08-24 latch: regate cannot reach a candidate outside its top-k,
    so forcing on an unmoved debt ran 1400 consecutive no-op cycles."""
    hist = [{"action": "search", "rc": 0,
             "settings": rl.settings_of("regate")}]
    assert rl.forced_arm(13, hist) is None
    assert rl.forced_arm(13, [{"action": "search", "rc": 0,
                               "settings": rl.settings_of("pruning")}]) == "regate"
    assert rl.forced_arm(13, []) == "regate"


def test_the_mode_defaults_to_the_llm_so_an_unset_variable_changes_nothing(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_DIRECTOR_MODE", raising=False)
    assert rl.mode() == "llm"
    monkeypatch.setenv("GTRADE_AR_DIRECTOR_MODE", "rl")
    assert rl.mode() == "rl"
    monkeypatch.setenv("GTRADE_AR_DIRECTOR_MODE", "nonsense")
    assert rl.mode() == "llm"          # unreadable degrades to today


def test_alternate_gives_odd_cycles_to_the_rl_director():
    assert rl.chooser_for(1) == "rl"
    assert rl.chooser_for(2) == "llm"


def test_choose_returns_a_legal_cycle_and_names_its_arm(monkeypatch):
    import random
    stored = {}
    monkeypatch.setattr(rl.ar_memory, "blob_put", lambda _k, obj: stored.update(obj))
    monkeypatch.setattr(rl.ar_memory, "blob_get", lambda _k, _d=None: stored)
    arm, settings = rl.choose([], [], [], set(), sig_of=lambda g: None,
                              rng=random.Random(0))
    assert arm in rl.ARMS
    assert settings.get("GTRADE_AR_AXES") or settings["GTRADE_AR_MODE"] == "regate"


def test_a_corrupt_blob_reads_as_a_fresh_bandit_rather_than_crashing(monkeypatch):
    for junk in (None, [], "not a dict", 7, {"scheduler": "broken",
                                             "credited": "nope",
                                             "hours": ["wrong"]}):
        monkeypatch.setattr(rl.ar_memory, "blob_get", lambda _k, _d=None, j=junk: j)
        sched, credited, hours = rl._load()
        assert credited == set() and hours == {}
        assert sched.posterior_mean(rl.ARMS[0], rl.PHASE) == 0.5   # the prior


def test_a_moved_campaign_base_halves_the_evidence_instead_of_trusting_it(monkeypatch):
    """The counts were earned on a different scale. Halving is what ar_rl
    already does to its own scheduler for the same reason."""
    stored = {}
    monkeypatch.setattr(rl.ar_memory, "blob_put",
                        lambda _k, obj: stored.update(obj))
    monkeypatch.setattr(rl.ar_memory, "blob_get", lambda _k, _d=None: stored)
    sched, credited, hours = rl._load("BASE_A")
    for _ in range(10):
        sched.update("qd_cheap", rl.PHASE, True)
    hot = sched.posterior_mean("qd_cheap", rl.PHASE)
    rl._save(sched, credited, hours, "BASE_A")
    same, _c, _h = rl._load("BASE_A")
    assert same.posterior_mean("qd_cheap", rl.PHASE) == hot     # unchanged
    moved, _c, _h = rl._load("BASE_B")
    assert moved.posterior_mean("qd_cheap", rl.PHASE) < hot     # halved
    assert moved.posterior_mean("qd_cheap", rl.PHASE) > 0.5     # not discarded


# --- the offline replay -----------------------------------------------------

def test_the_replay_refuses_to_rank_when_it_cannot_separate_a_dead_arm():
    """The generalisation the levels program paid for: an environment is only
    trusted once a parameter that MUST change the answer is shown to change
    it. One cycle per arm cannot separate anything."""
    hist = [_hist("2026-08-19T0%d:00:00" % i, arm)
            for i, arm in enumerate(("qd_cheap", "regate"), start=1)]
    findings = [{"ts": "2026-08-19T0%d:30:00" % i, "winners": []}
                for i in (1, 2)]
    rep = rl.replay(hist, findings, [], set(), sig_of=lambda g: None)
    assert rep["control_separated"] is False
    text = " ".join(rl.replay_lines(rep))
    assert "not measuring" in text
    assert "best arm" not in text          # no ranking is printed


def test_the_replay_separates_a_dead_arm_when_the_evidence_allows_it():
    """Positive control for the control: on a history where one arm always
    pays and the dead one never does, the replay must say so."""
    hist, findings, outcomes = [], [], []
    for i in range(1, 13):
        ts = "2026-08-%02dT01:00:00" % i
        hist.append(_hist(ts, "qd_cheap"))
        findings.append({"ts": "2026-08-%02dT02:00:00" % i, "winners": [
            {"adoptable": True, "genome": {"sig": "S%d" % i}}]})
        outcomes.append({"sig": "S%d" % i, "verdict": "PASSED"})
    rep = rl.replay(hist, findings, outcomes, set(), sig_of=lambda g: g.get("sig"))
    assert rep["control_separated"] is True
    assert rep["rows"]["qd_cheap"]["mean"] > 0


def test_the_replay_says_how_much_history_it_could_not_read():
    hist = [{"ts": "2026-08-10T02:00:00", "action": "search", "rc": 0,
             "cycle": 1},
            _hist("2026-08-19T02:00:00", "qd_cheap")]
    rep = rl.replay(hist, [{"ts": "2026-08-19T03:00:00", "winners": []}], [],
                    set(), sig_of=lambda g: None)
    assert rep["skipped"] == 1 and rep["read"] == 1
    assert "1 of 2" in " ".join(rl.replay_lines(rep))


# --- which arms a campaign may run -------------------------------------------

def test_the_neural_arm_drops_out_on_a_basis_that_cannot_read_the_nets(monkeypatch):
    """qd_neural is the only arm that trains real nets during illumination. On
    raw it would spend twelve hours ranking retraining noise."""
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "raw")
    assert rl.settings_of("qd_neural") is None
    assert "qd_neural" not in rl.legal_arms()
    assert len(rl.legal_arms()) == len(rl.ARMS) - 1
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "net_gain")
    assert rl.settings_of("qd_neural")["GTRADE_AR_ILLUM"] == "full"
    assert sorted(rl.legal_arms()) == sorted(rl.ARMS)


def test_an_unset_basis_leaves_every_arm_available(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_SCORE_BASIS", raising=False)
    assert sorted(rl.legal_arms()) == sorted(rl.ARMS)


def test_the_bandit_never_draws_an_arm_the_campaign_refuses(monkeypatch):
    """Drawing one would run the cycle with no settings applied and then credit
    the outcome to a recipe that was never used."""
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "raw")
    # choose() persists the scheduler. Unpatched, this test writes the LIVE
    # campaign's rl_director_v1.json in the checkout it runs from.
    from core import ar_memory
    monkeypatch.setattr(ar_memory, "blob_get", lambda *a, **k: {})
    monkeypatch.setattr(ar_memory, "blob_put", lambda *a, **k: None)
    drawn = []
    for _ in range(60):
        arm, settings = rl.choose([], [], [], set(), sig_of=lambda g: None)
        drawn.append(arm)
        assert settings is not None
    assert "qd_neural" not in drawn


def test_the_import_time_recipe_check_ignores_the_live_basis(monkeypatch):
    """It runs at import, so a raw campaign must not make the module unloadable."""
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "raw")
    rl._check_recipes()                       # raises if it read the live basis


def test_every_arm_keeps_the_campaigns_illumination(monkeypatch):
    """The 2026-08-22 bug: the campaign started on full illumination and every
    arm but one wrote cb back over it, so the search only ever illuminated on
    the CatBoost-only screen."""
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "net_gain")
    monkeypatch.setenv("GTRADE_AR_ILLUM", "full")
    assert sorted(rl.legal_arms()) == sorted(rl.ARMS)
    for arm in rl.ARMS:
        assert rl.settings_of(arm)["GTRADE_AR_ILLUM"] == "full", arm


def test_a_cheap_campaign_stays_cheap(monkeypatch):
    """Positive control: the inheritance carries whatever the campaign chose,
    it does not force full on everything."""
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "raw")
    monkeypatch.setenv("GTRADE_AR_ILLUM", "cb")
    for arm in rl.legal_arms():
        assert rl.settings_of(arm)["GTRADE_AR_ILLUM"] == "cb", arm
    assert "qd_neural" not in rl.legal_arms()
