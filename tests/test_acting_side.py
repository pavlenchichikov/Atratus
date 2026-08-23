"""One side for the levels: the one the timing layer is actually on.

Before this, the fitter walked Stage A, the journal recorded the raw gated
signal and the card annotated Stage B - three answers to one question on one
screen. These pin the single answer down, including the two disagreements that
motivated it: a signal the policy sits out, and a position it holds through a
bar whose signal went WAIT.
"""
import pytest

from core import levels as levels_mod


@pytest.mark.parametrize("signal", ["BUY", "SELL", "WAIT"])
def test_no_decision_falls_back_to_the_signal(signal):
    # Timing off, or its shadow skipped this asset: nothing decided, so the
    # card shows the signal and the levels belong to it.
    assert levels_mod.acting_side(signal, "SP500", None) == signal
    assert levels_mod.acting_side(signal, "SP500", "") == signal


@pytest.mark.parametrize("action,expected", [
    ("ENTER:+1", "BUY"),
    ("ENTER:-1", "SELL"),
    ("EXIT", "WAIT"),
    ("STAY_OUT", "WAIT"),
])
def test_actions_that_name_their_own_side(action, expected):
    # The raw signal is deliberately the opposite of the answer, so a pass
    # cannot come from quietly returning it.
    assert levels_mod.acting_side("SELL", "SP500", action) == expected
    assert levels_mod.acting_side("BUY", "SP500", action) == expected


@pytest.mark.parametrize("pos,expected", [(1, "BUY"), (-1, "SELL"), (0, "WAIT")])
def test_hold_reads_the_position_it_is_holding(monkeypatch, pos, expected):
    import performance_tracker as pt
    monkeypatch.setattr(pt, "timing_state", lambda a, **k: {"pos": pos})
    assert levels_mod.acting_side("WAIT", "SP500", "HOLD") == expected


def test_signal_the_policy_sits_out_gets_no_levels():
    # The case that used to journal a BUY entry and stop for a trade the
    # adopted policy declined to take.
    assert levels_mod.acting_side("BUY", "SP500", "STAY_OUT") == "WAIT"


def test_position_held_through_a_wait_bar_keeps_its_side(monkeypatch):
    # The mirror case: the signal went quiet but the stop is still live.
    import performance_tracker as pt
    monkeypatch.setattr(pt, "timing_state", lambda a, **k: {"pos": -1})
    assert levels_mod.acting_side("WAIT", "SP500", "HOLD") == "SELL"


def test_unreadable_tracker_falls_back_rather_than_raising(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no db")
    import performance_tracker as pt
    monkeypatch.setattr(pt, "timing_state", boom)
    assert levels_mod.acting_side("BUY", "SP500", "HOLD") == "BUY"


def test_the_fallback_is_reachable_only_on_failure(monkeypatch):
    # Positive control for the test above: with a WORKING tracker the same
    # call must NOT return the signal, or that test would pass for the wrong
    # reason on any implementation that ignored the position entirely.
    import performance_tracker as pt
    monkeypatch.setattr(pt, "timing_state", lambda a, **k: {"pos": -1})
    assert levels_mod.acting_side("BUY", "SP500", "HOLD") == "SELL"


def test_fitter_walks_whatever_is_served(monkeypatch):
    import train_levels as tl
    from core import timing_fqi as fq
    from core import timing_policy as tp

    stage_a = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    sentinel_b = object()

    monkeypatch.setattr(tp, "timing_on", lambda: False)
    off = tl.served_policy()
    assert isinstance(off, tp.RulesPolicy)

    monkeypatch.setattr(tp, "timing_on", lambda: True)
    monkeypatch.setattr(tp, "load_policy", lambda *a, **k: stage_a)
    monkeypatch.setattr(fq, "stage_b_on", lambda: False)
    assert tl.served_policy() is stage_a

    monkeypatch.setattr(fq, "stage_b_on", lambda: True)
    monkeypatch.setattr(fq, "load_served_policy", lambda *a, **k: sentinel_b)
    assert tl.served_policy() is sentinel_b

    # Stage B switched on but nothing adopted falls back to the rules, never
    # to nothing - the same rule serving follows in core.scoring.
    monkeypatch.setattr(fq, "load_served_policy", lambda *a, **k: None)
    assert tl.served_policy() is stage_a


# --- which bars issue at all -------------------------------------------------
#
# The side was unified first; this is the other half. The fit scores one trade
# per entered segment, so the journal has to record one row per entered segment
# too, or the two measure different things again.

@pytest.mark.parametrize("action", ["ENTER:+1", "ENTER:-1"])
def test_a_bar_that_opens_a_position_issues(action):
    assert levels_mod.issues_levels(action) is True


@pytest.mark.parametrize("action", ["HOLD", "EXIT", "STAY_OUT"])
def test_a_bar_inside_or_leaving_a_position_does_not(action):
    assert levels_mod.issues_levels(action) is False


@pytest.mark.parametrize("action", [None, ""])
def test_no_timing_decision_keeps_issuing_as_before(action):
    """Switching the timing layer off must not silently empty the journal."""
    assert levels_mod.issues_levels(action) is True


def test_the_journal_rule_matches_the_fitting_rule():
    """The pairing itself, so the two cannot drift apart unnoticed.

    train_levels._issue_bars picks the ENTER bars out of a walk; issues_levels
    answers the same question one bar at a time. Given one walk, they have to
    select the same bars.
    """
    import numpy as np

    import train_levels as tl
    actions = ["STAY_OUT", "ENTER:+1", "HOLD", "HOLD", "EXIT",
               "STAY_OUT", "ENTER:-1", "HOLD"]
    sides = np.array([0, 1, 1, 1, 0, 0, -1, -1])
    # _issue_bars keys on the bare verb, which is what walk_policy emits.
    fit = tl._issue_bars(sides, [a.split(":")[0] for a in actions], "equity")
    served = [i for i, a in enumerate(actions) if levels_mod.issues_levels(a)]
    assert fit == served == [1, 6]
