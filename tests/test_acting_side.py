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
