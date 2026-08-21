"""Stage C: the trust region, the generation stack and the tick's decision."""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import timing_online as on


class _Sides:
    """A policy stand-in that holds a fixed pattern of positions."""

    def __init__(self, pattern):
        self.pattern = pattern

    def sides(self, n):
        reps = int(np.ceil(n / len(self.pattern)))
        return np.tile(np.asarray(self.pattern, dtype=int), reps)[:n]


def _sides_of(policy, series):
    return policy.sides(len(series["probs"]))


def _assets(n_assets=4, n_bars=100):
    return {"A%d" % k: {"probs": np.zeros(n_bars)} for k in range(n_assets)}


def test_a_policy_agrees_with_itself_completely():
    p = _Sides([1, 1, 0, -1])
    assert on.agreement(_assets(), p, p, _sides_of) == 1.0


def test_a_policy_that_never_agrees_scores_zero():
    """The positive control for the trust region: if a total disagreement does
    not read as zero, no threshold over this number means anything."""
    assert on.agreement(_assets(), _Sides([1]), _Sides([-1]), _sides_of) == 0.0


def test_partial_agreement_is_the_fraction_of_bars():
    got = on.agreement(_assets(n_bars=100), _Sides([1, 1, 1, 0]),
                       _Sides([1, 1, 1, 1]), _sides_of)
    assert abs(got - 0.75) < 1e-9


def test_a_generation_outside_the_trust_region_is_refused():
    st = on.fresh_state()
    verdict, reason = on.decide(st, agree=0.5, challenger_score=99.0)
    assert verdict == "REJECT" and "trust region" in reason


def test_the_first_generation_is_accepted_when_it_stays_inside():
    st = on.fresh_state()
    verdict, _r = on.decide(st, agree=0.95, challenger_score=10.0)
    assert verdict == "ACCEPT"


def test_a_generation_that_loses_to_the_live_one_is_rolled_back():
    """The positive control for the rollback: a challenger that must lose has
    to be refused, or nothing this tick does is a gate."""
    st = on.apply_decision(on.fresh_state(), "ACCEPT", "first", 0.95, 10.0,
                           "2026-08-21T00:00:00")
    verdict, reason = on.decide(st, agree=0.95, challenger_score=9.0)
    assert verdict == "ROLLBACK" and "lost" in reason


def test_a_draw_is_not_an_improvement():
    st = on.apply_decision(on.fresh_state(), "ACCEPT", "first", 0.95, 10.0,
                           "2026-08-21T00:00:00")
    verdict, _r = on.decide(st, agree=0.95, challenger_score=10.0)
    assert verdict == "ROLLBACK"


def test_two_rollbacks_in_a_row_halt_the_schedule():
    """A mechanism that keeps producing losers is broken, and a broken
    mechanism must stop rather than keep spending CatBoost fits."""
    st = on.apply_decision(on.fresh_state(), "ACCEPT", "first", 0.95, 10.0,
                           "2026-08-21T00:00:00")
    st = on.apply_decision(st, "ROLLBACK", "lost", 0.95, 9.0,
                           "2026-08-22T00:00:00")
    verdict, reason = on.decide(st, agree=0.95, challenger_score=8.0)
    assert verdict == "HALT" and "stage_a" in reason
    st = on.apply_decision(st, verdict, reason, 0.95, 8.0,
                           "2026-08-23T00:00:00")
    assert st["halted"] is True
    assert on.decide(st, agree=0.99, challenger_score=999.0)[0] == "HALT"


def test_an_accepted_generation_resets_the_rollback_counter():
    st = on.apply_decision(on.fresh_state(), "ACCEPT", "first", 0.95, 10.0,
                           "2026-08-21T00:00:00")
    st = on.apply_decision(st, "ROLLBACK", "lost", 0.95, 9.0,
                           "2026-08-22T00:00:00")
    st = on.apply_decision(st, "ACCEPT", "won", 0.95, 11.0,
                           "2026-08-23T00:00:00")
    assert st["consecutive_rollbacks"] == 0
    assert st["generation"] == 2 and st["score"] == 11.0


def test_every_tick_is_journalled_including_the_quiet_ones():
    st = on.fresh_state()
    for i, v in enumerate(("REJECT", "ACCEPT", "ROLLBACK")):
        st = on.apply_decision(st, v, "r", 0.9, float(i),
                               "2026-08-2%dT00:00:00" % (1 + i))
    assert [e["verdict"] for e in st["journal"]] == ["REJECT", "ACCEPT",
                                                     "ROLLBACK"]


def test_the_state_round_trips(tmp_path):
    p = str(tmp_path / "s.json")
    st = on.apply_decision(on.fresh_state(), "ACCEPT", "first", 0.9, 5.0,
                           "2026-08-21T00:00:00")
    on.save_state(st, p)
    back = on.load_state(p)
    assert back["generation"] == 1 and back["score"] == 5.0
    assert len(back["journal"]) == 1


def test_a_missing_or_corrupt_file_reads_as_a_fresh_stack(tmp_path):
    """Never an exception: the tick is scheduled, and a scheduled job that
    crashes on a bad file stops running and nobody notices for a week."""
    assert on.load_state(str(tmp_path / "absent.json")) == on.fresh_state()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert on.load_state(str(bad)) == on.fresh_state()


def test_the_stage_is_off_unless_it_is_switched_on(monkeypatch):
    monkeypatch.delenv("GTRADE_TIMING_ONLINE", raising=False)
    assert on.online_on() is False
    monkeypatch.setenv("GTRADE_TIMING_ONLINE", "1")
    assert on.online_on() is True


def _fake_series(n=260, seed=0):
    rng = np.random.default_rng(seed)
    probs = np.clip(0.5 + rng.normal(0, 0.06, n), 0.01, 0.99)
    next_ret = rng.normal(0, 0.01, n)
    return {"probs": probs, "next_ret": next_ret,
            "atr": np.abs(rng.normal(1.0, 0.2, n)),
            "taleb_hi": rng.random(n) > 0.8,
            "close": 100.0 * np.cumprod(1.0 + next_ret),
            "buy_thr": 0.55, "sell_thr": 0.45,
            "risky": False, "is_forex": False}


def test_a_tick_decides_journals_and_returns_a_report():
    import train_timing_online as tto
    from core import timing_policy as tp

    by_asset = {"A%d" % k: _fake_series(seed=k) for k in range(6)}
    anchor = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    state, report = tto.tick(by_asset, on.fresh_state(), iters=2, seed=0,
                             anchor=anchor)
    assert report["verdict"] in ("ACCEPT", "REJECT", "ROLLBACK", "HALT")
    assert 0.0 <= report["agreement"] <= 1.0
    assert report["assets"] == 6
    assert len(state["journal"]) == 1


def test_a_halted_stack_does_no_work_at_all():
    """The halt has to stop the fit, not just the decision: the expensive part
    of a tick is six CatBoost fits, and a halted schedule that still pays for
    them has not stopped anything."""
    import train_timing_online as tto
    st = dict(on.fresh_state(), halted=True)
    state, report = tto.tick({}, st, iters=6)
    assert report["verdict"] == "HALT"
    assert report["assets"] == 0
    assert len(state["journal"]) == 1
