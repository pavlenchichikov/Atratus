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


class _CountingPolicy:
    """An FqiPolicy-shaped stand-in that records how often it was rolled out.

    It only has to answer act(); rollout dispatches on that, which is the whole
    point of the change under test.
    """

    def __init__(self, action=1):
        self.action = action
        self.calls = 0
        self.model = object()

    def act(self, feat, i, st):
        self.calls += 1
        return self.action


def test_share_zero_is_byte_for_byte_the_old_tick():
    """The positive control for every assertion below. If the default moved,
    a tick that never asked for self-collection would silently change what it
    measures, and every earlier generation would stop being comparable."""
    import train_timing_online as tto
    from core import timing_policy as tp

    by_asset = {"A%d" % k: _fake_series(seed=k) for k in range(6)}
    anchor = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    champ = _CountingPolicy()

    _s1, r1 = tto.tick(by_asset, on.fresh_state(), iters=2, seed=0,
                       anchor=anchor)
    _s2, r2 = tto.tick(by_asset, on.fresh_state(), iters=2, seed=0,
                       anchor=anchor, champion=champ, self_share=0.0)
    assert champ.calls == 0, "the champion collected data at share 0"
    assert r1["self_rollouts"] == 0 and r2["self_rollouts"] == 0
    assert r1["score"] == r2["score"] and r1["agreement"] == r2["agreement"]
    assert r1["selected_on_val"] == r2["selected_on_val"]


def test_a_share_moves_only_the_data_never_the_trust_region():
    """The invariant the design turns on: the anchor keeps measuring, whatever
    collects. Half the assets roll out under the champion, and agreement is
    still computed against the rules."""
    import train_timing_online as tto
    from core import timing_policy as tp

    by_asset = {"A%d" % k: _fake_series(seed=k) for k in range(6)}
    anchor = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    champ = _CountingPolicy()
    _state, report = tto.tick(by_asset, on.fresh_state(), iters=2, seed=0,
                              anchor=anchor, champion=champ, self_share=0.5)
    assert report["self_rollouts"] == 3, report
    assert champ.calls > 0, "the champion was chosen but never rolled out"
    assert 0.0 <= report["agreement"] <= 1.0


def test_no_stored_champion_falls_back_to_the_anchor():
    """First ever tick, or a killed run that left no model. Asking for
    self-collection must not stop the schedule."""
    import train_timing_online as tto
    from core import timing_policy as tp

    by_asset = {"A%d" % k: _fake_series(seed=k) for k in range(4)}
    anchor = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    _state, report = tto.tick(by_asset, on.fresh_state(), iters=2, seed=0,
                              anchor=anchor, champion=None, self_share=0.5)
    assert report["self_rollouts"] == 0
    assert report["verdict"] in ("ACCEPT", "REJECT", "ROLLBACK", "HALT")


def test_the_split_is_deterministic_and_sized(monkeypatch):
    """By asset, not by bar: a Q that is wrong must not be able to poison every
    series a little. The same seed has to pick the same assets, or two ticks
    would not be comparable."""
    import random

    import train_timing_online as tto

    names = {"A%d" % k: None for k in range(10)}
    a, c = object(), object()
    first = tto._behaviour_for(names, a, c, 0.3, random.Random(7))
    again = tto._behaviour_for(names, a, c, 0.3, random.Random(7))
    assert first == again
    assert sum(1 for v in first.values() if v is c) == 3
    assert tto._behaviour_for(names, a, c, 1.0, random.Random(7)) == {
        n: c for n in names}
    assert tto._behaviour_for(names, a, None, 0.9, random.Random(7)) == {
        n: a for n in names}


def test_only_an_accepted_generation_is_kept(monkeypatch, tmp_path):
    """A rejected Q collecting the next tick's data is the lock-in this whole
    design guards against, handed the keys."""
    import train_timing_online as tto
    from core import timing_policy as tp

    saved = []
    monkeypatch.setattr(on, "save_champion", lambda m: saved.append(m) or True)
    by_asset = {"A%d" % k: _fake_series(seed=k) for k in range(6)}
    anchor = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    _state, report = tto.tick(by_asset, on.fresh_state(), iters=2, seed=0,
                              anchor=anchor)
    assert len(saved) == (1 if report["verdict"] == "ACCEPT" else 0), report


def test_a_missing_or_broken_champion_file_reads_as_none(tmp_path):
    """Never raises: this runs on a schedule, and a half-written model from a
    killed run must fall back to the anchor rather than stop the tick."""
    assert on.load_champion(str(tmp_path / "nothing.cbm")) is None
    junk = tmp_path / "junk.cbm"
    junk.write_bytes(b"not a catboost model")
    assert on.load_champion(str(junk)) is None
