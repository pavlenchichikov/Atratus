"""Sizing authority: the environment, the rule, and the gate over it."""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import backtesting as bt


def _arrays(n=200, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.choice([-1, 0, 1], size=n)
    ret = rng.normal(0, 0.01, n)
    return sig, ret


def test_a_unit_size_changes_nothing_at_all():
    """The control the whole piece rests on. Every stored measurement in this
    repository was produced by the behaviour without sizes, so the extension
    has to be unable to move a number it should not."""
    sig, ret = _arrays()
    plain = bt.simulate_positions(sig, ret, 0.001, 0.0005)
    none_given = bt.simulate_positions(sig, ret, 0.001, 0.0005, sizes=None)
    all_ones = bt.simulate_positions(sig, ret, 0.001, 0.0005,
                                     sizes=np.ones(len(sig)))
    assert plain[:3] == none_given[:3] == all_ones[:3]
    assert np.array_equal(plain[3], all_ones[3])


def test_a_bigger_size_scales_the_bar_return():
    sig = np.ones(10, dtype=int)
    ret = np.full(10, 0.01)
    small = bt.simulate_positions(sig, ret, 0.0, 0.0, sizes=np.full(10, 0.5))
    big = bt.simulate_positions(sig, ret, 0.0, 0.0, sizes=np.full(10, 2.0))
    assert big[0] > small[0]
    assert abs(big[3][5] - 4.0 * small[3][5]) < 1e-12


def test_a_resize_pays_for_the_notional_it_moves():
    sig = np.ones(6, dtype=int)
    ret = np.zeros(6)
    sizes = np.array([1.0, 1.0, 1.5, 1.5, 1.5, 1.5])
    _p, _t, _w, daily = bt.simulate_positions(sig, ret, 0.01, 0.0, sizes=sizes)
    assert abs(daily[2] - (-0.5 * 0.01)) < 1e-12     # half a leg for half a unit
    assert daily[3] == 0.0                            # holding costs nothing


def test_a_resize_is_not_a_new_trade():
    """Otherwise a policy could inflate its trade count by resizing, and dodge
    the minimum-trades floor that every score in this system leans on."""
    sig = np.ones(6, dtype=int)
    ret = np.zeros(6)
    flat = bt.simulate_positions(sig, ret, 0.0, 0.0)
    resized = bt.simulate_positions(sig, ret, 0.0, 0.0,
                                    sizes=np.array([1.0, 2.0, 0.5, 2.0,
                                                    0.5, 1.0]))
    assert flat[1] == resized[1] == 1


def _series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    probs = np.clip(0.5 + rng.normal(0, 0.06, n), 0.01, 0.99)
    next_ret = rng.normal(0, 0.01, n)
    return {"probs": probs, "next_ret": next_ret,
            "atr": np.abs(rng.normal(1.0, 0.2, n)),
            "taleb_hi": rng.random(n) > 0.8,
            "close": 100.0 * np.cumprod(1.0 + next_ret),
            "buy_thr": 0.55, "sell_thr": 0.45,
            "risky": False, "is_forex": False}


def test_the_default_parameters_are_the_unit_size():
    """The identity the ES starts from, exactly as the timing fit does."""
    from core import sizing_policy as sp
    sizes = sp.SizingPolicy(dict(sp.DEFAULT_PARAMS)).sizes_for(_series())
    assert np.allclose(sizes, 1.0)


def test_a_stronger_signal_buys_a_bigger_position():
    from core import sizing_policy as sp
    s = _series()
    s["probs"] = np.array([0.56, 0.95] * 100)     # just past, and far past
    sizes = sp.SizingPolicy(dict(sp.DEFAULT_PARAMS, k_margin=1.0)).sizes_for(s)
    assert sizes[1] > sizes[0]


def test_tail_risk_and_volatility_shrink_it():
    from core import sizing_policy as sp
    s = _series()
    s["taleb_hi"] = np.array([False, True] * 100)
    sizes = sp.SizingPolicy(dict(sp.DEFAULT_PARAMS, k_taleb=0.5)).sizes_for(s)
    assert sizes[1] < sizes[0]


def test_the_size_is_clipped_at_both_ends():
    from core import sizing_policy as sp
    s = _series()
    huge = sp.SizingPolicy(dict(sp.DEFAULT_PARAMS, base=99.0)).sizes_for(s)
    tiny = sp.SizingPolicy(dict(sp.DEFAULT_PARAMS, base=-99.0)).sizes_for(s)
    assert np.allclose(huge, sp.SIZE_HI) and np.allclose(tiny, sp.SIZE_LO)


def test_the_identity_rule_scores_exactly_like_the_incumbent():
    """The second control: a rule that multiplies every position by one has to
    be indistinguishable from no rule at all, or the fit starts from a place
    the gate cannot compare against."""
    import train_sizing as ts
    from core import sizing_policy as sp
    s = _series(300, seed=2)
    same = ts.eval_sizing(s, sp.SizingPolicy(dict(sp.DEFAULT_PARAMS)))
    base = ts.eval_sizing(s, None)
    assert same["score"] == base["score"]
    assert same["n_trades"] == base["n_trades"]


def test_the_gate_holds_on_a_draw():
    import train_sizing as ts
    from core import sizing_policy as sp
    by_asset = {"A%d" % k: _series(300, seed=10 + k) for k in range(9)}
    out = ts.gate_sizing(by_asset, dict(sp.DEFAULT_PARAMS))
    assert out["verdict"] == "HOLD" and abs(out["mean_d"]) < 1e-9


def test_a_constant_size_carries_no_information_and_must_score_like_one():
    """The control that closes the leverage channel. Measured before it
    existed: a constant 1.5x scored +61.0 against the unit arm at p 0.0010,
    which is not a sizing result, it is a bigger bet. After exposure matching a
    constant of ANY value has to be the incumbent exactly."""
    import train_sizing as ts
    from core import sizing_policy as sp
    s = _series(300, seed=2)
    base = ts.eval_sizing(s, None)
    for value in (0.5, 1.0, 1.5, 2.0):
        flat = ts.eval_sizing(s, sp.SizingPolicy(dict(sp.DEFAULT_PARAMS,
                                                      base=value)))
        assert abs(flat["score"] - base["score"]) < 1e-9, value


def test_a_shape_can_still_move_the_score():
    """The other end of the control: matching exposure must not flatten
    everything, or the gate could never say anything at all."""
    import train_sizing as ts
    from core import sizing_policy as sp
    s = _series(300, seed=2)
    shaped = ts.eval_sizing(s, sp.SizingPolicy(dict(sp.DEFAULT_PARAMS,
                                                    k_margin=2.0)))
    base = ts.eval_sizing(s, None)
    assert shaped["score"] != base["score"]
