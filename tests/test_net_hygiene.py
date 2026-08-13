import numpy as np
import pytest

from core import net_hygiene as nh


def test_env_readers(monkeypatch):
    for k in ("GTRADE_NET_SEEDS", "GTRADE_NET_UNIQUENESS",
              "GTRADE_NET_CALIBRATE", "GTRADE_NET_ABSTAIN_EPS"):
        monkeypatch.delenv(k, raising=False)
    assert nh.net_seeds() == 1
    assert nh.uniqueness_on() is False
    assert nh.calibrate_nets_on() is False
    assert nh.abstain_eps() == 0.0
    monkeypatch.setenv("GTRADE_NET_SEEDS", "5")
    monkeypatch.setenv("GTRADE_NET_UNIQUENESS", "1")
    monkeypatch.setenv("GTRADE_NET_CALIBRATE", "true")
    monkeypatch.setenv("GTRADE_NET_ABSTAIN_EPS", "0.05")
    assert nh.net_seeds() == 5
    assert nh.uniqueness_on() is True
    assert nh.calibrate_nets_on() is True
    assert nh.abstain_eps() == 0.05
    monkeypatch.setenv("GTRADE_NET_SEEDS", "bad")
    assert nh.net_seeds() == 1                 # bad value - floor 1
    monkeypatch.setenv("GTRADE_NET_SEEDS", "0")
    assert nh.net_seeds() == 1                 # floored at 1


def test_seed_base_env(monkeypatch):
    monkeypatch.delenv("GTRADE_SEED", raising=False)
    assert nh.seed_base() == nh.DEFAULT_SEED
    monkeypatch.setenv("GTRADE_SEED", "7")
    assert nh.seed_base() == 7
    monkeypatch.setenv("GTRADE_SEED", "bad")
    assert nh.seed_base() == nh.DEFAULT_SEED


def test_asset_seed_is_stable_and_separating(monkeypatch):
    monkeypatch.delenv("GTRADE_SEED", raising=False)
    # Same inputs, same seed - this is what makes a research A/B paired. It must
    # not depend on call order, which is what a counter-based seed would.
    assert nh.asset_seed("SP500", 1, 0) == nh.asset_seed("SP500", 1, 0)
    for a, b in (("SP500", "NVDA"), ("BTC", "GOLD")):
        assert nh.asset_seed(a) != nh.asset_seed(b)
    # fold and seed-member each move the draw
    assert nh.asset_seed("SP500", 1, 0) != nh.asset_seed("SP500", 2, 0)
    assert nh.asset_seed("SP500", 1, 0) != nh.asset_seed("SP500", 1, 1)
    # a different GTRADE_SEED re-rolls the whole run
    before = nh.asset_seed("SP500", 1, 0)
    monkeypatch.setenv("GTRADE_SEED", "12345")
    assert nh.asset_seed("SP500", 1, 0) != before
    # stays inside the int32 range every RNG here accepts
    for asset in ("SP500", "NVDA", "^GSPC", "SBER.ME"):
        assert 0 <= nh.asset_seed(asset, 9, 3) < 2 ** 31 - 1


def test_average_probs():
    a = np.array([0.2, 0.8])
    assert np.allclose(nh.average_probs([a]), a)          # single passthrough
    b = np.array([0.4, 0.6])
    assert np.allclose(nh.average_probs([a, b]), [0.3, 0.7])
    with pytest.raises(ValueError):
        nh.average_probs([])


def test_uniqueness_weights_direction_is_uniform():
    w = nh.uniqueness_weights(10, 1)
    assert np.allclose(w, np.ones(10))
    assert np.allclose(nh.uniqueness_weights(10, 0), np.ones(10))


def test_uniqueness_weights_window_downweights_interior():
    w = nh.uniqueness_weights(9, 3)
    assert len(w) == 9
    assert abs(w.mean() - 1.0) < 1e-9                     # normalized to mean 1
    # interior samples overlap fully (concurrency ~horizon) - lower weight than edges
    assert w[0] > w[4] and w[-1] > w[4]


def test_calibrate_and_abstain_maps_and_abstains():
    # Balanced dataset: 0 to 0.5 all-zero labels, 0.5 to 1 all-one labels.
    # 0.5 appears at the boundary of both halves so the isotonic calibrator
    # must assign it exactly 0.5 - reliably abstains with eps=0.05.
    val_prob = np.r_[np.linspace(0, 0.5, 200), np.linspace(0.5, 1.0, 200)]
    val_target = np.r_[np.zeros(200, dtype=int), np.ones(200, dtype=int)]
    test_prob = np.array([0.5, 0.9, 0.1, 0.51])
    cv, ct = nh.calibrate_and_abstain(val_prob, val_target, test_prob, 0.05)
    assert len(ct) == 4 and len(cv) == 400
    # exactly 0.5 calibrates to 0.5 - |0.5-0.5|=0 < eps - abstains
    assert ct[0] == 0.5
    # confident 0.9 / 0.1 calibrate to ~1 / ~0 - not abstained
    assert ct[1] != 0.5 and ct[2] != 0.5


def test_calibrate_and_abstain_one_class_is_identity():
    # one-class targets - calibrator None - identity (plus abstention)
    val_prob = np.array([0.3, 0.4, 0.6, 0.7])
    val_target = np.array([1, 1, 1, 1])
    test_prob = np.array([0.2, 0.8])
    _cv, ct = nh.calibrate_and_abstain(val_prob, val_target, test_prob, 0.0)
    assert np.allclose(ct, [0.2, 0.8])                    # identity, no abstention (eps 0)


class TestUniquenessSpans:
    def test_empty(self):
        out = nh.uniqueness_weights_spans([])
        assert len(out) == 0

    def test_all_unit_spans_are_all_ones(self):
        out = nh.uniqueness_weights_spans([1, 1, 1, 1])
        assert np.allclose(out, 1.0)

    def test_hand_computed_three_samples(self):
        # spans [2, 2, 1]:
        #   sample 0 covers bars [0,2), sample 1 covers [1,3), sample 2 covers [2,3)
        #   concurrency: bar0=1, bar1=2, bar2=2
        #   u0 = mean(1/1, 1/2) = 0.75, u1 = mean(1/2, 1/2) = 0.5, u2 = 1/2 = 0.5
        #   mean(u) = 0.5833..., normalized -> [1.2857, 0.8571, 0.8571]
        out = nh.uniqueness_weights_spans([2, 2, 1])
        assert np.allclose(out, [1.28571429, 0.85714286, 0.85714286])
        assert abs(out.mean() - 1.0) < 1e-9

    def test_overlapping_samples_are_down_weighted(self):
        # A long-lived sample buried in overlap must weigh less than an isolated one.
        spans = [10] * 20 + [1] * 20
        out = nh.uniqueness_weights_spans(spans)
        assert out[:20].mean() < out[20:].mean()

    def test_variable_spans_produce_real_spread(self):
        # The whole point: unlike the fixed-horizon version, this must not collapse
        # to ~all ones after normalization.
        rng = np.random.default_rng(0)
        spans = np.minimum(rng.geometric(0.15, 4000), 20)
        out = nh.uniqueness_weights_spans(spans)
        outside = np.mean((out < 0.95) | (out > 1.05))
        assert outside > 0.5

    def test_fixed_spans_match_the_scalar_horizon_version(self):
        n, h = 500, 8
        a = nh.uniqueness_weights(n, h)
        b = nh.uniqueness_weights_spans([h] * n)
        # Same construction, so the interior must agree closely; only the tail
        # differs because the scalar version lets windows run past the last sample.
        assert np.allclose(a[: n - h], b[: n - h], atol=0.02)

    def test_zero_and_negative_spans_are_floored_to_one(self):
        out = nh.uniqueness_weights_spans([0, -3, 1])
        assert np.allclose(out, 1.0)

    def test_cb_env_reader(self, monkeypatch):
        monkeypatch.delenv("GTRADE_CB_UNIQUENESS", raising=False)
        assert nh.cb_uniqueness_on() is False
        monkeypatch.setenv("GTRADE_CB_UNIQUENESS", "1")
        assert nh.cb_uniqueness_on() is True
        monkeypatch.setenv("GTRADE_CB_UNIQUENESS", "true")
        assert nh.cb_uniqueness_on() is True
        monkeypatch.setenv("GTRADE_CB_UNIQUENESS", "0")
        assert nh.cb_uniqueness_on() is False
