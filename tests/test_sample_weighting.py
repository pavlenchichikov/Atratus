"""The weights themselves are unit-tested in tests/test_net_hygiene.py. This file
guards the WIRING inside _train_one_asset, which is too large and too data-hungry to
call directly, so it asserts on the source the way tests/test_timing_policy.py does."""

import inspect

import numpy as np

import train_hybrid
from core import net_hygiene


def _src():
    return inspect.getsource(train_hybrid._train_one_asset)


class TestWiring:
    def test_span_is_read_from_the_dataframe(self):
        src = _src()
        assert "label_span" in src

    def test_span_rides_the_fold_dict(self):
        src = _src()
        assert "'span_train'" in src
        assert "'span_seq_train'" in src

    def test_catboost_receives_sample_weight(self):
        src = _src()
        assert src.count("sample_weight=_cb_w") == 2   # CPU and GPU branches

    def test_catboost_weights_are_gated_by_the_flag(self):
        src = _src()
        assert "cb_uniqueness_on()" in src

    def test_nets_use_the_span_based_weights(self):
        src = _src()
        assert "uniqueness_weights_spans" in src

    def test_scalar_horizon_weights_are_gone_from_the_trainer(self):
        # The fixed-horizon call is a measured no-op; it must not linger next to the
        # real one where a future reader could reinstate it.
        src = _src()
        assert "uniqueness_weights(" not in src.replace("uniqueness_weights_spans(", "")


class TestWeightsAreUsable:
    def test_weights_are_finite_positive_and_mean_one(self):
        rng = np.random.default_rng(1)
        spans = np.minimum(rng.geometric(0.2, 500), 15).astype(float)
        w = net_hygiene.uniqueness_weights_spans(spans)
        assert np.all(np.isfinite(w))
        assert np.all(w > 0)
        assert abs(float(w.mean()) - 1.0) < 1e-9

    def test_missing_span_column_degrades_to_ones(self):
        # train_hybrid falls back to ones when an older frame has no label_span.
        w = net_hygiene.uniqueness_weights_spans(np.ones(100))
        assert np.allclose(w, 1.0)


class TestEmbargoCoversLabel:
    def test_label_footprint_is_imported(self):
        src = inspect.getsource(train_hybrid)
        assert "label_footprint" in src

    def test_embargo_line_calls_the_helper(self):
        src = inspect.getsource(train_hybrid._train_one_asset)
        assert "embargo_for(opt, profile)" in src

    def test_label_horizon_wins_when_it_exceeds_the_lookback(self, monkeypatch):
        monkeypatch.setenv("GTRADE_LABEL_MODE", "triple_barrier")
        monkeypatch.setenv("GTRADE_LABEL_HORIZON", "20")
        assert train_hybrid.embargo_for({"lookback": 5}, {"lookback": 5}) == 20

    def test_lookback_wins_when_it_already_covers_the_label(self, monkeypatch):
        monkeypatch.setenv("GTRADE_LABEL_MODE", "triple_barrier")
        monkeypatch.setenv("GTRADE_LABEL_HORIZON", "20")
        assert train_hybrid.embargo_for({"lookback": 60}, {"lookback": 60}) == 60

    def test_default_label_leaves_the_embargo_at_the_lookback(self, monkeypatch):
        monkeypatch.delenv("GTRADE_LABEL_MODE", raising=False)
        monkeypatch.delenv("GTRADE_LABEL_HORIZON", raising=False)
        for lb in (5, 30, 90):
            assert train_hybrid.embargo_for({"lookback": lb}, {"lookback": lb}) == lb

    def test_absurd_horizon_is_capped_at_ninety(self, monkeypatch):
        monkeypatch.setenv("GTRADE_LABEL_MODE", "triple_barrier")
        monkeypatch.setenv("GTRADE_LABEL_HORIZON", "500")
        assert train_hybrid.embargo_for({"lookback": 5}, {"lookback": 5}) == 90
        assert train_hybrid.embargo_for({"lookback": 90}, {"lookback": 90}) == 90

    def test_embargo_is_never_below_the_plain_lookback(self, monkeypatch):
        # The property that must hold no matter what the label does: this change
        # may only ever widen the purge, never narrow it.
        monkeypatch.setenv("GTRADE_LABEL_MODE", "triple_barrier")
        for horizon in ("1", "20", "500"):
            monkeypatch.setenv("GTRADE_LABEL_HORIZON", horizon)
            for lb in (5, 30, 90):
                opt = {"lookback": lb}
                assert train_hybrid.embargo_for(opt, opt) >= train_hybrid.lookback_for(opt, opt)
