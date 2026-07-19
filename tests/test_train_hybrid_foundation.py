"""Foundation integration unit tests (no real training - shapes/mocks only)."""
import numpy as np


def _th():
    import train_hybrid
    return train_hybrid


class TestFoundationGate:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("GTRADE_FOUNDATION", raising=False)
        assert _th().foundation_on() is False

    def test_on(self, monkeypatch):
        monkeypatch.setenv("GTRADE_FOUNDATION", "1")
        assert _th().foundation_on() is True


class TestLoadFoundation:
    def test_missing_dir_returns_none(self, monkeypatch, tmp_path):
        th = _th()
        monkeypatch.setattr(th, "_FOUNDATION_DIR", str(tmp_path / "nope"),
                            raising=False)
        monkeypatch.setattr(th, "_FOUNDATION_CACHE", None, raising=False)
        assert th._load_foundation() is None

    def test_corrupt_manifest_returns_none(self, monkeypatch, tmp_path):
        th = _th()
        d = tmp_path / "foundation"
        d.mkdir()
        (d / "manifest.json").write_text("{not json")
        monkeypatch.setattr(th, "_FOUNDATION_DIR", str(d), raising=False)
        monkeypatch.setattr(th, "_FOUNDATION_CACHE", None, raising=False)
        assert th._load_foundation() is None

    def test_caches_result(self, monkeypatch, tmp_path):
        """Once cached (True or False), a second call must not re-touch disk."""
        th = _th()
        monkeypatch.setattr(th, "_FOUNDATION_DIR", str(tmp_path / "nope"),
                            raising=False)
        monkeypatch.setattr(th, "_FOUNDATION_CACHE", None, raising=False)
        assert th._load_foundation() is None
        assert th._FOUNDATION_CACHE is False
        # Second call short-circuits on the cached False, no re-raise/log.
        assert th._load_foundation() is None

    def test_refuses_when_run_max_folds_exceeds_manifest_cutoff(self, monkeypatch, tmp_path):
        """Training with more folds than the foundation's own cutoff protected
        against would leak eval bars into pretraining - _load_foundation must
        refuse (return None) rather than best-effort-load anyway. The check
        happens before any weight file is touched, so a manifest with no
        weight files present is enough to exercise the refusal path."""
        import json

        th = _th()
        d = tmp_path / "foundation"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"max_folds": 5}))
        monkeypatch.setattr(th, "_FOUNDATION_DIR", str(d), raising=False)
        monkeypatch.setattr(th, "_FOUNDATION_CACHE", None, raising=False)
        monkeypatch.setenv("GTRADE_MAX_FOLDS", "10")
        assert th._load_foundation() is None
        assert th._FOUNDATION_CACHE is False


class TestWsLoad:
    def test_returns_true_when_applied_false_otherwise(self, monkeypatch):
        th = _th()
        monkeypatch.setattr(th, "_NET_WARMSTART", True, raising=False)

        class _Model:
            def __init__(self):
                self.set = None

            def set_weights(self, w):
                self.set = w

        class _BadModel:
            def set_weights(self, w):
                raise ValueError("shape mismatch")

        # Nothing staged in the store - nothing to load.
        assert th._ws_load(_Model(), "lstm", {}) is False
        # Shapes match - loads and returns True.
        assert th._ws_load(_Model(), "lstm", {"lstm": [np.zeros(3)]}) is True
        # Shapes don't match - set_weights raises, falls back, returns False.
        assert th._ws_load(_BadModel(), "lstm", {"lstm": [np.zeros(3)]}) is False


class TestSeedWarm:
    def test_seeds_all_three_keys(self):
        th = _th()
        warm = {}
        foundation = {"manifest": {}, "weights": {
            "lstm": [np.zeros(2)], "tf": [np.ones(2)], "tcn": [np.ones(3)]}}
        th._foundation_seed_warm(warm, foundation, "TEST")
        assert set(warm) == {"lstm", "tf", "tcn"}
        assert (warm["tf"][0] == 1).all()


class TestFoundationOnehot:
    def test_known_class_sets_one_slot(self):
        th = _th()
        classes = ["crypto", "us", "eu"]
        v = th._foundation_onehot("BTC", classes)
        assert v.shape == (3,)
        assert v.sum() == 1.0
        assert v[classes.index("crypto")] == 1.0

    def test_unrecognized_asset_falls_back_to_us(self):
        th = _th()
        classes = ["crypto", "us", "eu"]
        v = th._foundation_onehot("__totally_unknown_ticker__", classes)
        assert v[classes.index("us")] == 1.0


class TestFitFrozenThenFull:
    def test_freezes_then_unfreezes_and_calls_fit_twice(self):
        th = _th()

        class _Layer:
            def __init__(self):
                self.trainable = True

        class _Model:
            def __init__(self):
                self.layers = [_Layer(), _Layer(), _Layer()]
                self.compiled = 0

            def compile(self, **kwargs):
                self.compiled += 1

        model = _Model()
        calls = []

        def fit_fn(n):
            # Record trainable flags of every layer at call time.
            calls.append((n, [ly.trainable for ly in model.layers]))

        th._fit_frozen_then_full(model, 5, model.compile, fit_fn)

        assert model.compiled == 2
        assert len(calls) == 2
        # Stage 1: frozen (all but the last Dense), for freeze_epochs.
        n0, trainable0 = calls[0]
        assert n0 == 5
        assert trainable0 == [False, False, True]
        # Stage 2: everything unfrozen, fit_fn receives None (remaining).
        n1, trainable1 = calls[1]
        assert n1 is None
        assert trainable1 == [True, True, True]
