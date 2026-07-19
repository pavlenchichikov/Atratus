"""Tests for pretrain_foundation.py dataset builder (synthetic DB)."""
import numpy as np
import pandas as pd
import pytest
import sqlalchemy

import pretrain_foundation as pf


@pytest.fixture()
def engine(tmp_path):
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    rng = np.random.default_rng(5)
    for name in ("aaa", "bbb"):
        n = 700
        dates = pd.bdate_range("2022-01-03", periods=n)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        df = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.integers(1e5, 1e6, n),
        })
        df.to_sql(name, eng, index=False)
    return eng


class TestClassOf:
    def test_known_and_fallback(self):
        assert pf.class_of("__nope__") == "us"
        assert isinstance(pf.CLASS_GROUPS, tuple) and len(pf.CLASS_GROUPS) == 7


class TestCutoff:
    def test_cutoff_before_all_val_windows(self, engine):
        cutoff = pf.compute_cutoff(engine, ["aaa", "bbb"], max_folds=5,
                                   lookback=30, embargo=5)
        # 700 rows -> splits exist; cutoff strictly before the last date
        assert cutoff < "2024-12-31"
        assert len(cutoff) == 10  # ISO date


class TestPooled:
    def test_shapes_and_cutoff_respected(self, engine):
        cutoff = "2023-06-01"
        X, y, skipped = pf.build_pooled_sequences(
            engine, ["aaa", "bbb"], cutoff, lookback=30)
        assert skipped == []
        assert X.ndim == 3 and X.shape[1] == 30
        assert X.shape[2] == len(pf.FOUNDATION_FEATURES) + 7
        assert set(np.unique(y)) <= {0, 1}
        # class one-hot columns are constant within a sequence
        onehot = X[0, :, len(pf.FOUNDATION_FEATURES):]
        assert (onehot == onehot[0]).all()

    def test_short_table_skipped(self, engine):
        X, y, skipped = pf.build_pooled_sequences(
            engine, ["aaa"], "2022-02-01", lookback=30)  # almost no pre-cutoff rows
        assert skipped == ["aaa"]

    def test_per_asset_scaling(self, engine):
        cutoff = "2024-06-01"
        X, y, _ = pf.build_pooled_sequences(engine, ["aaa"], cutoff, lookback=30)
        feats = X[:, :, :len(pf.FOUNDATION_FEATURES)]
        # z-scored features have near-zero mean over the pooled window
        assert abs(float(feats.mean())) < 0.5


class TestTrainFoundation:
    def test_one_epoch_smoke_and_manifest(self, engine, tmp_path):
        cutoff = "2024-06-01"
        X, y, _ = pf.build_pooled_sequences(engine, ["aaa", "bbb"], cutoff,
                                            lookback=30)
        out = tmp_path / "foundation"
        metrics = pf.train_foundation(X, y, epochs=1, batch=128,
                                      out_dir=str(out))
        import os
        for f in ("lstm.keras", "transformer.keras", "tcn.keras"):
            assert os.path.exists(out / f)
        assert set(metrics) == {"lstm", "transformer", "tcn"}
        pf.write_manifest(str(out), cutoff, ["aaa", "bbb"], metrics)
        import json
        m = json.load(open(out / "manifest.json"))
        assert m["version"] == 1 and m["cutoff"] == cutoff
        assert m["features"] == list(pf.FOUNDATION_FEATURES)
        assert m["n_assets"] == 2
        # net_kwargs derives from FOUNDATION_UNITS - single source of truth,
        # exactly the kwargs train_foundation passed to each builder.
        u1 = pf.FOUNDATION_UNITS
        assert m["net_kwargs"] == {
            "lstm": {"units1": u1, "units2": max(16, u1 // 2), "head_dim": max(16, u1 // 2)},
            "transformer": {"ff_dim": u1, "num_heads": 4},
            "tcn": {"n_filters": u1},
        }
        # max_folds/embargo default to None when the caller doesn't pass them.
        assert m["max_folds"] is None
        assert m["embargo"] is None
        pf.write_manifest(str(out), cutoff, ["aaa", "bbb"], metrics, lookback=30,
                          max_folds=5, embargo=10)
        m = json.load(open(out / "manifest.json"))
        assert m["lookback"] == 30
        assert m["max_folds"] == 5
        assert m["embargo"] == 10
