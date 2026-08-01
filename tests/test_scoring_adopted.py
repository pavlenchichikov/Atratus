"""Serving refuses to score on a feature vector the model never saw."""

import pandas as pd

from core import scoring


def frame(cols):
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    data = dict.fromkeys(cols, 1.0)
    data["close"] = 1.0
    return pd.DataFrame(data, index=idx)


def test_no_champion_list_means_nothing_is_missing():
    # An asset absent from the registry has nothing to compare against.
    assert scoring.missing_champion_features(frame(["rsi", "atr"]), None) == []
    assert scoring.missing_champion_features(frame(["rsi"]), {}) == []


def test_a_champion_feature_present_in_the_frame_is_not_missing():
    reg = {"features": ["rsi", "zscore_vol_z_20"]}
    assert scoring.missing_champion_features(
        frame(["rsi", "zscore_vol_z_20"]), reg) == []


def test_a_champion_feature_absent_from_the_frame_is_reported():
    reg = {"features": ["rsi", "zscore_vol_z_20", "ratio_bb_pos_rsi"]}
    got = scoring.missing_champion_features(frame(["rsi", "ratio_bb_pos_rsi"]),
                                            reg)
    assert got == ["zscore_vol_z_20"]


def test_it_fires_after_a_revert_too(monkeypatch):
    # The window this guard exists for: the adoption is gone, so an adopted-file
    # check would see nothing, but the champions still expect the feature until
    # the retrain finishes. Nothing here consults the adopted file.
    monkeypatch.delenv("GTRADE_ADOPTED_PATH", raising=False)
    reg = {"features": ["rsi", "zscore_vol_z_20"]}
    assert scoring.missing_champion_features(frame(["rsi"]), reg) == [
        "zscore_vol_z_20"]


def test_score_asset_returns_none_when_a_champion_feature_is_missing(
        monkeypatch, tmp_path):
    # Publishing a signal computed without a feature the model was fit on is
    # worse than publishing nothing for that asset.
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: printed.append(" ".join(map(str, a))))
    # A champion file must exist: score_asset returns None on a missing champion
    # BEFORE it looks at features, so without this the test would pass for the
    # wrong reason and would not notice the check being deleted.
    (tmp_path / "btc_cb.cbm").write_bytes(b"not a real model")
    out = scoring.score_asset(frame(["rsi", "atr"]), "BTC", "btc",
                              {"features": ["rsi", "zscore_vol_z_20"]}, {},
                              str(tmp_path))
    assert out is None
    assert any("zscore_vol_z_20" in line for line in printed), printed


def test_an_asset_with_no_champion_is_not_blamed_on_features(
        monkeypatch, tmp_path):
    # Order matters: an asset that has no champion file at all must not be
    # reported as having a feature problem.
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: printed.append(" ".join(map(str, a))))
    out = scoring.score_asset(frame(["rsi"]), "BTC", "btc",
                              {"features": ["rsi", "zscore_vol_z_20"]}, {},
                              str(tmp_path))
    assert out is None
    assert printed == []
