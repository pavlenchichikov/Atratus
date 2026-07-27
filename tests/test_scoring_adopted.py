"""Serving refuses to score on a feature vector the model never saw."""

import pandas as pd

from core import scoring


def frame(cols):
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    data = {c: 1.0 for c in cols}
    data["close"] = 1.0
    return pd.DataFrame(data, index=idx)


def test_no_adoption_means_nothing_is_missing(monkeypatch):
    monkeypatch.setattr(scoring, "_adopted_record", lambda: None)
    assert scoring.missing_adopted_features(frame(["rsi", "atr"])) == []


def test_an_adopted_feature_present_in_the_frame_is_not_missing(monkeypatch):
    monkeypatch.setattr(scoring, "_adopted_record", lambda: {
        "genome": {"extra": [{"name": "zscore_vol_z_20"}]}})
    assert scoring.missing_adopted_features(frame(["zscore_vol_z_20"])) == []


def test_an_adopted_feature_absent_from_the_frame_is_reported(monkeypatch):
    monkeypatch.setattr(scoring, "_adopted_record", lambda: {
        "genome": {"extra": [{"name": "zscore_vol_z_20"},
                             {"name": "ratio_bb_pos_rsi"}]}})
    got = scoring.missing_adopted_features(frame(["ratio_bb_pos_rsi"]))
    assert got == ["zscore_vol_z_20"]


def test_score_asset_returns_none_when_an_adopted_feature_is_missing(
        monkeypatch, tmp_path):
    # Publishing a signal computed without a feature the model was fit on is
    # worse than publishing nothing for that asset.
    monkeypatch.setattr(scoring, "_adopted_record", lambda: {
        "genome": {"extra": [{"name": "zscore_vol_z_20"}]}})
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: printed.append(" ".join(map(str, a))))
    # A champion file must exist: score_asset returns None on a missing champion
    # BEFORE it looks at features, so without this the test would pass for the
    # wrong reason and would not notice the check being deleted.
    (tmp_path / "btc_cb.cbm").write_bytes(b"not a real model")
    out = scoring.score_asset(frame(["rsi", "atr"]), "BTC", "btc", None, {},
                              str(tmp_path))
    assert out is None
    assert any("zscore_vol_z_20" in line for line in printed), printed


def test_an_asset_with_no_champion_is_not_blamed_on_features(
        monkeypatch, tmp_path):
    # Order matters: an asset that has no champion at all must not be reported as
    # having an adopted-feature problem.
    monkeypatch.setattr(scoring, "_adopted_record", lambda: {
        "genome": {"extra": [{"name": "zscore_vol_z_20"}]}})
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: printed.append(" ".join(map(str, a))))
    out = scoring.score_asset(frame(["rsi"]), "BTC", "btc", None, {},
                              str(tmp_path))
    assert out is None
    assert printed == []
