"""The health checks that decide whether an asset is really serving.

--mismatched compares two timestamps and can therefore report a clean registry
while assets serve on fewer members than the registry claims: the file is
readable to the trainer and not to the serving environment. --degraded exists
because of that, so what it must prove is that it OPENS things.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_health


def _repo(tmp_path, assets):
    mdir = tmp_path / "models"
    mdir.mkdir()
    reg = {}
    for asset in assets:
        reg[asset] = {"features": ["a", "b"], "lookback": 20,
                      "updated_at": "2026-08-21T10:00:00"}
        for member in ("lstm", "transformer", "tcn"):
            (mdir / ("%s_%s.keras" % (asset.lower(), member))).write_bytes(b"x")
    (mdir / "champion_registry.json").write_text(json.dumps(reg), encoding="utf-8")
    return str(tmp_path)


def test_degraded_names_the_members_that_did_not_load(tmp_path, monkeypatch):
    base = _repo(tmp_path, ["GOOD", "HALF", "DEAD"])
    fails = {"half_transformer", "dead_lstm", "dead_transformer", "dead_tcn"}

    def fake_lstm(path, lookback, n_features):
        stem = os.path.basename(path).rsplit(".", 1)[0]
        return (None if stem in fails else object()), "mode", lookback

    def fake_native(path):
        stem = os.path.basename(path).rsplit(".", 1)[0]
        return None if stem in fails else object()

    monkeypatch.setattr("core.model_io.load_lstm_model", fake_lstm)
    monkeypatch.setattr("core.model_io.load_keras_native", fake_native)

    rows = {r["asset"]: r["lost"] for r in model_health.degraded_members(base)}
    assert "GOOD" not in rows                       # a loadable asset is not flagged
    assert rows["HALF"] == ["transformer"]
    assert rows["DEAD"] == ["lstm", "transformer", "tcn"]


def test_a_timestamp_check_cannot_see_an_unreadable_champion(tmp_path, monkeypatch):
    """The positive control for the whole point of --degraded: the registry it
    calls healthy is the same one whose champions do not load."""
    base = _repo(tmp_path, ["DEAD"])
    monkeypatch.setattr("core.model_io.load_lstm_model",
                        lambda *a, **k: (None, "CB ONLY (Err)", 20))
    monkeypatch.setattr("core.model_io.load_keras_native", lambda *a, **k: None)
    assert model_health.mismatched_registry(base) == []
    assert model_health.degraded_members(base)[0]["asset"] == "DEAD"


def test_unservable_champions_names_the_assets_the_current_genome_cannot_feed(monkeypatch):
    """After adopting axis:qd+ref, 83 champions still wanted the previous
    genome's synthetic features, so core.scoring skipped them and those assets
    went silent while their registry entry looked healthy."""
    import model_health as mh
    from core import feature_dsl, features

    monkeypatch.setattr(features, "active_candidate_features",
                        lambda: ["ret_1", "rsi"])
    monkeypatch.setattr(feature_dsl, "load_dsl_specs",
                        lambda: [{"name": "m74807", "op": "ratio"}])
    registry = {
        "NEW": {"features": ["ret_1", "rsi", "m74807"], "updated_at": "2026-09-23"},
        "OLD": {"features": ["ret_1", "m59561"], "updated_at": "2026-09-14"},
        # Raw price columns are in every frame and in no candidate list.
        "RAW": {"features": ["ret_1", "close", "volume"], "updated_at": "2026-03-20"},
    }
    # AVB left the asset map long ago: nothing scans it, so nothing skips it,
    # and listing it made the trainer refuse the whole run on 2026-09-23.
    registry["AVB"] = {"features": ["m59561"], "updated_at": "2026-08-25"}
    monkeypatch.setattr(mh, "FULL_ASSET_MAP", {"NEW": 1, "OLD": 2, "RAW": 3})

    rows = mh.unservable_champions(registry)
    assert [r["asset"] for r in rows] == ["OLD"]
    assert rows[0]["missing"] == ["m59561"]
    assert mh._collect("unservable") is not None
