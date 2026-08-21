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
