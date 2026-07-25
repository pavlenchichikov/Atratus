"""The agent publishes phase and step, and folds measured durations into history."""

import auto_research
from core import ar_progress


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "ar_progress.json"))
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "ar_progress_unit.json"))


def test_publish_writes_phase_step_and_seeds_history(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("search", step={"i": 3, "n": 15, "kind": "screen"})
    rec = ar_progress.read_agent()
    assert rec["phase"] == "search"
    assert rec["step"] == {"i": 3, "n": 15, "kind": "screen"}
    # The measured unit-kind history from the 2026-07-23 run is seeded, so the very
    # first run after this change already has a basis for an estimate.
    assert rec["history"]["holdout_14"]
    assert rec["history"]["assets"] == {}


def test_publish_keeps_history_across_calls(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("search", step={"i": 1, "n": 15})
    auto_research._progress_fold_unit("holdout_14", 12345)
    auto_research._progress_publish("gate", step={"i": 1, "n": 3})
    rec = ar_progress.read_agent()
    assert 12345 in rec["history"]["holdout_14"]
    assert rec["phase"] == "gate"


def test_fold_unit_absorbs_per_asset_service_times(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("gate", step={"i": 1, "n": 3})
    ar_progress.write_unit({"done": [["USDJPY", 7200], ["NASDAQ", 600]]})
    auto_research._progress_fold_unit("holdout_14", 9000)
    hist = ar_progress.read_agent()["history"]
    assert hist["assets"]["USDJPY"] == [7200]
    assert hist["assets"]["NASDAQ"] == [600]
    assert 9000 in hist["holdout_14"]


def test_history_lists_are_capped(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("gate", step={"i": 1, "n": 3})
    for i in range(30):
        auto_research._progress_fold_unit("tier_4", 1000 + i)
    kept = ar_progress.read_agent()["history"]["tier_4"]
    assert len(kept) <= auto_research.PROGRESS_KEEP
    assert 1029 in kept        # the newest measurement survives


def test_publish_never_raises_when_storage_fails(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(ar_progress, "write_agent", lambda *_a, **_k: (_ for _ in ()).throw(OSError("x")))
    auto_research._progress_publish("search", step={"i": 1, "n": 15})   # must return
