"""Unit tests for core.ar_progress: the estimate math is pure and testable."""

import datetime
import os
import time

from core import ar_progress


def test_median_ignores_junk_and_handles_even_counts():
    assert ar_progress.median([3, 1, 2]) == 2.0
    assert ar_progress.median([1, 3]) == 2.0
    assert ar_progress.median([]) is None
    assert ar_progress.median(None) is None
    assert ar_progress.median(["x", None, -5, 4]) == 4.0


def test_no_pending_assets_means_nothing_left():
    est, basis = ar_progress.unit_remaining([], {"A": [10]}, workers=4)
    assert est == 0.0
    assert "finished" in basis


def test_expensive_asset_last_is_not_the_average():
    # Three cheap assets already done, the slow one is what remains. The per-asset
    # median correctly prices the remaining work, not the pooled median of history.
    hist = {"C1": [60], "C2": [60], "C3": [60], "SLOW": [3600]}
    est, basis = ar_progress.unit_remaining(["SLOW"], hist, workers=4)
    pooled_median = ar_progress.median([60, 60, 60, 3600])
    assert est == 3600.0
    assert est > pooled_median
    assert "per-asset history" in basis


def test_parallel_workers_do_not_add_up():
    hist = {"A0": [3600], "A1": [3600], "A2": [3600], "A3": [3600]}
    pending = ["A0", "A1", "A2", "A3"]
    parallel, _b = ar_progress.unit_remaining(pending, hist, workers=4)
    assert parallel == 3600.0          # four lanes, four equal assets
    serial, _b2 = ar_progress.unit_remaining(pending, hist, workers=1)
    assert serial == 4 * 3600.0        # one lane: they queue


def test_one_long_asset_left_ignores_worker_count():
    hist = {"USDJPY": [7200]}
    for workers in (1, 4, 16):
        est, _b = ar_progress.unit_remaining(["USDJPY"], hist, workers=workers)
        assert est == 7200.0


def test_unknown_asset_falls_back_to_the_median_of_known_ones():
    hist = {"A": [100], "B": [300]}
    est, basis = ar_progress.unit_remaining(["A", "NEWCO"], hist, workers=1)
    assert est == 100.0 + 200.0        # NEWCO priced at median(100, 300)
    assert "unknown assets" in basis


def test_without_per_asset_history_it_uses_the_unit_kind_median():
    est, basis = ar_progress.unit_remaining(
        ["A", "B"], {}, workers=4, unit_kind="holdout_14",
        unit_hist={"holdout_14": [30000, 40000]})
    assert est == 35000.0
    assert "unit-kind median" in basis


def test_without_any_history_it_says_so_instead_of_guessing():
    est, basis = ar_progress.unit_remaining(["A"], {}, workers=4,
                                            unit_kind="holdout_14", unit_hist={})
    assert est is None
    assert basis == "no history yet"


def test_multiple_different_cost_assets_on_single_lane():
    # Several pending assets with different costs show why per-asset medians
    # matter: a naive average-based estimate is visibly wrong.
    hist = {"A": [90, 100, 110], "B": [190, 200], "C": [490, 500, 510]}
    pending = ["A", "B", "C"]
    est, _basis = ar_progress.unit_remaining(pending, hist, workers=1)

    # Correct: sum of per-asset medians
    correct = 100.0 + 195.0 + 500.0
    assert est == correct

    # Naive: average of all history times length of pending
    all_times = [90, 100, 110, 190, 200, 490, 500, 510]
    avg_all = sum(all_times) / len(all_times)
    naive = len(pending) * avg_all

    # They differ; per-asset sum beats naive average
    assert est != naive


def test_unit_remaining_defends_against_non_numeric_workers():
    # workers parameter may come from JSON and be corrupted; should not raise.
    hist = {"A": [100]}
    est, basis = ar_progress.unit_remaining(["A"], hist, workers="four")
    assert est == 100.0

    est, basis = ar_progress.unit_remaining(["A"], hist, workers=None)
    assert est == 100.0


def test_run_remaining_defends_against_non_numeric_unit_left():
    # unit_left parameter may come from JSON and be corrupted; should not raise.
    est, basis = ar_progress.run_remaining("soon", ["tier_4"], {"tier_4": [4000]})
    assert est is None
    assert basis == "no history yet"


def test_run_remaining_adds_the_median_of_each_unit_not_started():
    est, basis = ar_progress.run_remaining(
        1000.0, ["tier_4", "holdout_14"],
        {"tier_4": [4000], "holdout_14": [30000, 40000]})
    assert est == 1000.0 + 4000.0 + 35000.0
    assert "unit-kind medians" in basis


def test_run_remaining_names_unmeasured_future_units():
    est, basis = ar_progress.run_remaining(1000.0, ["tier_4", "mystery"],
                                            {"tier_4": [4000]})
    assert est == 5000.0
    assert "1 future unit" in basis


def test_run_remaining_without_a_current_estimate_is_unknown():
    est, basis = ar_progress.run_remaining(None, ["tier_4"], {"tier_4": [4000]})
    assert est is None
    assert basis == "no history yet"


def _isolate(monkeypatch, tmp_path):
    """Point both progress files at tmp_path so tests never touch the repo root."""
    agent = str(tmp_path / "ar_progress.json")
    unit = str(tmp_path / "ar_progress_unit.json")
    monkeypatch.setattr(ar_progress, "AGENT_FILE", agent)
    monkeypatch.setattr(ar_progress, "UNIT_FILE", unit)
    return agent, unit


def test_write_then_read_roundtrip_stamps_pid_and_time(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.write_agent({"phase": "search", "step": {"i": 3, "n": 15}})
    rec = ar_progress.read_agent()
    assert rec["phase"] == "search"
    assert rec["step"]["n"] == 15
    assert rec["pid"] == os.getpid()
    assert rec["updated_at"]


def test_missing_and_corrupt_files_read_as_no_data(monkeypatch, tmp_path):
    agent, _unit = _isolate(monkeypatch, tmp_path)
    assert ar_progress.read_agent() == {}
    with open(agent, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert ar_progress.read_agent() == {}
    with open(agent, "w", encoding="utf-8") as fh:
        fh.write("[1, 2, 3]")
    assert ar_progress.read_agent() == {}


def test_write_leaves_no_temp_file_behind(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.write_agent({"phase": "gate"})
    assert [p.name for p in tmp_path.iterdir()] == ["ar_progress.json"]


def test_a_failing_write_never_raises(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    def boom(*_a, **_kw):
        raise OSError("disk gone")

    monkeypatch.setattr("builtins.open", boom)
    ar_progress.write_agent({"phase": "gate"})     # must return normally
    ar_progress.write_unit({"assets_total": 14})   # must return normally


def test_a_failing_replace_never_raises_and_cleans_up_temp_file(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    def boom(*_a, **_kw):
        raise OSError("replace failed: destination locked")

    monkeypatch.setattr(ar_progress.os, "replace", boom)
    ar_progress.write_agent({"phase": "gate"})     # must return normally
    assert list(tmp_path.iterdir()) == []          # no leftover *.tmp file


def test_age_and_staleness(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    now = datetime.datetime(2026, 7, 25, 12, 0, 0)
    fresh = {"updated_at": (now - datetime.timedelta(seconds=30)).isoformat(timespec="seconds")}
    old = {"updated_at": (now - datetime.timedelta(seconds=3600)).isoformat(timespec="seconds")}
    assert ar_progress.age_seconds(fresh, now) == 30
    assert ar_progress.age_seconds(old, now) == 3600
    assert ar_progress.age_seconds({}, now) is None
    assert ar_progress.age_seconds({"updated_at": "not a date"}, now) is None
    assert ar_progress.age_seconds(["not", "a", "dict"], now) is None


def test_snapshot_without_files_reports_no_data(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    snap = ar_progress.snapshot()
    assert snap["state"] == "no data"


def test_snapshot_merges_agent_and_unit_and_estimates(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    now = datetime.datetime(2026, 7, 25, 12, 0, 0)
    ar_progress.write_agent({
        "phase": "gate",
        "step": {"i": 2, "n": 3, "kind": "elite_holdout", "unit_kind": "holdout_14"},
        "pending_units": ["tier_4", "holdout_14"],
        "results": {"tried": 985, "adoptable": 2, "replicated": 0},
        "history": {"holdout_14": [30000], "tier_4": [4000],
                     "assets": {"SLOW": [3600], "FAST": [600]}},
    })
    ar_progress.write_unit({
        "workers": 4, "assets_total": 2, "order": ["FAST", "SLOW"],
        "done": [["FAST", 600]],
    })
    snap = ar_progress.snapshot(now)
    assert snap["state"] == "running"
    assert snap["phase"] == "gate"
    assert snap["step"]["i"] == 2
    assert snap["unit"]["assets_done"] == 1
    assert snap["unit"]["pending"] == ["SLOW"]
    assert snap["eta"]["unit_left_s"] == 3600.0          # only SLOW is left
    assert snap["eta"]["run_left_s"] == 3600.0 + 4000.0 + 30000.0
    assert "per-asset history" in snap["eta"]["basis"]
    assert snap["results"]["adoptable"] == 2


def test_snapshot_marks_a_silent_run_stale(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.write_agent({"phase": "gate"})
    later = datetime.datetime.now() + datetime.timedelta(seconds=ar_progress.STALE_AFTER_S + 60)
    snap = ar_progress.snapshot(later)
    assert snap["stale"] is True
    assert snap["age_s"] > ar_progress.STALE_AFTER_S


def test_snapshot_survives_malformed_nested_structures(monkeypatch, tmp_path):
    # A hand-edited or partially-written file can put anything where a nested dict
    # is expected. snapshot() must degrade, never raise.
    _isolate(monkeypatch, tmp_path)

    ar_progress.write_agent({"phase": "gate", "history": ["not", "a", "dict"]})
    snap = ar_progress.snapshot()
    assert snap["state"] == "running"

    ar_progress.write_agent({"phase": "gate", "step": "not a dict"})
    snap = ar_progress.snapshot()
    assert snap["state"] == "running"

    ar_progress.write_agent({"phase": "gate", "history": {"assets": 42}})
    snap = ar_progress.snapshot()
    assert snap["state"] == "running"


def test_heartbeat_refreshes_updated_at(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(ar_progress, "HEARTBEAT_S", 0.05)
    ar_progress.write_agent({"phase": "gate"})
    first = ar_progress.read_agent()["updated_at"]
    ar_progress.start_heartbeat("agent")
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            if ar_progress.read_agent().get("beats"):
                break
            time.sleep(0.05)
    finally:
        ar_progress.stop_heartbeat()
    assert ar_progress.read_agent().get("beats")
    assert ar_progress.read_agent()["phase"] == "gate"    # heartbeat preserves content
    assert first is not None
