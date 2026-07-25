"""The trainer's unit record: many threads, one writer, service times recorded."""

import threading
import time

from core import ar_progress


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "ar_progress_unit.json"))
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "ar_progress.json"))


def test_unit_records_order_workers_and_progress(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.unit_begin(["A", "B", "C"], workers=2)
    rec = ar_progress.read_unit()
    assert rec["order"] == ["A", "B", "C"]
    assert rec["workers"] == 2
    assert rec["assets_total"] == 3
    assert rec["done"] == []
    ar_progress.unit_asset_start("A")
    ar_progress.unit_asset_start("B")
    assert sorted(ar_progress.read_unit()["in_flight"]) == ["A", "B"]
    ar_progress.unit_asset_done("A")
    rec = ar_progress.read_unit()
    assert rec["in_flight"] == ["B"]
    assert rec["done"][0][0] == "A"
    assert isinstance(rec["done"][0][1], int)


def test_service_time_is_measured_per_asset_not_wall_clock(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.unit_begin(["SLOW", "FAST"], workers=2)
    ar_progress.unit_asset_start("SLOW")
    time.sleep(0.25)
    ar_progress.unit_asset_start("FAST")
    ar_progress.unit_asset_done("FAST")
    took = dict(ar_progress.read_unit()["done"])["FAST"]
    # FAST started late, so its own service time must not include SLOW's wait.
    assert took < 1


def test_done_for_an_asset_that_never_started_is_tolerated(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.unit_begin(["A"], workers=1)
    ar_progress.unit_asset_done("A")          # must not raise
    assert dict(ar_progress.read_unit()["done"])["A"] == 0


def test_parallel_writers_do_not_lose_entries(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    names = ["A%d" % i for i in range(24)]
    ar_progress.unit_begin(names, workers=8)

    def work(name):
        ar_progress.unit_asset_start(name)
        ar_progress.unit_asset_done(name)

    threads = [threading.Thread(target=work, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec = ar_progress.read_unit()
    assert len(rec["done"]) == 24
    assert rec["in_flight"] == []


def test_unit_end_clears_in_flight(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.unit_begin(["A"], workers=1)
    ar_progress.unit_asset_start("A")
    ar_progress.unit_end()
    assert ar_progress.read_unit()["in_flight"] == []


def test_unit_end_trims_order_to_assets_that_ran(monkeypatch, tmp_path):
    # Ctrl+C stops the submit loop early: three of five assets are never
    # submitted, so they never start and never finish. unit_end() must not
    # leave them looking "pending" on an ended unit.
    _isolate(monkeypatch, tmp_path)
    ar_progress.unit_begin(["A", "B", "C", "D", "E"], workers=2)
    ar_progress.unit_asset_start("A")
    ar_progress.unit_asset_done("A")
    ar_progress.unit_asset_start("B")
    ar_progress.unit_asset_done("B")
    ar_progress.unit_end()
    snap = ar_progress.snapshot()
    assert snap["unit"]["pending"] == []
    assert snap["eta"]["unit_left_s"] == 0.0


def test_unit_end_normal_completion_still_reports_no_pending(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    ar_progress.unit_begin(["A", "B"], workers=2)
    ar_progress.unit_asset_start("A")
    ar_progress.unit_asset_done("A")
    ar_progress.unit_asset_start("B")
    ar_progress.unit_asset_done("B")
    ar_progress.unit_end()
    snap = ar_progress.snapshot()
    assert snap["unit"]["pending"] == []
    assert snap["eta"]["unit_left_s"] == 0.0
