"""Unit tests for core.ar_progress: the estimate math is pure and testable."""

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
    # Three cheap assets already done, the slow one is what remains. A naive
    # average-based estimate would say 945 s; the truth is 3600 s.
    hist = {"C1": [60], "C2": [60], "C3": [60], "SLOW": [3600]}
    est, basis = ar_progress.unit_remaining(["SLOW"], hist, workers=4)
    naive = ar_progress.median([60, 60, 60, 3600])
    assert est == 3600.0
    assert est > naive
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
