"""What the policy layers concluded, and what they were worth on live signals.

The distinction this file defends: a backtest verdict and a live reading are
different claims, and an arm with nothing logged has NO result rather than a
neutral one. The two were confused once in this project already.
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import policy_report as pr


def _row(asset, date, signal, ret, action=None, prob=0.6):
    return {"asset": asset, "date": date, "signal": signal,
            "probability": prob, "actual_next_ret": ret,
            "timing_action": action}


def test_the_logged_timing_decision_is_replayed_not_recomputed():
    """ENTER opens on the signal's side, HOLD keeps it, EXIT and STAY_OUT go
    flat. Nothing is recomputed, so nothing can be improved with hindsight."""
    sig = np.array([1, 1, 1, -1, -1])
    actions = ["ENTER", "HOLD", "EXIT", "STAY_OUT", "ENTER"]
    assert list(pr._timing_sides(sig, actions)) == [1, 1, 0, 0, -1]


def test_a_bar_with_no_logged_decision_falls_back_to_the_raw_signal():
    """A gap in the shadow column is a gap, not an instruction to stand aside:
    what production actually did on that bar was follow the signal."""
    sig = np.array([1, -1])
    assert list(pr._timing_sides(sig, [None, None])) == [1, -1]


def test_an_arm_with_nothing_logged_reads_as_no_data_not_as_neutral():
    """The positive control for the whole reconciliation. A timing column that
    was never populated must not be reported as a zero result."""
    rows = [_row("A", "2026-08-%02d" % d, "BUY", 0.01) for d in range(1, 12)]
    out = pr.reconcile(rows)
    assert out["emitted"]["status"] == "measured"
    for arm in ("timing A", "timing B"):
        assert out[arm]["status"] == "no data"
        assert out[arm]["rows"] == 0


def test_the_emitted_arm_earns_what_the_signal_earned():
    rows = [_row("A", "2026-08-%02d" % d, "BUY", 0.01) for d in range(1, 12)]
    out = pr.reconcile(rows)
    assert out["emitted"]["profit"] > 0
    flipped = [_row("A", "2026-08-%02d" % d, "SELL", 0.01) for d in range(1, 12)]
    assert pr.reconcile(flipped)["emitted"]["profit"] < 0


def test_coverage_counts_only_the_rows_an_arm_actually_had():
    rows = [_row("A", "2026-08-%02d" % d, "BUY", 0.01,
                 action=("HOLD" if d > 5 else None)) for d in range(1, 12)]
    out = pr.reconcile(rows)
    assert out["emitted"]["rows"] == 11
    # A row with no stage is Stage A's: it is the only policy that had ever
    # served before the column existed.
    assert out["timing A"]["rows"] == 6         # only the logged ones
    assert out["timing B"]["status"] == "no data"


def test_a_missing_report_is_reported_as_missing(tmp_path):
    """A policy nobody has fitted and a policy that failed are different
    states, and a report that quietly skipped the first would say neither."""
    rows = {r["name"]: r for r in pr.reports(str(tmp_path))}
    assert all(r["present"] is False for r in rows.values())
    (tmp_path / "sizing_report.json").write_text(json.dumps(
        {"verdict": "ADOPT", "n": 207, "mean_d": 9.0,
         "per_asset": {"A": 1.0, "B": -1.0, "C": 3.0}}), encoding="utf-8")
    got = {r["name"]: r for r in pr.reports(str(tmp_path))}["sizing"]
    assert got["present"] and got["verdict"] == "ADOPT"
    assert got["median_d"] == 1.0 and got["up"] == 2 and got["down"] == 1


def test_the_two_timing_stages_are_never_blended_into_one_number():
    """They run over different days, so one number over both would describe a
    policy that never existed. Added when Stage B started logging live on
    2026-08-22 beside Stage A's 2882 existing rows."""
    rows = []
    for d in range(1, 8):                       # Stage A's week, winning
        r = _row("A", "2026-08-%02d" % d, "BUY", 0.01,
                 action=("ENTER" if d == 1 else "HOLD"))
        r["timing_stage"] = "A"
        rows.append(r)
    for d in range(8, 15):                      # Stage B's week, losing
        r = _row("A", "2026-08-%02d" % d, "BUY", -0.01,
                 action=("ENTER" if d == 8 else "HOLD"))
        r["timing_stage"] = "B"
        rows.append(r)
    out = pr.reconcile(rows)
    assert out["timing A"]["rows"] == 7
    assert out["timing B"]["rows"] == 7
    assert out["timing A"]["profit"] > 0
    assert out["timing B"]["profit"] < 0, "B's losses must not hide inside A"


def test_a_stage_is_scored_only_on_the_days_it_decided():
    """Filling the other stage's days with the raw signal made the answer turn
    on where a stage's window started: a window opening on HOLD has nothing to
    hold. Measured, before the fix: the SAME losing week read -0.72 or -0.03
    depending only on which half of the log it sat in."""
    winning = [dict(_row("A", "2026-08-%02d" % d, "BUY", 0.01,
                         action=("ENTER" if d == 1 else "HOLD")),
                    timing_stage="A") for d in range(1, 8)]
    losing = [dict(_row("A", "2026-08-%02d" % d, "BUY", -0.01,
                        action=("ENTER" if d == 8 else "HOLD")),
                   timing_stage="B") for d in range(8, 15)]

    both = pr.reconcile(winning + losing)
    alone = pr.reconcile(losing)
    assert both["timing B"]["profit"] == alone["timing B"]["profit"], (
        "B's number must not move because A logged days beside it")


def test_a_row_from_before_the_stage_column_counts_as_stage_a():
    rows = [_row("A", "2026-08-%02d" % d, "BUY", 0.01, action="HOLD")
            for d in range(1, 8)]              # no timing_stage key at all
    out = pr.reconcile(rows)
    assert out["timing A"]["rows"] == 7
    assert out["timing B"]["status"] == "no data"
