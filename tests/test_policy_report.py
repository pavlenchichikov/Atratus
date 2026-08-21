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
    assert out["timing"]["status"] == "no data"
    assert out["timing"]["rows"] == 0


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
    assert out["timing"]["rows"] == 6           # only the logged ones


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
