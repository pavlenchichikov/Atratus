"""Threshold scale A/B on the live journal: calibrated (served) vs model (tuned).

Training tunes buy/sell thresholds on the ensemble output BEFORE calibration;
serving compares them with the CALIBRATED probability. prediction_log.model_prob
(from 2026-10-06) keeps the uncalibrated value, so both scales can be scored on
the same verified bars:

  calibrated  the served call (`signal`)
  model       model_prob against the asset's current tuned thresholds

Accuracy on calls only (a WAIT is not scored), plus how many bars each scale
calls. Fewer, more extreme calls are naturally more accurate, so the coverage
column is part of the answer, not a footnote.

    python ab_thr_scale.py [--since YYYY-MM-DD] [--min-calls 20]
"""
import argparse
import json
import os
import sqlite3

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def _side(p, buy, sell):
    return "BUY" if p > buy else ("SELL" if p < sell else "WAIT")


def _hit(sig, ret):
    return (sig == "BUY" and ret > 0) or (sig == "SELL" and ret < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-06")
    ap.add_argument("--min-calls", type=int, default=20)
    a = ap.parse_args()

    with open(os.path.join(BASE, "models", "tuned_thresholds.json")) as f:
        thr = json.load(f)
    con = sqlite3.connect(os.path.join(BASE, "market.db"))
    if "model_prob" not in {r[1] for r in con.execute("PRAGMA table_info(prediction_log)")}:
        print("prediction_log has no model_prob yet: merge thr-scale-shadow and run predict.")
        return
    rows = con.execute(
        "SELECT asset, signal, model_prob, actual_next_ret FROM prediction_log "
        "WHERE model_prob IS NOT NULL AND actual_next_ret IS NOT NULL "
        "AND date >= ?", (a.since,)).fetchall()
    con.close()

    per = {}
    n_bars = 0
    for asset, served, mp, ret in rows:
        t = thr.get(asset)
        if not t:
            continue
        n_bars += 1
        model = _side(mp, t["buy"], t["sell"])
        st = per.setdefault(asset, {"calibrated": [0, 0], "model": [0, 0]})
        for arm, sig in (("calibrated", served), ("model", model)):
            if sig in ("BUY", "SELL"):
                st[arm][0] += 1
                st[arm][1] += _hit(sig, ret)

    if not n_bars:
        print(f"No verified rows with model_prob since {a.since} yet.")
        return
    print(f"verified bars {n_bars}, assets {len(per)}, since {a.since}")
    for arm in ("calibrated", "model"):
        n = sum(v[arm][0] for v in per.values())
        h = sum(v[arm][1] for v in per.values())
        print(f"  {arm:10s} calls {n:6d} ({n / n_bars:5.1%} of bars)  "
              f"acc {h / max(n, 1):.4f}")
    ds = np.array([v["model"][1] / v["model"][0] - v["calibrated"][1] / v["calibrated"][0]
                   for v in per.values()
                   if min(v["model"][0], v["calibrated"][0]) >= a.min_calls])
    if len(ds) > 1:
        t = ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds)))
        print(f"  model - calibrated per asset (>= {a.min_calls} calls each): "
              f"mean {ds.mean():+.4f}  wins {int((ds > 0).sum())}/{len(ds)}  t {t:+.2f}")
    else:
        print(f"  per-asset comparison needs >= {a.min_calls} calls on both scales; "
              "not enough yet")


if __name__ == "__main__":
    main()
