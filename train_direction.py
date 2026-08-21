"""Fit and gate the direction override, on live outcomes only.

SP-3b of the RL program: the widest authority, and the only one that can make
the system trade against its own ensemble. It is fitted on LIVE rows rather
than on reconstructed history on purpose. The reconstructed backtest and the
live stream disagreed about the sign of the most basic relationship here: the
sizing rule fitted on history says bet more on confident signals, and the live
stream says confident signals are the wrong ones.

Fit on the earlier days, gate on the later ones, paired over assets, one-sided
Wilcoxon, and `follow` is in the search space so the incumbent can win.

Run:  python train_direction.py [--days 120]
"""
import argparse
import json
import os
import sqlite3
from datetime import datetime

import numpy as np

import config
from core import direction_policy as dp
from core import policy_report as pr

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
REPORT_PATH = os.path.join(BASE, "direction_report.json")

MIN_ASSETS = 8      # the same floor every other gate in this project uses
# Above this share of signals suppressed, the rule has stopped being an
# override and become an off switch, and the report has to say the word.
SILENCE_SHARE = 0.90
MIN_DAYS = 8        # below this a time split is not a split
FOREX_GROUPS = ("FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC")


def forex_assets():
    return {a for g in FOREX_GROUPS for a in config.ASSET_TYPES.get(g, [])}


def live_rows(conn, days=120):
    cur = conn.execute(
        "SELECT date, asset, signal, probability, actual_next_ret "
        "FROM prediction_log WHERE actual_next_ret IS NOT NULL "
        "AND date >= date('now', ?)", ("-%d days" % days,))
    cols = ("date", "asset", "signal", "probability", "actual_next_ret")
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def split_in_time(rows, frac=0.6):
    """(earlier, later, cut) split on the DATE. None when there are too few."""
    days = sorted({r["date"] for r in rows})
    if len(days) < MIN_DAYS:
        return None
    cut = days[int(len(days) * frac)]
    early = [r for r in rows if r["date"] < cut]
    late = [r for r in rows if r["date"] >= cut]
    if not early or not late:
        return None
    return early, late, cut


def score(rows, mode, thr, forex):
    """Mean per-asset profit under one rule."""
    per = pr.per_asset_profit(dp.apply_direction(rows, mode, thr), forex)
    return (float(np.mean(list(per.values()))) if per else float("-inf")), per


def fit(rows, forex):
    """The best (mode, thr) on these rows. `follow` is a candidate, so the
    incumbent can win and usually should."""
    best = ("follow", 0.20, *score(rows, "follow", 0.20, forex))
    for mode in dp.MODES:
        if mode == "follow":
            continue
        for thr in dp.THRESHOLDS:
            val, per = score(rows, mode, thr, forex)
            if val > best[2]:
                best = (mode, thr, val, per)
    return {"mode": best[0], "thr": best[1], "fit_profit": best[2]}


def suppression(rows, params):
    """Share of live BUY/SELL rows the rule would silence or reverse."""
    touched = total = 0
    for r in rows:
        if (r.get("signal") or "").upper() not in ("BUY", "SELL"):
            continue
        total += 1
        p = r.get("probability")
        if p is not None and abs(float(p) - 0.5) >= params["thr"]:
            touched += 1
    return (touched / total) if total else 0.0


def gate(rows, params, forex):
    """The fitted rule against `follow` on days it never saw, paired by asset."""
    from scipy.stats import wilcoxon
    base = pr.per_asset_profit(dp.apply_direction(rows, "follow"), forex)
    cand = pr.per_asset_profit(
        dp.apply_direction(rows, params["mode"], params["thr"]), forex)
    shared = sorted(set(base) & set(cand))
    deltas = [cand[a] - base[a] for a in shared]
    n = len(deltas)
    if n >= MIN_ASSETS and any(abs(d) > 1e-12 for d in deltas):
        try:
            p = float(wilcoxon(deltas, alternative="greater").pvalue)
        except ValueError:
            p = 1.0
    else:
        p = 1.0
    mean_d = float(np.mean(deltas)) if deltas else 0.0
    up = sum(1 for d in deltas if d > 0)
    verdict = "ADOPT" if (n >= MIN_ASSETS and p < 0.05 and mean_d > 0) else "HOLD"
    return {"verdict": verdict, "p": p, "mean_d": mean_d, "n": n, "up": up,
            "median_d": float(np.median(deltas)) if deltas else 0.0,
            "per_asset": {a: round(cand[a] - base[a], 5) for a in shared}}


def main():
    ap = argparse.ArgumentParser(description="fit and gate the direction rule")
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    rows = live_rows(conn, args.days)
    conn.close()
    split = split_in_time(rows)
    if split is None:
        print("[direction] not enough distinct live days to split in time. "
              "Nothing fitted: a rule judged on the days it was chosen on has "
              "seen its own answer.")
        return
    early, late, cut = split
    forex = forex_assets()
    params = fit(early, forex)
    g = gate(late, params, forex)
    print("[direction] fitted on days before %s (%d rows), gated on %s onward "
          "(%d rows)" % (cut, len(early), cut, len(late)))
    print("[direction] best rule on the fit window: %s at |prob-0.5| >= %.2f "
          "(mean per-asset profit %+.4f)"
          % (params["mode"], params["thr"], params["fit_profit"]))
    print("[direction] held out: mean %+.4f  median %+.4f  better on %d of %d "
          "assets  p=%.4f" % (g["mean_d"], g["median_d"], g["up"], g["n"],
                              g["p"]))
    print("[direction] VERDICT: %s" % g["verdict"])
    silenced = suppression(late, params)
    if params["mode"] != "follow" and silenced >= SILENCE_SHARE:
        print("[direction] READ THIS BEFORE ACTING ON IT: the rule suppresses "
              "%.1f%% of the signals, so what passed the gate is not a clever "
              "override, it is NOT TRADING. On this live window standing aside "
              "beat following the ensemble, which is a statement about the "
              "signal, not about the overlay." % (silenced * 100.0))
    report_extra = {"suppressed": silenced}
    if params["mode"] == "follow":
        print("[direction] the incumbent won: nothing in the search space beat "
              "following the ensemble on the fit window.")
    report = dict(params, **{k: v for k, v in g.items() if k != "per_asset"})
    report.update(report_extra if params["mode"] != "follow"
                  else {"suppressed": 0.0})
    report["per_asset"] = g["per_asset"]
    report["cut"] = cut
    report["fitted"] = datetime.now().isoformat(timespec="seconds")
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("[direction] nothing is served. This writes direction_report.json "
          "and no serve path reads it.")


if __name__ == "__main__":
    main()
