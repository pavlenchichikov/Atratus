"""Fit and gate the position-sizing rule.

The same offline shape as train_timing.py: the same reconstructed per-asset
history, the same separable ES, the same time split, the same one-sided
Wilcoxon gate. What differs is the authority under test. The side is decided
elsewhere and is not touched here; this only asks how big the position should
be once there is one.

The reference is the unit-size arm, which is what the backtest does today.

Run:  python train_sizing.py [--assets A,B] [--budget 300]
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np

import train_timing as tt
from core import sizing_policy as sp
from core import timing_policy as tp
from core.ar_rl import CmaEmitter
from core.backtesting import evaluate_signals_v2, score_strategy
from core.sizing_policy import match_exposure

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sizing_report.json")
MIN_EFFECT = 0.5


def _sides(series, timing):
    """The positions the incumbent stack takes, which sizing then scales.

    Cached on the series: the sides do not depend on the sizing, and an ES
    budget of 300 over 200 assets would otherwise replay the same 3700-bar
    python loop a hundred thousand times to get the same answer.
    """
    cached = series.get("_sides")
    # Length-checked, not just present: split_series copies unknown keys into
    # every slice, so a cache built on the full history would otherwise be
    # handed to a slice it does not describe. A sides array cannot be sliced
    # either, because the policy carries state across bars and a slice starts
    # from a different position.
    if cached is not None and len(cached) == len(series["probs"]):
        return cached
    policy = timing or tp.load_policy() or tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    sides, _a, _r = policy.apply(
        series["probs"], series["buy_thr"], series["sell_thr"],
        series["atr"], series["taleb_hi"], series["risky"],
        next_ret=series["next_ret"])
    series["_sides"] = sides
    return sides


def eval_sizing(series, sizing, timing=None):
    """Score one asset with `sizing` applied on top of the timing decision.

    The sizes are exposure-matched to the unit-size arm first, so a candidate
    is judged on the shape of its sizing and never on holding more.
    """
    comm, slip = tt._costs(series)
    sides = _sides(series, timing)
    sizes = (None if sizing is None
             else match_exposure(sizing.sizes_for(series), sides))
    profit, n_trades, win_rate, max_dd, sharpe = evaluate_signals_v2(
        sides, series["next_ret"], comm, slip, sizes=sizes)
    return {"score": score_strategy(profit, max_dd, win_rate, n_trades,
                                    sharpe, min_trades=5),
            "profit": profit, "n_trades": n_trades}


class _P:
    """Attribute carrier so CmaEmitter can read and write the params."""

    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)


def _params_of(obj):
    return {name: getattr(obj, name) for name, _, _, _ in sp.PARAM_SPECS}


def fit_sizing(train_by_asset, val_by_asset, budget=300, seed=42, timing=None):
    """Separable-ES fit, driven by TRAIN and selected on VAL."""
    import random as _random
    rng = _random.Random(seed)
    es = CmaEmitter(rng=rng, dims=sp.PARAM_SPECS)
    es.seed_from(_P(dict(sp.DEFAULT_PARAMS)))
    best_params, best_val = dict(sp.DEFAULT_PARAMS), float("-inf")
    for it in range(budget):
        cand = es.ask(_P(dict(sp.DEFAULT_PARAMS)))
        params = _params_of(cand)
        pol = sp.SizingPolicy(params)
        train_fit = tt.fitness([eval_sizing(s, pol, timing)["score"]
                                for s in train_by_asset.values()])
        es.tell(es.vector_of(cand), train_fit)
        val_fit = tt.fitness([eval_sizing(s, pol, timing)["score"]
                              for s in val_by_asset.values()])
        if it % 20 == 0 or it == budget - 1:
            print("[sizing]   ES %d/%d  train=%+.2f  val=%+.2f  best=%+.2f"
                  % (it + 1, budget, train_fit, val_fit,
                     max(best_val, val_fit)), flush=True)
        if val_fit > best_val:
            best_val, best_params = val_fit, params
    return best_params


def gate_sizing(test_by_asset, params, timing=None):
    """Sizing-vs-unit-size on TEST: one-sided Wilcoxon over per-asset deltas."""
    from scipy.stats import wilcoxon
    pol = sp.SizingPolicy(params)
    per_asset, deltas = {}, []
    for asset, s in test_by_asset.items():
        d = (eval_sizing(s, pol, timing)["score"]
             - eval_sizing(s, None, timing)["score"])
        per_asset[asset] = round(d, 4)
        deltas.append(d)
    n = len(deltas)
    if n >= 8 and any(abs(d) > 1e-12 for d in deltas):
        try:
            p = float(wilcoxon(deltas, alternative="greater").pvalue)
        except ValueError:
            p = 1.0
    else:
        p = 1.0
    mean_d = float(np.mean(deltas)) if deltas else 0.0
    verdict = ("ADOPT" if (n >= 8 and p < 0.05 and mean_d > MIN_EFFECT)
               else "HOLD")
    return {"verdict": verdict, "p": p, "mean_d": mean_d, "n": n,
            "per_asset": per_asset}


def main():
    ap = argparse.ArgumentParser(description="fit and gate the sizing rule")
    ap.add_argument("--assets", default="")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import config
    assets = ([a.strip() for a in args.assets.split(",") if a.strip()]
              or [a for grp in config.ASSET_TYPES.values() for a in grp])
    series = {}
    for i, a in enumerate(assets, 1):
        s = tt.build_asset_series(a)
        if s is not None:
            series[a] = s
        print("[sizing] [%d/%d] %s: %s" % (
            i, len(assets), a, "%d bars" % len(s["probs"]) if s else "skipped"),
            flush=True)
    splits = {a: tt.split_series(s) for a, s in series.items()}
    train = {a: v[0] for a, v in splits.items()}
    val = {a: v[1] for a, v in splits.items()}
    test = {a: v[2] for a, v in splits.items()}
    params = fit_sizing(train, val, budget=args.budget, seed=args.seed)
    gate = gate_sizing(test, params)
    report = {"params": params, "verdict": gate["verdict"], "p": gate["p"],
              "mean_d": gate["mean_d"], "n": gate["n"],
              "per_asset": gate["per_asset"],
              "fitted": datetime.utcnow().isoformat()}
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("[sizing] params: %s" % params)
    print("[sizing] VERDICT: %s  p=%.4f  mean_d=%+.3f  n=%d"
          % (gate["verdict"], gate["p"], gate["mean_d"], gate["n"]))


if __name__ == "__main__":
    main()
