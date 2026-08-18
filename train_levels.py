"""Fit and gate the trade-levels policy.

The same offline shape as train_timing.py, deliberately: the same reconstructed
per-asset history, the same separable ES from core.ar_rl, the same time split,
the same one-sided Wilcoxon gate. Only the ENVIRONMENT differs. A timing policy
decides when to act on a side; a levels policy decides WHERE the entry zone and
the stop sit once a side exists, so its episode is the trade that
core.levels.resolve_trade defines, and its reward is that trade's realised
return net of both legs.

Two things this fixes about `K_ENTRY = 0.5` and `K_STOP = 2.0` in core/levels.py:
they were never fitted, and they were never measured.

The adopted TIMING policy is held fixed while these are fitted. It has already
passed its own gate, fitting both at once doubles the search space, and a joint
move would leave no way to say which half earned the result.

What is NOT fitted here: the trailing multiplier. The journal stores one stop
per issued day, already trailed as of that day, and scores the trade against it.
Trailing therefore lives BETWEEN daily issues, not inside one trade, so this
environment cannot identify it. Fitting it needs a different episode definition
and is its own piece of work.

Usage:
    python train_levels.py                       # fit + gate + report
    python train_levels.py --assets SP500,NVDA   # subset
    python train_levels.py --budget 400          # ES evaluations
"""
import argparse
import json
import os

import numpy as np

from core import levels as levels_mod
from core import timing_policy as tp
from core.ar_rl import CmaEmitter
from core.backtesting import COMMISSION, FOREX_COMMISSION, FOREX_SLIPPAGE, SLIPPAGE
from train_timing import build_asset_series, fitness, split_series

BASE = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(BASE, "levels_policy.json")

# (name, low, high, is_int), the shape CmaEmitter reads. Everything is a
# multiple of one ATR: a base pair, then four regime deltas that ride on top.
PARAM_SPECS = (
    ("k_entry", 0.10, 1.50, False),
    ("k_stop", 0.50, 6.00, False),
    # Regime deltas, added on top of the base and free to be negative: whether a
    # fat-tail bar wants a wider stop or a tighter one is the question, not the
    # assumption. Zero reproduces the flat policy exactly, which is what makes
    # the flat policy this one's honest baseline.
    ("d_entry_hi_taleb", -0.40, 0.80, False),
    ("d_entry_risky", -0.40, 0.80, False),
    ("d_stop_hi_taleb", -1.50, 3.00, False),
    ("d_stop_risky", -1.50, 3.00, False),
)

DEFAULT_PARAMS = dict(levels_mod.POLICY_DEFAULTS)


class _P:
    """Attribute carrier, the same one train_timing uses: CmaEmitter reads and
    writes parameters through getattr/setattr per PARAM_SPECS."""

    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)

# How far ahead one issued level is allowed to run before it is dropped from the
# sample. A bound is needed or a signal that never flips walks the whole history
# on every one of a few hundred evaluations; 60 bars is the same window the live
# card and the sheet already load.
MAX_HORIZON = 60

# The practical-effect floor for adoption, in the units this gate MEASURES:
# mean net return per issued signal. Five basis points. Copying the timing
# gate's 0.5 would have been the mistake of 2026-08-18 all over again, where a
# floor in Score units was checked against a value in AUC units and passed
# everything.
ADOPT_FLOOR = 0.0005


def _costs(series):
    if series.get("is_forex"):
        return FOREX_COMMISSION, FOREX_SLIPPAGE
    return COMMISSION, SLIPPAGE


def sides_for(series):
    """The sides a person is actually acting on: the ADOPTED timing policy.

    Not the raw thresholded signal. Levels are drawn on what the card shows,
    and the card shows the timing policy's decision, so fitting against the raw
    signal would fit a screen nobody sees.
    """
    # load_policy already returns a RulesPolicy, not a parameter dict.
    policy = tp.load_policy() or tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    sides, _actions, _reasons = policy.apply(
        series["probs"], series["buy_thr"], series["sell_thr"],
        series["atr"], series["taleb_hi"], series["risky"],
        next_ret=series["next_ret"])
    return np.asarray(sides, dtype=int)


def eval_levels(series, params, sides=None):
    """Replay one asset under one levels policy.

    Levels are issued on EVERY bar that carries a side, exactly as predict.py
    issues them every day, and each issue is resolved on its own. That mirrors
    level_log row for row, so the number fitted here is the number the live
    journal will later report.

    Returns {"mean_ret", "n", "wins"}; mean_ret is 0.0 with n = 0 when the
    policy never got a resolvable trade.
    """
    comm, slip = _costs(series)
    leg_cost = comm + slip
    if sides is None:
        sides = sides_for(series)
    o, h, lo_, c = (series["open"], series["high"], series["low"], series["close"])
    atr = series["atr"]
    taleb = series.get("taleb_hi")
    risky = bool(series.get("risky"))
    rets = []
    for i in range(len(sides) - 1):
        side = int(sides[i])
        if side == 0:
            continue
        a, close = atr[i], c[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(close) or close <= 0:
            continue
        k_entry, k_stop = levels_mod.effective_multipliers(
            params, taleb_hi=bool(taleb[i]) if taleb is not None else False,
            risky=risky)
        j = min(i + 1 + MAX_HORIZON, len(sides))
        bars = list(zip(o[i + 1:j], h[i + 1:j], lo_[i + 1:j], c[i + 1:j]))
        out = levels_mod.resolve_trade(
            side, close - k_entry * a, close + k_entry * a,
            close - side * k_stop * a, bars, sides[i + 1:j], leg_cost)
        if out is not None:
            rets.append(out["ret_net"])
    if not rets:
        return {"mean_ret": 0.0, "n": 0, "wins": 0}
    return {"mean_ret": float(np.mean(rets)), "n": len(rets),
            "wins": int(sum(1 for r in rets if r > 0))}


def baseline_params():
    """What a candidate has to beat: the policy that is LIVE, not the constants.

    A regime-conditioned fit measured against the shipped constants would take
    credit for whatever the flat fit already earned. Same reasoning as the A/B
    reference being the adopted genome rather than production defaults.
    """
    return levels_mod.load_policy() or dict(DEFAULT_PARAMS)


def eval_baseline(series, sides=None, params=None):
    """The levels production issues today."""
    return eval_levels(series, params or baseline_params(), sides=sides)


def _params_of(obj):
    return {name: getattr(obj, name) for name, _lo, _hi, _i in PARAM_SPECS}


def fit_policy(train_by_asset, budget=300, seed=42, val_by_asset=None):
    """Separable ES over the multipliers, the same loop train_timing runs.

    Scored on TRAIN, SELECTED on VAL: the returned parameters are the best
    vector seen on the validation split, not the ES's own training peak. The
    per-asset scores reduce through fitness(), median minus a quarter of the
    interquartile range, which is what stops one asset carrying a result no
    other asset shares.
    """
    import random as _random

    rng = _random.Random(seed)
    es = CmaEmitter(rng=rng, dims=PARAM_SPECS)
    es.seed_from(_P(dict(DEFAULT_PARAMS)))
    val_by_asset = val_by_asset or train_by_asset
    # The sides are the frozen timing policy's, so they do not change with the
    # levels parameters and are computed once instead of per evaluation.
    tr_sides = {a: sides_for(s) for a, s in train_by_asset.items()}
    va_sides = {a: sides_for(s) for a, s in val_by_asset.items()}

    best_params, best_val = dict(DEFAULT_PARAMS), float("-inf")
    for it in range(budget):
        cand = es.ask(_P(dict(DEFAULT_PARAMS)))
        params = _params_of(cand)
        train_fit = fitness([eval_levels(s, params, sides=tr_sides[a])["mean_ret"]
                             for a, s in train_by_asset.items()])
        es.tell(es.vector_of(cand), train_fit)
        val_fit = fitness([eval_levels(s, params, sides=va_sides[a])["mean_ret"]
                           for a, s in val_by_asset.items()])
        if it % 10 == 0 or it == budget - 1:
            print("[levels]   ES %d/%d  train=%+.5f  val=%+.5f  best_val=%+.5f"
                  % (it + 1, budget, train_fit, val_fit, max(best_val, val_fit)),
                  flush=True)
        if val_fit > best_val:
            best_val, best_params = val_fit, params
    return best_params


def gate_policy(test_by_asset, params):
    """Policy against today's constants on TEST: one-sided Wilcoxon over
    per-asset deltas in mean net return per issued signal."""
    from scipy.stats import wilcoxon

    base = baseline_params()
    per_asset, deltas = {}, []
    for asset, s in test_by_asset.items():
        sides = sides_for(s)
        d = (eval_levels(s, params, sides=sides)["mean_ret"]
             - eval_baseline(s, sides=sides, params=base)["mean_ret"])
        per_asset[asset] = round(d, 6)
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
    verdict = ("ADOPT" if (n >= 8 and p < 0.05 and mean_d > ADOPT_FLOOR)
               else "HOLD")
    return {"verdict": verdict, "p": p, "mean_d": mean_d, "n": n,
            "floor": ADOPT_FLOOR, "per_asset": per_asset}


def save_policy(params, gate, path=None):
    """Write the policy only when the gate says ADOPT, with its evidence."""
    if gate["verdict"] != "ADOPT":
        return None
    body = {"params": {n: params.get(n, DEFAULT_PARAMS[n])
                       for n, _l, _h, _i in PARAM_SPECS},
            "gate": gate, "baseline": baseline_params()}
    with open(path or POLICY_PATH, "w", encoding="utf-8") as fh:
        json.dump(body, fh, ensure_ascii=False, indent=2)
    return path or POLICY_PATH


def main():
    ap = argparse.ArgumentParser(description="fit and gate the levels policy")
    ap.add_argument("--assets", default="")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from config import FULL_ASSET_MAP
    names = ([a.strip().upper() for a in args.assets.split(",") if a.strip()]
             or list(FULL_ASSET_MAP))
    train, val, test = {}, {}, {}
    for asset in names:
        series = build_asset_series(asset)
        if series is None:
            continue
        tr, va, te = split_series(series)
        train[asset], val[asset], test[asset] = tr, va, te
    if len(train) < 8:
        print("Only %d assets have a champion-scorable history; the gate needs "
              "8. Nothing fitted." % len(train))
        return 1

    print("Fitting on %d assets, budget %d." % (len(train), args.budget))
    params = fit_policy(train, budget=args.budget, seed=args.seed,
                        val_by_asset=val)
    gate = gate_policy(test, params)
    print("\n  baseline   k_entry %.3f  k_stop %.3f"
          % (DEFAULT_PARAMS["k_entry"], DEFAULT_PARAMS["k_stop"]))
    print("  fitted     k_entry %.3f  k_stop %.3f"
          % (params["k_entry"], params["k_stop"]))
    print("  gate       %s  mean_d %+.5f (floor %+.5f)  p %.4f  n %d"
          % (gate["verdict"], gate["mean_d"], gate["floor"], gate["p"], gate["n"]))
    written = save_policy(params, gate)
    print("  %s" % ("wrote %s" % os.path.basename(written) if written
                    else "HOLD: nothing written, production keeps its constants"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
