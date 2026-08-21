"""Fit and gate the Stage-A timing policy.

Offline pipeline: reconstruct per-asset historical CatBoost-champion
probabilities (the What-If pattern), replay the rules policy on the
position-persistent simulator (core.backtesting.simulate_positions - the
Stage-0 timing environment), fit the 8 rule parameters with a separable
ES on a global time split, and gate policy-vs-baseline with a
one-sided Wilcoxon before writing timing_policy.json.

Usage:
    python train_timing.py                # fit + gate + report
    python train_timing.py --assets SP500,NVDA,BTC   # subset
    python train_timing.py --budget 400   # ES evaluations
"""
import argparse
import json
import os

import numpy as np

from core import timing_policy as tp
from core.ar_rl import CmaEmitter
from core.backtesting import (
    COMMISSION,
    FOREX_COMMISSION,
    FOREX_SLIPPAGE,
    SLIPPAGE,
    UNRELIABLE_SCORE,
    evaluate_signals_v2,
    score_strategy,
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
DB_PATH = os.path.join(BASE, "market.db")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")

MIN_PROB_ROWS = 300


def _costs(series):
    """(commission, slippage) for one asset's series - forex legs are cheaper."""
    if series.get("is_forex"):
        return FOREX_COMMISSION, FOREX_SLIPPAGE
    return COMMISSION, SLIPPAGE


def _run_sides(series, sides):
    comm, slip = _costs(series)
    profit, n_trades, win_rate, max_dd, sharpe = evaluate_signals_v2(
        sides, series["next_ret"], comm, slip)
    score = score_strategy(profit, max_dd, win_rate, n_trades, sharpe,
                           min_trades=5)
    return {"score": score, "profit": profit, "sharpe": sharpe,
            "n_trades": n_trades, "win_rate": win_rate, "max_dd": max_dd}


def eval_policy(series, policy):
    """Run `policy` over one asset's reconstructed series.

    Returns a dict with keys score, profit, sharpe, n_trades, win_rate
    (plus max_dd) from the position-aware v2 objective.

    A Stage-B policy needs the whole series (its state carries the regime and
    the volatility percentile, which .apply's argument list does not pass), so
    a policy that offers apply_series is given the series.
    """
    if hasattr(policy, "apply_series"):
        sides, _actions, _reasons = policy.apply_series(series)
    else:
        sides, _actions, _reasons = policy.apply(
            series["probs"], series["buy_thr"], series["sell_thr"],
            series["atr"], series["taleb_hi"], series["risky"],
            next_ret=series["next_ret"])
    return _run_sides(series, sides)


def eval_baseline(series):
    """Same as eval_policy but sides = the raw thresholded signal per bar
    (DEFAULT_PARAMS reproduce this identity), i.e. today's production
    baseline behavior. One known divergence: on a direct one-bar flip
    (prob jumps across the whole neutral band) production reverses the
    position the same bar, while the policy exits flat and re-enters next
    bar; flips are rare and the effect is common-mode across both arms of
    the gate, so the ADOPT/HOLD delta is unaffected."""
    return eval_policy(series, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)))


def fitness(per_asset_scores):
    """median - 0.25 * IQR over per-asset scores; -inf for an empty list."""
    if not per_asset_scores:
        return float("-inf")
    a = np.asarray(per_asset_scores, dtype=float)
    iqr = np.percentile(a, 75) - np.percentile(a, 25)
    return float(np.median(a) - 0.25 * iqr)


def build_asset_series(asset, engine=None):
    """Reconstruct one asset's champion-scorable history for policy fitting.

    Mirrors whatif_simulator._predict_cb's loading pattern (registry lookup,
    champion model path, feature engineering, scaler parity via load_or_fit
    pattern matching production serve) but over the FULL price history instead
    of a `days_back` slice, so the returned arrays cover every bar the
    champion can be scored on.

    Returns a dict with keys probs, next_ret, atr, taleb_hi, buy_thr,
    sell_thr, risky, is_forex, dates (numpy arrays / scalars, oldest-first),
    or None when the asset has no CatBoost champion or too little
    champion-scorable history (< 300 rows).
    """
    import pandas as pd
    from catboost import CatBoostClassifier

    import config
    from core.features import build_features, compute_taleb_risk
    from core.scaling import load_or_fit_scaler
    from core.track_record import _table_name
    from whatif_simulator import _load_registry

    if engine is None:
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{DB_PATH}")

    table = _table_name(asset)
    model_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
    if not os.path.exists(model_path):
        return None

    try:
        df_raw = pd.read_sql(
            f'SELECT * FROM "{table}"', engine,
            index_col="Date", parse_dates=["Date"])
    except Exception as e:
        print(f"[timing] skip {asset}: db read failed ({e})")
        return None
    if df_raw.empty:
        return None
    df_raw = df_raw[~df_raw.index.duplicated(keep="last")].sort_index()

    try:
        # Full production feature chain (mirrors predict.py lines 93-97) so
        # champions trained with cross-asset / macro / lead-lag features find
        # every column they expect in the pool.
        df_feat, _ = build_features(df_raw.copy(), table, engine)
    except Exception as e:
        print(f"[timing] skip {asset}: feature engineering failed ({e})")
        return None

    if "Date" in df_feat.columns:
        df_feat = df_feat.set_index("Date")
    elif "date" in df_feat.columns:
        df_feat = df_feat.set_index("date")
    df_feat.index = pd.to_datetime(df_feat.index)

    registry = _load_registry()
    feat_list = None
    if asset in registry:
        feat_list = registry[asset].get("features")
    if not feat_list:
        drop_cols = {"target", "next_ret", "close", "open", "high", "low", "volume"}
        feat_list = [c for c in df_feat.columns
                     if c not in drop_cols and pd.api.types.is_numeric_dtype(df_feat[c])]
    lost = [f for f in feat_list if f not in df_feat.columns]
    if lost:
        print(f"[timing] skip {asset}: features missing from pool {lost}")
        return None

    df_feat = df_feat.dropna(subset=feat_list)
    n = len(df_feat)
    split = int(n * 0.8)
    if split <= 0 or split >= n:
        return None

    X = df_feat[feat_list].values
    # Scaler parity with production serve (core.scoring): prefer the SAVED
    # train-fold scaler. With it, the champion can be scored over the WHOLE
    # engineered history without leakage (the scaler saw only its own train
    # fold). Legacy models without a saved scaler fall back to an ad-hoc fit
    # on the first 80% and are scored only on the remaining 20%.
    scaler, _src = load_or_fit_scaler(MODEL_DIR, table, X[:split])
    if _src == "saved":
        split = 0
    X_pred_scaled = scaler.transform(X[split:])

    try:
        cb = CatBoostClassifier()
        cb.load_model(model_path)
        probs = cb.predict_proba(X_pred_scaled)[:, 1]
    except Exception as e:
        print(f"[timing] skip {asset}: prediction failed ({e})")
        return None

    if len(probs) < MIN_PROB_ROWS:
        return None

    # Taleb risk needs the earlier history to warm up its rolling window, so
    # compute it on the full close series and slice afterward (same alignment
    # as atr/close below).
    taleb_full = compute_taleb_risk(df_feat["close"])
    close_slice = df_feat["close"].iloc[split:]
    next_ret = close_slice.pct_change().shift(-1).to_numpy(dtype=float)
    atr = df_feat["atr"].iloc[split:].to_numpy(dtype=float)
    taleb_hi = (taleb_full.iloc[split:] > 0.7).fillna(False).to_numpy()
    dates = df_feat.index[split:].to_numpy()

    thresholds = {}
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, encoding="utf-8") as fh:
            thresholds = json.load(fh)
    thr = thresholds.get(asset, {})
    buy_thr = thr.get("buy", 0.55)
    sell_thr = thr.get("sell", 0.45)

    forex_groups = ("FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC")
    is_forex = any(asset in config.ASSET_TYPES.get(g, []) for g in forex_groups)
    risky = is_forex or asset in config.ASSET_TYPES.get("CRYPTO", [])

    # high/low/close ride along for the LEVELS policy (train_levels.py), which
    # needs to know whether a bar traded into an entry zone or through a stop.
    # Same slice as atr/next_ret above, so every per-bar array stays aligned.
    return {
        "probs": probs, "next_ret": next_ret, "atr": atr, "taleb_hi": taleb_hi,
        "high": df_feat["high"].iloc[split:].to_numpy(dtype=float),
        "low": df_feat["low"].iloc[split:].to_numpy(dtype=float),
        "close": close_slice.to_numpy(dtype=float),
        "open": df_feat["open"].iloc[split:].to_numpy(dtype=float),
        "buy_thr": buy_thr, "sell_thr": sell_thr, "risky": risky,
        "is_forex": is_forex, "dates": dates,
    }


_SLICED = ("probs", "next_ret", "atr", "taleb_hi", "dates",
           "high", "low", "close", "open")


def split_series(series, train_frac=0.6, val_frac=0.2):
    """Time-ordered (train, val, test) slices of `series`.

    Only the per-bar arrays in `_SLICED` are cut; scalar fields (buy_thr,
    sell_thr, risky, is_forex) are copied unchanged into every slice.
    """
    n = len(series["probs"])
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    out = []
    for lo, hi in ((0, a), (a, b), (b, n)):
        part = dict(series)
        for k in _SLICED:
            # Only what this series actually carries. A timing series has no
            # OHLC and does not need any: those arrays exist for the levels
            # policy, and demanding them here would make every caller build
            # columns it has no use for.
            if k in series:
                part[k] = series[k][lo:hi]
        out.append(part)
    return tuple(out)


class _P:
    """Attribute carrier so CmaEmitter.ask/vector_of/seed_from can read and
    write the 8 timing-policy params via getattr/setattr per PARAM_SPECS."""

    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)


def _params_of(obj):
    return {name: getattr(obj, name) for name, _, _, _ in tp.PARAM_SPECS}


def fit_policy(train_by_asset, budget=300, seed=42, val_by_asset=None):
    """Separable-ES fit of the 8 timing params over `train_by_asset`.

    Each candidate is scored on TRAIN (drives the ES) and on VAL (drives
    model selection, i.e. the returned params are the best-on-VAL vector
    seen across the whole budget, not just the ES's final mean).
    """
    import random as _random
    rng = _random.Random(seed)
    es = CmaEmitter(rng=rng, dims=tp.PARAM_SPECS)
    es.seed_from(_P(dict(tp.DEFAULT_PARAMS)))
    best_params, best_val = dict(tp.DEFAULT_PARAMS), float("-inf")
    val_by_asset = val_by_asset or train_by_asset
    for it in range(budget):
        cand = es.ask(_P(dict(tp.DEFAULT_PARAMS)))
        params = _params_of(cand)
        pol = tp.RulesPolicy(params)
        train_fit = fitness([eval_policy(s, pol)["score"]
                             for s in train_by_asset.values()])
        es.tell(es.vector_of(cand), train_fit)
        val_fit = fitness([eval_policy(s, pol)["score"]
                           for s in val_by_asset.values()])
        if it % 10 == 0 or it == budget - 1:
            print(f"[timing]   ES {it + 1}/{budget}  train={train_fit:+.2f}  "
                  f"val={val_fit:+.2f}  best_val={max(best_val, val_fit):+.2f}",
                  flush=True)
        if val_fit > best_val:
            best_val, best_params = val_fit, params
    return best_params


def gate_policy(test_by_asset, params, reference=None):
    """Policy-vs-reference verdict on TEST: one-sided Wilcoxon over per-asset
    score deltas. ADOPT requires n >= 8 assets, p < 0.05, and mean_d > 0.5.

    `params` is a parameter dict for Stage A or a policy object for Stage B.
    `reference` defaults to the production baseline, which is what Stage A is
    measured against; Stage B passes the ADOPTED Stage-A policy, because
    beating the baseline again would prove nothing about replacing the rules.
    """
    from scipy.stats import wilcoxon
    # Any policy object passes through; only a parameter dict is wrapped.
    pol = (params if hasattr(params, "apply_series") or hasattr(params, "apply")
           else tp.RulesPolicy(params))
    per_asset, deltas, unscorable = {}, [], []
    for asset, s in test_by_asset.items():
        base = (eval_baseline(s)["score"] if reference is None
                else eval_policy(s, reference)["score"])
        cand = eval_policy(s, pol)["score"]
        # UNRELIABLE_SCORE is not a score, it is "this arm made too few trades
        # to judge". Subtracting it produces a delta near -999 that the rank
        # test cannot see and the mean cannot survive: one short asset out of
        # twenty moved mean_d from +18.5 to -29.4 and flipped ADOPT to HOLD
        # while p stayed at 0.0002. Drop the pair and say how many were dropped.
        if UNRELIABLE_SCORE in (base, cand):
            unscorable.append(asset)
            continue
        d = cand - base
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
    verdict = "ADOPT" if (n >= 8 and p < 0.05 and mean_d > 0.5) else "HOLD"
    return {"verdict": verdict, "p": p, "mean_d": mean_d, "n": n,
            "n_unscorable": len(unscorable), "unscorable": unscorable,
            "per_asset": per_asset}


GATE_MIN_ASSETS = 8


def require_scorable(series, what):
    """True when enough assets were rebuilt to gate anything at all.

    Without this the fitters answer a question nobody can have asked: an ES
    over an empty set reports fitness -inf and the gate then prints
    "verdict: HOLD  p=1.0000  mean_d=+0.00  n=0", which reads like a
    measurement and is not one. Stage B did not even get that far - it raised
    ValueError out of np.concatenate on an empty batch list. train_levels.py
    already refused this case in so many words; this is the same refusal for
    the other three entry points.
    """
    if len(series) >= GATE_MIN_ASSETS:
        return True
    print("[%s] only %d asset(s) have a champion-scorable history; the gate "
          "needs %d. Nothing fitted." % (what, len(series), GATE_MIN_ASSETS))
    print("[%s] assets with no champion are skipped silently - list them with "
          "`python model_health.py --missing` and train those first." % what)
    return False


def save_policy(params, gate, path=None):
    """Always write timing_report.json next to `path`; write the adopted
    timing_policy.json itself only when gate["verdict"] == "ADOPT"."""
    from datetime import datetime
    path = path or tp.POLICY_PATH
    report = {"verdict": gate["verdict"], "per_asset": gate.get("per_asset"),
              "p": gate.get("p"), "mean_d": gate.get("mean_d"),
              "params": params, "fitted": datetime.utcnow().isoformat()}
    with open(os.path.join(os.path.dirname(path), "timing_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    if gate["verdict"] == "ADOPT":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "params": params,
                       "fitted": report["fitted"]}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", default="a", choices=("a", "b"),
                    help="a = the interpretable rules fit (default), "
                         "b = the fitted-Q challenger to the adopted rules")
    ap.add_argument("--iters", type=int, default=6,
                    help="stage b: fitted-Q iterations, one horizon rung each")
    ap.add_argument("--gamma", type=float, default=0.97,
                    help="stage b: discount per bar")
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="stage b: exploration rate when logging transitions")
    args = ap.parse_args()
    import config
    assets = ([a.strip() for a in args.assets.split(",") if a.strip()]
              or [a for grp in config.ASSET_TYPES.values() for a in grp])
    series = {}
    for i, a in enumerate(assets, 1):
        s = build_asset_series(a)
        if s is not None:
            series[a] = s
            print(f"[timing] [{i}/{len(assets)}] {a}: {len(s['probs'])} bars",
                  flush=True)
        else:
            print(f"[timing] [{i}/{len(assets)}] {a}: skipped", flush=True)
    print(f"[timing] {len(series)}/{len(assets)} assets with usable history")
    if not require_scorable(series, "timing"):
        return 1
    if args.stage == "b":
        out = stage_b(series, iters=args.iters, gamma=args.gamma,
                      epsilon=args.epsilon, seed=args.seed)
        print("[timing-b] reference stage_a | assets %d | selected on val: %s"
              % (out["assets"], out["selected_on_val"]))
        for r in out["rows"]:
            print("[timing-b]   %-12s mean_d %+.3f  p %.4f  p_bh %.4f  "
                  "n %d%s  %s"
                  % (r["name"], r["mean_d"], r["p"], r["p_bh"], r["n"],
                     ("  -%d unscorable" % r["n_unscorable"]
                      if r.get("n_unscorable") else ""),
                     "ADOPT" if r["adopt"] else "-"))
        pr = out["proxy"]
        print("[timing-b] objective vs gate: rho %+.3f (p %.4f), sign "
              "agreement %d/%d. A verdict that arrives with a flat or negative "
              "correlation is a coincidence until replicated."
              % (pr["rho"], pr["p"], pr["sign_agree"], pr["n"]))
        print("[timing-b] VERDICT: %s" % out["verdict"])
        save_stage_b(out)
        return

    tr = {a: split_series(s)[0] for a, s in series.items()}
    va = {a: split_series(s)[1] for a, s in series.items()}
    te = {a: split_series(s)[2] for a, s in series.items()}
    params = fit_policy(tr, budget=args.budget, seed=args.seed,
                        val_by_asset=va)
    gate = gate_policy(te, params)
    save_policy(params, gate)
    print(f"[timing] params: {params}")
    print(f"[timing] verdict: {gate['verdict']}  p={gate['p']:.4f}  "
          f"mean_d={gate['mean_d']:+.2f}  n={gate['n']}")
    if gate["n_unscorable"]:
        print("[timing] %d asset(s) dropped as unscorable (an arm traded fewer "
              "than the minimum): %s"
              % (gate["n_unscorable"], ", ".join(gate["unscorable"][:8])))
    if gate["verdict"] == "ADOPT":
        print("[timing] wrote timing_policy.json - set GTRADE_TIMING_POLICY=1 "
              "to run in shadow mode.")



MIN_EFFECT = 0.5          # Score units, the same bar Stage A cleared


def bh_rows(rows, alpha=0.05):
    """Benjamini-Hochberg adjusted p-values, one per candidate.

    Reuses auto_research's implementation rather than a second copy: six
    iteration counts are six chances to look good once, and the correction for
    that is a solved problem this repository already solved.
    """
    from auto_research import benjamini_hochberg
    flags = benjamini_hochberg([r["p"] for r in rows], alpha=alpha)
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i]["p"])
    out = [dict(r) for r in rows]
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, out[i]["p"] * m / rank)
        out[i]["p_bh"] = running
        out[i]["bh_flag"] = bool(flags[i])
    return out


def gate_challenger(test_by_asset, candidates, reference):
    """Gate every candidate Q against the incumbent, corrected together.

    candidates is [(name, policy)]. A candidate adopts only with a BH-adjusted
    p below alpha AND a mean delta above MIN_EFFECT. Lose or draw and the
    rules stay, which is the spec's rule and also the safe direction: Stage A
    is interpretable and already live.
    """
    rows = []
    for name, pol in candidates:
        g = gate_policy(test_by_asset, pol, reference=reference)
        rows.append({"name": name, "p": g["p"], "mean_d": g["mean_d"],
                     "n": g["n"], "n_unscorable": g["n_unscorable"],
                     "per_asset": g["per_asset"]})
    rows = bh_rows(rows)
    for r in rows:
        r["adopt"] = bool(r["bh_flag"] and r["mean_d"] > MIN_EFFECT
                          and r["n"] >= 8)
    winners = [r for r in rows if r["adopt"]]
    best = max(winners, key=lambda r: r["mean_d"], default=None)
    return {"rows": rows, "best": best,
            "verdict": "ADOPT" if best else "HOLD"}


def objective_vs_gate(test_by_asset, policy, reference):
    """Does what the fit maximises move with what the gate decides on?

    The FQI maximises discounted net return per bar; the gate reads
    score_strategy, which also carries drawdown, win rate and Sharpe. On
    2026-08-18 a campaign that optimised one quantity and adopted on another
    was found to have a rank correlation of -0.24 between them, and every
    result it produced said nothing about production. So this number is printed
    next to the verdict, always, and a verdict that arrives with a flat or
    negative correlation is a coincidence until it is replicated.
    """
    from scipy.stats import spearmanr
    d_obj, d_gate = [], []
    for s in test_by_asset.values():
        a = eval_policy(s, policy)
        b = eval_policy(s, reference)
        d_obj.append(a["profit"] - b["profit"])
        d_gate.append(a["score"] - b["score"])
    if len(d_obj) < 3:
        return {"rho": 0.0, "p": 1.0, "sign_agree": 0, "n": len(d_obj)}
    r = spearmanr(d_obj, d_gate)
    agree = sum(1 for x, y in zip(d_obj, d_gate) if x * y > 0)
    return {"rho": float(r.statistic), "p": float(r.pvalue),
            "sign_agree": agree, "n": len(d_obj)}


def stage_b(by_asset, iters=6, gamma=0.97, epsilon=0.1, seed=0,
            challenger_factory=None, reference=None):
    """Fit a Q challenger on TRAIN, pick the horizon on VAL, gate on TEST.

    The incumbent is the ADOPTED Stage-A policy when timing_policy.json exists,
    and DEFAULT_PARAMS otherwise, which is the baseline. Either way the
    challenger has to beat what is actually running. `reference` overrides that
    lookup, which is what lets a test pin the incumbent instead of inheriting
    whatever happens to be adopted on this machine.
    """
    import random as _random

    from core import timing_fqi as fq

    incumbent = (reference or tp.load_policy()
                 or tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)))
    splits = {a: split_series(s) for a, s in by_asset.items()}
    train = {a: v[0] for a, v in splits.items()}
    val = {a: v[1] for a, v in splits.items()}
    test = {a: v[2] for a, v in splits.items()}

    rng = _random.Random(seed)
    batches = [fq.rollout(s, incumbent, rng, epsilon=epsilon,
                          costs=_costs(s)) for s in train.values()]
    models = fq.fit_q(batches, iters=iters, gamma=gamma, seed=seed)

    if challenger_factory is not None:
        candidates = challenger_factory(models)
    else:
        candidates = [("q_iter_%d" % (k + 1), fq.FqiPolicy(m))
                      for k, m in enumerate(models)]

    # VAL picks the horizon; TEST judges it. The same discipline Stage A runs
    # on, and the reason a candidate list is gated rather than a single fit.
    val_scores = [(name, fitness([eval_policy(s, pol)["score"]
                                  for s in val.values()]))
                  for name, pol in candidates]
    best_name = max(val_scores, key=lambda kv: kv[1])[0]
    gate = gate_challenger(test, candidates, reference=incumbent)
    chosen = dict(candidates)[best_name]
    proxy = objective_vs_gate(test, chosen, incumbent)
    return {"verdict": gate["verdict"], "rows": gate["rows"],
            "best": gate["best"], "val": dict(val_scores),
            "selected_on_val": best_name, "proxy": proxy,
            "reference": "stage_a", "assets": len(by_asset),
            # The fitted models ride along so saving does not mean fitting the
            # whole ladder a second time. Stripped before the report is written.
            "_models": models}


def save_stage_b(out, models=None, path=None):
    """Write the Stage-B report always; write the model only on ADOPT."""
    from datetime import datetime
    path = path or tp.POLICY_PATH
    here = os.path.dirname(path)
    report = {k: v for k, v in out.items() if k != "_models"}
    report["fitted"] = datetime.utcnow().isoformat()
    with open(os.path.join(here, "timing_fqi_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    models = models if models is not None else out.get("_models")
    if out["verdict"] == "ADOPT" and out.get("best") and models:
        idx = int(out["best"]["name"].rsplit("_", 1)[1]) - 1
        models[idx].save_model(os.path.join(here, "timing_fqi.cbm"))


if __name__ == "__main__":
    raise SystemExit(main())
