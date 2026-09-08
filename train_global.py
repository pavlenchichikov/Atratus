"""One model over every asset, measured against the same model fitted per asset.

The question this answers is NOT "is the pooled model better". It is whether
direction carries any information at all, which the per-asset setup cannot ask:
3760 rows put the standard error of a rank correlation at 0.016, so an IC of
0.02 reads t = 1.2 and disappears. The same 0.02 on a 3.2M-row panel reads t=36.

Both arms see the SAME rows, the same features and the same date folds. The only
difference is whether the fit is one model over the panel or one model per asset,
so the delta isolates pooling and nothing else.

    python train_global.py                    # both arms, full universe
    python train_global.py --assets AAPL,BTC  # a subset, for a quick look
    python train_global.py --no-baseline      # pooled arm only

Writes models/global_report.json. Trains nothing that serving reads.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import stats as sps
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine

import config
from core.features import CANDIDATE_FEATURES_EXT, build_features
from core.panel import POOL_MIN_BARS, build_panel, date_folds

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "market.db")
REPORT = os.path.join(BASE, "models", "global_report.json")

CB = {"iterations": 400, "depth": 6, "learning_rate": 0.05, "verbose": 0,
      "task_type": "CPU", "random_seed": 42, "thread_count": 12}

# Pre-registered, written down before the first run. The floor is the smaller of
# the two things worth having: a real edge over the per-asset fit on the same
# rows. Alpha is the project's usual 0.05, and the pairing is BY ASSET, which is
# where the power lives: hundreds of pairs, not five folds.
GATE_FLOOR = 0.005
GATE_ALPHA = 0.05
MIN_TEST_ROWS = 100


def _table(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def load_frames(engine, assets):
    """{asset: featured frame}. Anything that fails to build is simply absent."""
    out = {}
    for i, a in enumerate(assets, 1):
        t = _table(a)
        try:
            raw = pd.read_sql("SELECT * FROM %s" % t, engine,
                              index_col="Date", parse_dates=["Date"])
        except Exception:
            continue
        if len(raw) < POOL_MIN_BARS:
            continue
        try:
            df, _ = build_features(raw, t, engine)
        except Exception:
            continue
        if df is None or len(df) < POOL_MIN_BARS or "target" not in df.columns:
            continue
        # build_features hands back an integer index and keeps the date as a
        # COLUMN. Stacking on that index would align AAPL's row 7 with BTC's.
        if "Date" in df.columns:
            df = df.set_index(pd.to_datetime(df["Date"]).dt.normalize())
        out[a] = df
        if i % 100 == 0:
            print("  built %d/%d" % (i, len(assets)), flush=True)
    return out


def fit_predict(train_df, test_df, feats):
    cb = CatBoostClassifier(**CB)
    cb.fit(train_df[feats].values, train_df["target"].astype(int).values)
    return cb.predict_proba(test_df[feats].values)[:, 1]


def per_asset_auc(frame, prob_col):
    """AUC per asset on the test rows, skipping assets with one class only."""
    out = {}
    for a, g in frame.groupby("asset"):
        y = g["target"].astype(int).values
        if len(y) < 30 or len(set(y)) < 2:
            continue
        out[a] = float(roc_auc_score(y, g[prob_col].values))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", help="comma-separated subset")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--drop", default="",
                    help="comma-separated features to leave out of both arms")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the per-asset arm (then there is nothing to compare to)")
    args = ap.parse_args()

    engine = create_engine("sqlite:///%s" % DB)
    assets = ([a.strip().upper() for a in args.assets.split(",")] if args.assets
              else sorted(config.FULL_ASSET_MAP))
    print("building features for %d assets..." % len(assets), flush=True)
    t0 = time.time()
    frames = load_frames(engine, assets)
    print("  %d usable assets in %.0fs" % (len(frames), time.time() - t0), flush=True)
    if not frames:
        print("nothing to pool.")
        return 1

    dropped = {c.strip() for c in args.drop.split(",") if c.strip()}
    feats = [c for c in CANDIDATE_FEATURES_EXT
             if c not in dropped and any(c in df.columns for df in frames.values())]
    if dropped:
        print("  dropping %d feature(s): %s" % (len(dropped), sorted(dropped)))
    keep = feats + (["next_ret"] if any("next_ret" in df.columns
                                        for df in frames.values()) else [])
    panel = build_panel(frames, keep)
    feats = [c for c in feats if c in panel.columns]
    panel[feats] = panel[feats].astype("float32")
    panel = panel.dropna(subset=feats)
    print("panel: %d rows, %d assets, %d features, %s to %s"
          % (len(panel), panel["asset"].nunique(), len(feats),
             panel["date"].min().date(), panel["date"].max().date()), flush=True)

    folds = date_folds(panel["date"], n_folds=args.folds)
    if not folds:
        print("not enough dates for %d folds." % args.folds)
        return 1

    pooled = []
    for i, (tr_dates, te_dates) in enumerate(folds, 1):
        tr = panel[panel["date"].isin(tr_dates)]
        te = panel[panel["date"].isin(te_dates)].copy()
        if len(te) < MIN_TEST_ROWS or te["target"].nunique() < 2:
            continue
        t1 = time.time()
        te["p_pool"] = fit_predict(tr, te, feats)
        auc = roc_auc_score(te["target"].astype(int).values, te["p_pool"].values)
        print("  fold %d  train %7d  test %7d  pooled AUC %.4f  (%.0fs)"
              % (i, len(tr), len(te), auc, time.time() - t1), flush=True)

        if not args.no_baseline:
            te["p_asset"] = np.nan
            t2 = time.time()
            done = 0
            for a, g in te.groupby("asset"):
                gtr = tr[tr["asset"] == a]
                if len(gtr) < POOL_MIN_BARS or gtr["target"].nunique() < 2:
                    continue
                te.loc[g.index, "p_asset"] = fit_predict(gtr, g, feats)
                done += 1
            print("      per-asset arm: %d assets in %.0fs" % (done, time.time() - t2),
                  flush=True)
        pooled.append(te)

    if not pooled:
        print("no usable folds.")
        return 1
    allte = pd.concat(pooled, ignore_index=True)
    y = allte["target"].astype(int).values

    res = {"rows": len(allte), "assets": int(allte["asset"].nunique()),
           "folds": len(pooled), "features": len(feats),
           "gate_floor": GATE_FLOOR, "gate_alpha": GATE_ALPHA}
    res["pooled_auc"] = float(roc_auc_score(y, allte["p_pool"].values))
    if "next_ret" in allte.columns:
        ok = allte["next_ret"].notna()
        ic, p = sps.spearmanr(allte.loc[ok, "p_pool"], allte.loc[ok, "next_ret"])
        res["pooled_ic"] = float(ic)
        res["pooled_ic_p"] = float(p)
        res["ic_se"] = float(1.0 / np.sqrt(int(ok.sum())))

    print()
    print("POOLED")
    print("  rows %d over %d assets, %d folds" % (res["rows"], res["assets"], res["folds"]))
    print("  out-of-sample AUC : %.4f" % res["pooled_auc"])
    if "pooled_ic" in res:
        print("  information coeff : %+.5f   se %.5f   p %.3g"
              % (res["pooled_ic"], res["ic_se"], res["pooled_ic_p"]))

    if not args.no_baseline and "p_asset" in allte.columns and allte["p_asset"].notna().any():
        have = allte[allte["p_asset"].notna()]
        res["baseline_auc"] = float(roc_auc_score(
            have["target"].astype(int).values, have["p_asset"].values))
        a_pool = per_asset_auc(have, "p_pool")
        a_base = per_asset_auc(have, "p_asset")
        both = sorted(set(a_pool) & set(a_base))
        deltas = [a_pool[a] - a_base[a] for a in both]
        res["paired_assets"] = len(both)
        res["mean_delta"] = float(np.mean(deltas)) if deltas else None
        res["wilcoxon_p"] = (float(sps.wilcoxon(deltas).pvalue)
                             if len(deltas) > 10 else None)
        res["gate_open"] = bool(deltas and res["mean_delta"] >= GATE_FLOOR
                                and res["wilcoxon_p"] is not None
                                and res["wilcoxon_p"] < GATE_ALPHA)
        print()
        print("PER-ASSET ARM, same rows")
        print("  out-of-sample AUC : %.4f" % res["baseline_auc"])
        print("  paired assets     : %d" % len(both))
        print("  mean AUC delta    : %+.5f   (floor %+.3f)"
              % (res["mean_delta"], GATE_FLOOR))
        print("  wilcoxon p        : %.4g" % res["wilcoxon_p"])
        print("  GATE              : %s" % ("OPEN" if res["gate_open"] else "CLOSED"))
        print("  pooled wins on    : %d of %d assets"
              % (sum(1 for d in deltas if d > 0), len(deltas)))

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print()
    print("report written to %s" % REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
