"""What every policy layer decided, and what it was worth on live signals.

Run:  python policy_status.py [--days N]

Two halves, deliberately separated. The first is what each offline fit
concluded, read straight out of its own report file. The second is the
reconciliation: the signals production actually emitted, the returns that
actually happened, and what each layer would have earned on them, charged the
same commission and slippage the backtest charges.

A backtest verdict and a live reading are not the same claim, and a layer with
no live decisions logged reads as "no data" rather than as a neutral result.
"""
import argparse
import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

import config
from core import policy_report as pr
from core import sizing_policy as sp
from core.features import compute_taleb_risk

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
THRESHOLDS_PATH = os.path.join(BASE, "models", "tuned_thresholds.json")

FOREX_GROUPS = ("FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC")


def forex_assets():
    return {a for g in FOREX_GROUPS for a in config.ASSET_TYPES.get(g, [])}


def _table_name(asset):
    from core.track_record import _table_name as tn
    return tn(asset)


def live_rows(conn, days=None):
    """prediction_log as dicts, newest window first if `days` is given."""
    # timing_stage arrives by migration on the first write after the upgrade, so
    # a database that has only been READ since then does not have it yet. Absent
    # means Stage A: it is the only policy that had ever served.
    have = {r[1] for r in conn.execute("PRAGMA table_info(prediction_log)")}
    stage = "timing_stage" if "timing_stage" in have else "NULL AS timing_stage"
    q = ("SELECT date, asset, signal, probability, actual_next_ret, "
         "timing_action, %s FROM prediction_log" % stage)
    if days:
        q += (" WHERE date >= (SELECT MAX(date) FROM prediction_log)"
              " AND 1=1")
    df = pd.read_sql(q, conn)
    if days:
        keep = sorted(df["date"].unique())[-int(days):]
        df = df[df["date"].isin(keep)]
    return df.to_dict("records")


def sizing_sizes(conn, thresholds, policy):
    """A callable for policy_report.reconcile that sizes one asset's rows.

    The features the rule needs are the ones the offline fit used, rebuilt from
    the raw bars: nothing about the size depends on the champion, so this needs
    no model and no feature engineering run.
    """
    cache = {}

    def _bars(asset):
        if asset in cache:
            return cache[asset]
        try:
            df = pd.read_sql('SELECT * FROM "%s"' % _table_name(asset), conn,
                             index_col="Date", parse_dates=["Date"])
        except Exception:
            cache[asset] = None
            return None
        if df.empty:
            cache[asset] = None
            return None
        df = df[~df.index.duplicated(keep="last")].sort_index()
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean() / (df["close"] + 1e-9)
        df["taleb_hi"] = (compute_taleb_risk(df["close"]) > 0.7).fillna(False)
        cache[asset] = df
        return df

    def sizes(asset, rows, probs, sides):
        df = _bars(asset)
        if df is None:
            return None
        idx = pd.to_datetime([r["date"] for r in rows])
        try:
            part = df.reindex(idx).ffill()
        except Exception:
            return None
        if part["close"].isna().all():
            return None
        thr = thresholds.get(asset, {})
        series = {
            "probs": probs,
            "close": part["close"].to_numpy(dtype=float),
            "atr": np.nan_to_num(part["atr"].to_numpy(dtype=float)),
            "taleb_hi": part["taleb_hi"].to_numpy(dtype=bool),
            "buy_thr": float(thr.get("buy", 0.55)),
            "sell_thr": float(thr.get("sell", 0.45)),
        }
        return sp.match_exposure(policy.sizes_for(series), sides)

    return sizes


def lines(reports, recon, days_seen):
    out = ["=== WHAT EACH POLICY CONCLUDED (backtest) ==="]
    for r in reports:
        if not r["present"]:
            out.append("  %-14s not fitted yet (%s)" % (r["name"], r["file"]))
            continue
        head = "  %-14s %-12s" % (r["name"], r.get("verdict") or "-")
        if r.get("n"):
            head += " n=%-4s" % r["n"]
        if r.get("mean_d") is not None:
            head += " mean %+.3f" % r["mean_d"]
        if r.get("median_d") is not None:
            head += " median %+.3f" % r["median_d"]
        if r.get("up") is not None:
            head += " up %d/%d" % (r["up"], r["up"] + r["down"])
        out.append(head)
        out.append("                 %s" % r["what"])
    out += ["", "=== WHAT HAPPENED ON LIVE SIGNALS (%d trading days) ===" % days_seen,
            "  %-9s %7s %7s %9s %8s %8s %7s"
            % ("arm", "assets", "rows", "profit%", "winrate", "sharpe", "trades")]
    for name in ("emitted", "timing A", "timing B", "sizing"):
        a = recon.get(name) or {}
        if a.get("status") != "measured":
            out.append("  %-9s %s" % (name, "no live decisions logged"))
            continue
        out.append("  %-9s %7d %7d %+9.3f %8.1f %8.3f %7d"
                   % (name, a["assets"], a["rows"], a["profit"], a["winrate"],
                      a["sharpe"], a["trades"]))
    out += ["",
            "  emitted  = the signal production actually sent, as a position.",
            "  timing A = the adopted rules' shadow decision, logged beside it.",
            "  timing B = the fitted Q's, when GTRADE_TIMING_STAGE=b is set.",
            "             Separate rows on separate days: the two are NOT",
            "             comparable until both have run over the same ones.",
            "  sizing   = the fitted rule, at matched exposure so it cannot win",
            "             by simply holding more."]
    return out


def accuracy_by_confidence(conn):
    """Live directional accuracy in confidence buckets.

    The one reading that decides whether an overlay can help at all: a policy
    that sizes up on strong signals is worth nothing if strong signals are the
    wrong ones. Measured 2026-08-21 over 4717 rows, accuracy FELL as confidence
    rose, so this belongs next to every policy verdict rather than in a
    one-off script.
    """
    out = []
    for lo, hi in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50)):
        row = conn.execute(
            "SELECT COUNT(*), AVG(CASE WHEN correct=1 THEN 1.0 ELSE 0.0 END), "
            "AVG(CASE WHEN signal='BUY' THEN actual_next_ret "
            "         ELSE -actual_next_ret END) "
            "FROM prediction_log WHERE actual_next_ret IS NOT NULL "
            "AND signal IN ('BUY','SELL') AND correct IS NOT NULL "
            "AND ABS(probability-0.5) >= ? AND ABS(probability-0.5) < ?",
            (lo, hi)).fetchone()
        n, acc, ret = row
        if n:
            out.append({"bucket": "%.2f-%.2f" % (lo, hi), "rows": int(n),
                        "accuracy": float(acc) * 100.0,
                        "mean_ret": float(ret or 0.0)})
    return out


# Where each adopted policy recorded the assets its gate actually scored. A
# policy cannot be validated out of sample without knowing what its sample was,
# and all three already write it - so the unseen set is derived, never guessed.
_FIT_SETS = {
    "timing (Stage A)": ("timing_report.json", ("per_asset",)),
    "sizing":           ("sizing_report.json", ("per_asset",)),
    "levels":           ("levels_policy.json", ("gate", "per_asset")),
}


def print_unseen():
    """Per adopted policy, the assets it has never been scored on.

    These are the only assets on which a rule that is already adopted can be
    checked without the check being a re-reading of the data that adopted it.
    """
    from config import FULL_ASSET_MAP

    for name, (fname, path) in _FIT_SETS.items():
        blob = pr._read(os.path.join(BASE, fname))
        if not blob:
            print("  %-18s %s is not there - nothing was fitted yet."
                  % (name, fname))
            continue
        note = blob.get("per_asset_lost") if isinstance(blob, dict) else None
        for key in path:
            blob = (blob or {}).get(key) or {}
        if not blob:
            # An empty fit set is not "it saw nothing" - it is "this file no
            # longer says". Reporting every asset as unseen would invite a
            # replication run against a set nobody can check.
            print("  %-18s %s records no per-asset set, so the unseen set "
                  "cannot be derived." % (name, fname))
            if note:
                print("    %s" % note)
            print()
            continue
        unseen = [a for a in FULL_ASSET_MAP if a not in blob]
        print("  %-18s scored on %d, never scored on %d"
              % (name, len(blob), len(unseen)))
        if unseen:
            print("    " + ",".join(unseen))
        print()
    print("  An asset with no champion cannot be scored at all - run")
    print("  `python model_health.py --missing` first and train those.")


def main():
    ap = argparse.ArgumentParser(description="policy results and live check")
    ap.add_argument("--days", type=int, default=0,
                    help="reconcile only the last N trading days (0 = all)")
    ap.add_argument("--unseen", action="store_true",
                    help="per adopted policy, the assets its gate never scored "
                         "- the only honest out-of-sample set for it")
    args = ap.parse_args()

    if args.unseen:
        print_unseen()
        return

    conn = sqlite3.connect(DB_PATH)
    rows = live_rows(conn, args.days or None)
    days_seen = len({r["date"] for r in rows})
    thresholds = pr._read(THRESHOLDS_PATH) or {}
    fitted = pr._read(os.path.join(BASE, "sizing_report.json")) or {}
    policy = (sp.SizingPolicy(fitted.get("params"))
              if fitted.get("params") else None)
    sizes = sizing_sizes(conn, thresholds, policy) if policy else None
    recon = pr.reconcile(rows, forex=forex_assets(), sizing=sizes)
    buckets = accuracy_by_confidence(conn)
    reports = pr.reports()
    for line in lines(reports, recon, days_seen):
        print(line)
    print()
    print("=== LIVE ACCURACY BY CONFIDENCE ===")
    print("  %-12s %7s %9s %12s" % ("|prob-0.5|", "rows", "accuracy", "mean ret"))
    for b in buckets:
        print("  %-12s %7d %8.1f%% %+12.5f"
              % (b["bucket"], b["rows"], b["accuracy"], b["mean_ret"]))
    # Written for the /research page, which must never recompute this on a
    # request: the sizing arm reads every asset's bars.
    with open(os.path.join(BASE, "policy_status.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"computed": datetime.now().isoformat(timespec="seconds"),
                   "days": days_seen, "reports": reports, "live": recon,
                   "buckets": buckets}, fh, indent=1, default=float)
    conn.close()


if __name__ == "__main__":
    main()
