"""The barrier every model change has to clear: a one-line rule, scored on the
bars the champion was never trained on.

The rule is "follow the last US session": predict the next bar in the direction
of SP500's most recent close-to-close move, inverted for an asset whose lead is
negative (the dollar pairs quoted the other way round). It fits nothing, so it
cannot overfit, and it needs no journal of its own - it is a function of stored
bars, recomputable for any window at any time.

Measured 2026-09-18, walk-forward over 2006-2026 with the ranking allowed to see
only the previous 750 bars: the top 30 assets by |IC| score 0.587 against 0.506
for the book, ahead in 22 periods of 22, while the same machinery with assets
picked at random lands on the book. Live over the journal window, on every day
rather than the days the model chose to speak: 0.5916 of 1954 calls, p 2.8e-16.

So a champion whose out-of-sample accuracy is below this line is not adding
anything on that asset, and the comparison is worth having in front of you
before the next search burns a night.

Deliberately NOT a column inside train_hybrid's quality report: that file is
written only by a training run, so the barrier would refresh once per retrain
and be stale in between. This reads market.db and the champion registry, takes
seconds, and can be run any day.

    python lead_baseline.py             # every asset with a champion
    python lead_baseline.py NIKKEI,GOLD # a subset
"""
import json
import os
import sqlite3
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")
REGISTRY_PATH = os.path.join(BASE_DIR, "models", "champion_registry.json")
OUT_PATH = os.path.join(BASE_DIR, "lead_baseline.json")

LEADER = "SP500"
RANK_BARS = 750     # bars before train_end the sign may be read from
MIN_RANK = 250      # fewer than this and the asset gets no sign at all
MIN_TEST = 30       # fewer scored bars than this and the asset is unmeasurable


def table_of(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def load_closes(con, asset):
    """Close series for one asset, oldest first, or None."""
    try:
        df = pd.read_sql('SELECT * FROM "%s"' % table_of(asset), con, index_col="Date")
    except Exception:
        return None
    col = [c for c in df.columns if c.lower() == "close"]
    if not col:
        return None
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    s = pd.to_numeric(df[col[0]], errors="coerce").dropna()
    return s[s > 0] if len(s) else None


def lead_sign(lead_ret, asset_ret, until):
    """+1, -1 or None: which way this asset follows the leader, read ONLY from
    bars before `until`.

    The sign is part of the rule and must be chosen causally, or the barrier
    becomes the same winner's curse it exists to guard against.

    It is also measured on the SAME relation the rule is scored on - the
    leader's move on date t against the asset's NEXT bar. Reading it from the
    same-date pair instead measures how the two move together, which for an
    asset that alternates can carry the opposite sign and hand the rule an
    inverted vote.
    """
    a = asset_ret[asset_ret.index < until]
    if len(a) < MIN_RANK:
        return None, None
    a = a.tail(RANK_BARS)
    x = lead_ret.reindex(a.index)
    y = asset_ret.shift(-1).reindex(a.index)
    both = x.notna() & y.notna()
    x, y = x[both], y[both]
    if len(x) < MIN_RANK:
        return None, None
    ic = x.corr(y, method="spearman")
    if pd.isna(ic):
        return None, None
    return (1 if ic > 0 else -1), float(ic)


def rule_accuracy(lead_ret, asset_ret, sign, since):
    """How often "follow the last US session" called the next bar, after `since`.

    The prediction made on bar t is about the move t -> t+1, so the leader's
    return ON date t is the one that lands between the two closes. For an asset
    whose session ends before the US one, that is information available before
    the bar being predicted, which is the whole mechanism.
    """
    fut = asset_ret.shift(-1)
    idx = asset_ret.index[asset_ret.index >= since]
    x = lead_ret.reindex(idx)
    y = fut.reindex(idx)
    both = x.notna() & y.notna()
    if int(both.sum()) < MIN_TEST:
        return None, 0
    call = (x[both] > 0) if sign > 0 else (x[both] <= 0)
    hit = call == (y[both] > 0)
    return float(hit.mean()), len(hit)


def evaluate(con, registry, assets=None):
    """One row per asset: the rule's accuracy after train_end against ens_acc."""
    lead = load_closes(con, LEADER)
    if lead is None:
        return []
    lead_ret = lead.pct_change().dropna()
    out = []
    for asset in (assets or sorted(registry)):
        entry = registry.get(asset)
        if not isinstance(entry, dict) or not entry.get("train_end"):
            continue
        s = load_closes(con, asset)
        if s is None:
            continue
        r = s.pct_change().dropna()
        train_end = pd.Timestamp(entry["train_end"])
        sign, ic = lead_sign(lead_ret, r, train_end)
        if sign is None:
            continue
        acc, n = rule_accuracy(lead_ret, r, sign, train_end)
        if acc is None:
            continue
        model = entry.get("ens_acc")
        out.append({"asset": asset, "ic": round(ic, 4), "sign": sign,
                    "rule_acc": round(acc, 4), "n": n,
                    "model_acc": None if model is None else round(float(model), 4),
                    "train_end": entry["train_end"]})
    return out


def report(rows, top=25):
    """The table, strongest lead first, plus the one line that matters."""
    rows = sorted(rows, key=lambda r: -abs(r["ic"]))
    print("%-12s %7s %8s %9s %7s %s" % ("asset", "IC", "rule", "champion", "bars", "train_end"))
    for r in rows[:top]:
        m = "-" if r["model_acc"] is None else "%9.3f" % r["model_acc"]
        print("%-12s %+7.3f %8.3f %s %7d %s"
              % (r["asset"], r["ic"], r["rule_acc"], m, r["n"], r["train_end"]))
    scored = [r for r in rows if r["model_acc"] is not None]
    if not scored:
        return
    strong = [r for r in scored if abs(r["ic"]) >= 0.20]
    for name, group in (("assets with a lead (|IC| >= 0.20)", strong),
                        ("every asset with a champion", scored)):
        if not group:
            continue
        rule = sum(r["rule_acc"] for r in group) / len(group)
        model = sum(r["model_acc"] for r in group) / len(group)
        wins = sum(1 for r in group if r["model_acc"] > r["rule_acc"])
        print("\n%-34s n=%3d | rule %.4f | champion %.4f | champion clears it on %d"
              % (name, len(group), rule, model, wins))
    print("\nThe champion column is its own walk-forward accuracy and the rule column is "
          "scored after train_end,\nso the two windows are close but not identical - read "
          "a few points as a tie, not as a win.")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    assets = ([a.strip().upper() for a in argv[0].split(",") if a.strip()]
              if argv else None)
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        registry = json.load(fh)
    con = sqlite3.connect(DB_PATH)
    try:
        rows = evaluate(con, registry, assets)
    finally:
        con.close()
    if not rows:
        print("nothing to score: no champion carries a train_end with bars after it.")
        return 1
    report(rows)
    if assets:
        # A subset run prints and stores NOTHING. train_sizing and train_timing
        # rewrite their report on every run, and a ten-asset smoke once destroyed
        # the 207-asset evidence that way; a barrier whose file can be replaced
        # by three rows is worth less than no file.
        print("\nsubset run: %s left alone (it holds the whole book)"
              % os.path.basename(OUT_PATH))
        return 0
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"leader": LEADER, "rows": rows}, fh, indent=1)
    print("\nwritten: %s" % os.path.basename(OUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
