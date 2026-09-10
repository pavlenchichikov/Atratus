"""What the dead band costs, measured on predictions the system refused to act on.

    python threshold_report.py
    python threshold_report.py --group asia

`core/scoring.py` turns a probability into BUY above 0.55, SELL below 0.45 and
WAIT in between (per-asset overrides in models/tuned_thresholds.json). That band
was calibrated when CatBoost was over-confident from the weekly leak
(CB_Acc 0.6326, of which 0.1027 was future). Honest probabilities sit closer to
0.5, so the same band now refuses far more often, and it refuses hardest exactly
where the model is most accurate: measured 2026-09-10, the twelve Asia-Pacific
indices answered WAIT on 60% of days against 29% everywhere else.

That refusal is not free. It is why 383 assets have no champion at all: a fold
is admitted only with 10 trades in it, honest models make fewer, folds fail, and
the asset is skipped with "no robust folds".

WAIT rows are never scored (`correct` stays NULL, correctly - a refusal is not a
directional call). So this recomputes the outcome from the bars: for every logged
probability, what the next bar actually did, and what acting on it would have
been worth. Sweeping the band then answers the only question that matters here -
does acting more often buy trades at the same accuracy, or just noise.
"""
import argparse
import collections
import os
import sqlite3
import statistics as st

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "market.db")

ASIA = ("TAIEX", "NIKKEI", "NZ50", "ASX200", "KLCI", "SENSEX", "STI", "KOSPI",
        "HANGSENG", "NIFTY", "SHANGHAI", "JAKARTA")
BANDS = (0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.0)
MIN_ROWS = 50


def _table(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def closes(con, asset):
    """{date: close} for one asset, or {} when it has no table."""
    try:
        rows = con.execute('SELECT Date, close FROM "%s" ORDER BY Date'
                           % _table(asset)).fetchall()
    except sqlite3.Error:
        return {}, []
    seq = [(str(d)[:10], float(c)) for d, c in rows if c]
    return {d: i for i, (d, _c) in enumerate(seq)}, seq


def outcomes(con, rows):
    """[(asset, prob, realised next return, logged signal)] for what can be scored."""
    cache = {}
    out = []
    for date, asset, sig, prob in rows:
        if prob is None:
            continue
        if asset not in cache:
            cache[asset] = closes(con, asset)
        idx, seq = cache[asset]
        pos = idx.get(str(date)[:10])
        if pos is None or pos + 1 >= len(seq):
            continue
        today, nxt = seq[pos][1], seq[pos + 1][1]
        if not today:
            continue
        out.append((asset, float(prob), (nxt - today) / today, sig))
    return out


def sweep(data, bands=BANDS):
    """[(half band, acted, accuracy)] - accuracy of acting on the implied side."""
    res = []
    for w in bands:
        acted = [(p, r) for _a, p, r, _s in data if abs(p - 0.5) > w and r != 0]
        if not acted:
            res.append((w, 0, None))
            continue
        hit = sum(1 for p, r in acted if (p > 0.5) == (r > 0))
        res.append((w, len(acted), hit / len(acted)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=("all", "asia", "rest"), default="all")
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    rows = con.execute("SELECT date, asset, signal, probability FROM prediction_log "
                       "WHERE probability IS NOT NULL").fetchall()
    if args.group == "asia":
        rows = [r for r in rows if r[1] in ASIA]
    elif args.group == "rest":
        rows = [r for r in rows if r[1] not in ASIA]
    data = outcomes(con, rows)
    print("logged predictions with a probability : %d" % len(rows))
    print("of them scoreable against a next bar  : %d" % len(data))
    if not data:
        return 1

    # Positive control. Every row the system called BUY must sit above 0.5 here,
    # or the logged probability is not the one the decision was taken on and
    # nothing below this line means anything.
    buys = [p for _a, p, _r, s in data if s == "BUY"]
    sells = [p for _a, p, _r, s in data if s == "SELL"]
    if buys and sells:
        agree = (sum(1 for p in buys if p > 0.5) + sum(1 for p in sells if p < 0.5))
        print("logged side agrees with the logged probability: %d of %d"
              % (agree, len(buys) + len(sells)))

    waits = [d for d in data if d[3] == "WAIT"]
    print()
    print("acting on the REFUSALS alone: n=%d  accuracy %.4f"
          % (len(waits), st.mean(1 if (p > 0.5) == (r > 0) else 0
                                 for _a, p, r, _s in waits) if waits else 0.0))
    acted = [d for d in data if d[3] in ("BUY", "SELL")]
    print("what it already acts on     : n=%d  accuracy %.4f"
          % (len(acted), st.mean(1 if (p > 0.5) == (r > 0) else 0
                                 for _a, p, r, _s in acted) if acted else 0.0))

    print()
    print("%-10s %8s %10s   %s" % ("half band", "acted", "accuracy", "note"))
    for w, n, acc in sweep(data):
        note = "the band in use today" if abs(w - 0.05) < 1e-9 else ""
        print("%-10.3f %8d %10s   %s"
              % (w, n, "%.4f" % acc if acc is not None else "-", note))

    print()
    print("per asset, at the widest and the narrowest band:")
    per = collections.defaultdict(list)
    for a, p, r, _s in data:
        per[a].append((p, r))
    wide = narrow = 0
    for a, v in per.items():
        if len(v) < MIN_ROWS:
            continue
        w5 = [(p, r) for p, r in v if abs(p - 0.5) > 0.05 and r != 0]
        w0 = [(p, r) for p, r in v if r != 0]
        if len(w5) >= 10:
            wide += 1
        if len(w0) >= 10:
            narrow += 1
    print("  assets with >= 10 acted days: %d at band 0.05, %d at band 0.0"
          % (wide, narrow))
    print("  (a fold needs 10 trades to be admitted; that floor is why "
          "assets come back \"no robust folds\")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
