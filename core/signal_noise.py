"""Signal against noise, measured on the live journal.

A next-bar move smaller than the asset's own noise says nothing about whether a
call was right: a BUY "wrong" by 0.05% on a stock that moves 2% a day is a coin
landing on its edge. So an outcome counts only when it cleared NOISE_K times
the asset's median absolute daily move over the previous NOISE_WINDOW bars,
known at the close the call was made on.

The same clean outcomes then grade the model's signals by four properties
(confidence, agreement of the four members, days the same call has held, the
timing policy's action). The table is a standing check, not a filter: measured
2026-09-28 on 12507 calls none of the four separated strong calls from weak
ones (every level 0.44-0.53), and it is here so the day one does, it shows.
"""
import math
import os
import sqlite3
from contextlib import closing

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "market.db")

NOISE_K = 0.5
NOISE_WINDOW = 60
_MEMBERS = ("cb_prob", "lstm_prob", "tf_prob", "tcn_prob")


def _score(frame):
    """{n, acc, lo, hi}: hit rate with a two-standard-error band."""
    n = len(frame)
    if not n:
        return {"n": 0, "acc": None, "lo": None, "hi": None}
    acc = float(frame["correct"].mean())
    se = math.sqrt(acc * (1 - acc) / n)
    return {"n": n, "acc": acc, "lo": acc - 2 * se, "hi": acc + 2 * se}


def _noise_bands(con, assets):
    """{asset: Series of the noise band by date string}."""
    from core.track_record import _table_name

    out = {}
    for a in assets:
        try:
            d = pd.read_sql('SELECT Date, close FROM "%s"' % _table_name(a), con)
        except Exception:
            continue
        s = pd.Series(d["close"].to_numpy(), index=d["Date"].astype(str).str[:10])
        s = s[~s.index.duplicated(keep="last")].sort_index()
        move = s.pct_change().abs()
        out[a] = NOISE_K * move.rolling(NOISE_WINDOW, min_periods=20).median()
    return out


def _grades(clean):
    side = clean["signal"].map({"BUY": 1, "SELL": -1})
    conf = (clean["probability"] - 0.5).abs()
    members = clean[list(_MEMBERS)]
    agree = members.sub(0.5).mul(side, axis=0).gt(0).sum(axis=1)
    agree = agree.where(members.notna().all(axis=1))
    policy = clean["timing_action"].fillna("").str.split(":").str[0]
    specs = (
        ("confidence", "Confidence |p - 0.5|",
         pd.cut(conf, [0, 0.05, 0.10, 0.20, 0.5],
                labels=["under 0.05", "0.05-0.10", "0.10-0.20", "over 0.20"])),
        ("agree", "Members on the call's side",
         agree.map(lambda k: None if pd.isna(k) else "%d of 4" % k)),
        ("streak", "Same call days in a row",
         pd.cut(clean["streak"], [0, 1, 2, 4, 9, 10 ** 6],
                labels=["1", "2", "3-4", "5-9", "10+"])),
        ("policy", "Timing policy action", policy.where(policy != "")),
    )
    out = []
    for key, title, levels in specs:
        rows = []
        for label, g in clean.groupby(levels, observed=True, sort=True):
            rows.append({"label": str(label), **_score(g)})
        if rows:
            out.append({"key": key, "title": title, "rows": rows})
    return out


def noise_report(db_path=None, since=None):
    """{raw, clean, noise_share, grades, per_asset} over the model's own BUY/SELL
    calls that have an outcome; `since` limits the journal to dates >= it."""
    empty = {"raw": _score(pd.DataFrame()), "clean": _score(pd.DataFrame()),
             "noise_share": None, "grades": [], "per_asset": {}}
    with closing(sqlite3.connect(db_path or DB_PATH)) as con:
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(prediction_log)")}
        except sqlite3.Error:
            return empty
        if not cols:
            return empty
        wanted = ["date", "asset", "signal", "probability", "actual_next_ret",
                  "correct", "timing_action", *_MEMBERS]
        sel = ", ".join(c if c in cols else "NULL AS %s" % c for c in wanted)
        sql = ("SELECT %s FROM prediction_log WHERE correct IS NOT NULL AND "
               "actual_next_ret IS NOT NULL AND signal IN ('BUY', 'SELL')" % sel)
        params = ()
        if since:
            sql += " AND date >= ?"
            params = (since,)
        j = pd.read_sql(sql + " ORDER BY asset, date", con, params=params)
        if j.empty:
            return empty
        bands = _noise_bands(con, j["asset"].unique())
    j["band"] = [bands[a].get(d) if a in bands else None
                 for a, d in zip(j["asset"], j["date"].astype(str).str[:10])]
    j["streak"] = j.groupby("asset")["signal"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    known = j[j["band"].notna()]
    clean = known[known["actual_next_ret"].abs() >= known["band"]]
    per_asset = {a: _score(g) for a, g in clean.groupby("asset")}
    return {"raw": _score(j), "clean": _score(clean),
            "noise_share": (1 - len(clean) / len(known)) if len(known) else None,
            "grades": _grades(clean), "per_asset": per_asset}
