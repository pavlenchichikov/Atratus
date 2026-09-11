"""Replace daily bars that were stored while their session was still open.

    python heal_partial_bars.py                 # scan only: what differs, per asset
    python heal_partial_bars.py --apply         # replace those bars, reset what they fed
    python heal_partial_bars.py --assets SBER,BTC --since 2026-03-01

Until 2026-09-11 the daily fetchers stored the bar of a session in progress and
resumed from the NEXT day, so the snapshot stayed. Measured against a fresh
fetch: stored closes off by more than 0.1% on 38 of 60 recent days for SBER,
43 of 59 for BTC, 40 of 52 for GOLD, worst 8.4% GAZP and 15.3% ETH; present in
every month of daily runs since March 2026, absent from the bulk-fetched history
before it. data_engine no longer stores an unfinished session; this repairs what
it stored before.

A healed bar also invalidates what was computed from it: prediction outcomes
(actual_next_ret, correct) and resolved trade levels. Those are reset, and the
next predict.py run recomputes them from the corrected closes.

--apply refuses to run while a chunked training run is active: rewriting bars a
trainer is reading gives it a history that is half old, half new.
"""
import argparse
import os
import sqlite3

import pandas as pd

import config

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
TOL = 0.001            # relative, on each of open/high/low/close
PRICE = ("open", "high", "low", "close")


def _training_active():
    import data_engine as de
    return de._training_looks_active()


def fresh_bars(asset, since):
    """Completed daily bars from the source since `since`, normalised the way
    data_engine stores them. None when the source returns nothing."""
    import data_engine as de
    last = pd.Timestamp(since) - pd.Timedelta(days=1)
    if asset in de.MOEX_TARGETS:
        df = de.fetch_moex_smart(config.FULL_ASSET_MAP[asset], last)
    else:
        df = de.fetch_yahoo_smart(asset, last)
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).normalize().strftime("%Y-%m-%d")
    df = df[~df.index.duplicated(keep="last")]
    df, _fixed, _dropped = de.scrub_ohlc(df)
    return df[[c for c in (*PRICE, "volume") if c in df.columns]]


def partial_dates(con, table, fresh):
    """Stored dates whose prices differ from the fresh bar by more than TOL."""
    stored = pd.read_sql(f'SELECT substr(Date, 1, 10) AS d, open, high, low, close FROM "{table}"',
                         con).set_index("d")
    common = stored.index.intersection(fresh.index)
    if not len(common):
        return []
    s = stored.loc[common, list(PRICE)].astype(float)
    f = fresh.loc[common, list(PRICE)].astype(float)
    off = ((s - f).abs() / f.abs().where(f.abs() > 0)).gt(TOL).any(axis=1)
    return sorted(common[off.to_numpy()])


SOURCE_CURRENT_DAYS = 7


def unfinished_dates(con, table, fresh, today=None):
    """Stored dates after the source's last FINISHED session: snapshots of a
    session still running. Comparing prices cannot find them (there is no fresh
    bar yet) and data_engine resumes from the day after, so they would stay.

    Only while the source is CURRENT. A source whose last bar is older than
    SOURCE_CURRENT_DAYS has a hole, not a live session: TON is served empty
    from 2026-06-16, and treating that as unfinished dropped 62 real bars.
    """
    last = fresh.index.max()
    today = pd.Timestamp(today) if today else pd.Timestamp.now().normalize()
    if (today - pd.Timestamp(last)).days > SOURCE_CURRENT_DAYS:
        return []
    return [r[0] for r in con.execute(
        f'SELECT substr(Date, 1, 10) FROM "{table}" WHERE substr(Date, 1, 10) > ? '
        f"ORDER BY 1", (last,))]


def heal_asset(con, asset, table, fresh, dates, drop=()):
    """Replace `dates` with the fresh bars, delete `drop`, and reset the
    outcomes computed from either.

    A prediction dated D is scored on close D and the next close, so the reset
    starts at the stored bar BEFORE the first healed one. A trade level is reset
    when it was issued on or after that day or exited on or after it.
    """
    dates, drop = list(dates), list(drop)
    first = min([*dates, *drop])
    prev = con.execute(f'SELECT max(substr(Date, 1, 10)) FROM "{table}" '
                       f"WHERE substr(Date, 1, 10) < ?", (first,)).fetchone()[0] or first
    gone = [*dates, *drop]
    con.execute(f'DELETE FROM "{table}" WHERE substr(Date, 1, 10) IN ({",".join("?" * len(gone))})',
                gone)
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
    put = [c for c in cols if c != "Date" and c in fresh.columns]
    con.executemany(
        f'INSERT INTO "{table}" (Date, {", ".join(put)}) VALUES (?, {", ".join("?" * len(put))})',
        [(d, *(float(fresh.at[d, c]) for c in put)) for d in dates])
    n_pred = n_lvl = 0
    try:
        n_pred = con.execute(
            "UPDATE prediction_log SET actual_next_ret = NULL, correct = NULL "
            "WHERE upper(asset) = ? AND date >= ?", (asset.upper(), prev)).rowcount
    except sqlite3.OperationalError:
        pass
    try:
        n_lvl = con.execute(
            "UPDATE level_log SET entered = NULL, entry_date = NULL, entry_price = NULL, "
            "exit_date = NULL, exit_price = NULL, exit_reason = NULL, bars_held = NULL, "
            "ret_net = NULL WHERE upper(asset) = ? AND exit_reason IS NOT NULL "
            "AND exit_reason NOT LIKE 'not a setup%' AND (date >= ? OR exit_date >= ?)",
            (asset.upper(), prev, prev)).rowcount
    except sqlite3.OperationalError:
        pass
    con.commit()
    return {"bars": len(dates), "dropped": len(drop), "predictions": n_pred, "levels": n_lvl}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="replace; without it nothing is written")
    ap.add_argument("--since", default="2026-01-01",
                    help="first date to compare (default 2026-01-01; the defect starts in March)")
    ap.add_argument("--assets", default=None, help="comma-separated subset")
    ap.add_argument("--training-stopped", action="store_true",
                    help="the operator states no trainer is running: skip the file-age "
                         "check, which keeps refusing for an hour after a run is killed")
    args = ap.parse_args(argv)

    if args.apply and not args.training_stopped:
        active, age = _training_active()
        if active:
            print("A training run touched its files %.0f minute(s) ago. Rewriting bars it is "
                  "reading would mix two histories; run --apply once it has finished." % age)
            return 2

    assets = ([a.strip().upper() for a in args.assets.split(",") if a.strip()]
              if args.assets else list(config.FULL_ASSET_MAP))
    con = sqlite3.connect(DB_PATH, timeout=60)
    total = {"assets": 0, "bars": 0, "dropped": 0, "predictions": 0, "levels": 0}
    failed = []
    try:
        for i, asset in enumerate(assets, 1):
            table = asset.lower().replace("^", "").replace(".", "").replace("-", "")
            try:
                fresh = fresh_bars(asset, args.since)
                if fresh is None:
                    continue
                dates = partial_dates(con, table, fresh)
                drop = unfinished_dates(con, table, fresh)
            except Exception as exc:
                failed.append("%s (%s)" % (asset, str(exc)[:60]))
                continue
            if not dates and not drop:
                continue
            total["assets"] += 1
            if args.apply:
                out = heal_asset(con, asset, table, fresh, dates, drop=drop)
                for k in ("bars", "dropped", "predictions", "levels"):
                    total[k] += out[k]
                print("[%d/%d] %-12s healed %3d bar(s), dropped %d unfinished, reset %d "
                      "prediction(s), %d level(s)" % (i, len(assets), asset, out["bars"],
                                                      out["dropped"], out["predictions"],
                                                      out["levels"]))
            else:
                total["bars"] += len(dates)
                total["dropped"] += len(drop)
                print("[%d/%d] %-12s %3d bar(s) differ%s, %d unfinished"
                      % (i, len(assets), asset, len(dates),
                         " from %s" % dates[0] if dates else "", len(drop)))
    finally:
        con.close()

    print()
    print("%s: %d asset(s), %d bar(s) replaced, %d unfinished removed%s" % (
        "healed" if args.apply else "would heal", total["assets"], total["bars"],
        total["dropped"],
        ", reset %d prediction outcome(s) and %d level(s)" % (total["predictions"], total["levels"])
        if args.apply else ""))
    if failed:
        print("could not compare: %s" % "; ".join(failed))
    if not args.apply and total["bars"]:
        print("scan only, nothing written. --apply replaces them; then run predict.py "
              "so the reset outcomes are recomputed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
