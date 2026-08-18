import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine

from core.features import feature_version

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(f"sqlite:///{DB_PATH}")
    return _ENGINE


def _conn():
    return sqlite3.connect(DB_PATH)


def current_model_version() -> str:
    """Feature-space id of the model writing predictions now (see feature_version)."""
    return feature_version()


def _migrate(cur):
    """Add model_version and meta_prob to an old prediction_log in place (rows before the
    migration keep NULL = legacy generation, so they never blend with the new
    model's track record)."""
    cols = [r[1] for r in cur.execute("PRAGMA table_info(prediction_log)").fetchall()]
    if cols and "model_version" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN model_version TEXT")
    if cols and "meta_prob" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN meta_prob REAL")
    if cols and "sig_shown" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN sig_shown TEXT")
    if cols and "gate_reason" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN gate_reason TEXT")
    if cols and "timing_action" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN timing_action TEXT")
    if cols and "timing_reason" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN timing_reason TEXT")


def _ensure_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            date TEXT,
            asset TEXT,
            signal TEXT,
            probability REAL,
            actual_next_ret REAL,
            correct INTEGER,
            cb_prob REAL,
            lstm_prob REAL,
            model_version TEXT,
            meta_prob REAL,
            sig_shown TEXT,
            gate_reason TEXT,
            timing_action TEXT,
            timing_reason TEXT
        )
    """)
    _migrate(cur)


def _prepare():
    """Ensure the table exists and is migrated before an aggregate read."""
    with _conn() as con:
        _ensure_table(con.cursor())
        con.commit()


def _has_bar(cur, asset, day):
    """True if the asset has a real price bar dated `day` (YYYY-MM-DD). False on a
    non-trading day (weekend/holiday), a missing price table, or a bar not yet
    fetched - i.e. when there is nothing to reconcile the prediction against."""
    try:
        row = cur.execute(
            f'SELECT 1 FROM "{asset.lower()}" WHERE Date = ? LIMIT 1', (day,)
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False


def log_prediction(asset, signal, probability, cb_prob=None, lstm_prob=None,
                   model_version=None, meta_prob=None, date=None,
                   sig_shown=None, gate_reason=None,
                   timing_action=None, timing_reason=None):
    # Date the prediction by the wall clock (one row per asset per day). Non-trading
    # days for an asset (a stock predicted on a weekend/holiday) are not stamped onto
    # a neighbouring bar here; update_actuals() reconciles only exact trading-bar dates
    # and purges the rest, so a closed-market row never double-counts a real move.
    # `date` may be passed to override for backfills/tests; default is today (UTC).
    today = date or datetime.utcnow().strftime("%Y-%m-%d")
    if model_version is None:
        model_version = feature_version()
    with _conn() as con:
        cur = con.cursor()
        _ensure_table(cur)
        # Only log a prediction that lands on a REAL trading bar for this asset. On a
        # weekend/holiday (or before today's bar is fetched) there is no bar to check
        # it against, so the row would sit "awaiting check" forever and skew the
        # pending count - skip it instead of creating an unverifiable row.
        if not _has_bar(cur, asset, today):
            return
        # Skip if already logged today for this asset
        dup = cur.execute(
            "SELECT 1 FROM prediction_log WHERE date=? AND asset=?",
            (today, asset),
        ).fetchone()
        if dup:
            return
        cur.execute(
            """INSERT INTO prediction_log
               (date, asset, signal, probability, actual_next_ret, correct,
                cb_prob, lstm_prob, model_version, meta_prob, sig_shown, gate_reason,
                timing_action, timing_reason)
               VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (today, asset, signal, probability, cb_prob, lstm_prob,
             model_version, meta_prob, sig_shown, gate_reason,
             timing_action, timing_reason),
        )
        con.commit()


def timing_state(asset, cooldown_days=0):
    """Rebuild the timing policy's position state from the shadow log.

    Scans the asset's logged `timing_action` history oldest-first: an
    ``ENTER:+1``/``ENTER:-1`` row sets the position and resets the segment,
    an ``EXIT`` row flattens it, and any other row (HOLD/STAY_OUT) just
    ages the open position by one bar. `actual_next_ret` (filled in later by
    update_actuals) feeds the running segment return/peak used by the
    trailing-stop rule. Returns a FRESH_STATE-shaped dict; a shadow-history-
    free asset gets FRESH_STATE back unchanged.

    `streak`/`last_raw`/`cooldown_left` are rebuilt from the same rows' raw
    `signal` column (the model's un-gated BUY/SELL/WAIT call for that bar):
      - `last_raw`: the LAST row's signal mapped to BUY->1, SELL->-1, else 0.
      - `streak`: count of consecutive trailing rows whose signal maps to the
        same non-zero side as `last_raw` (0 when `last_raw` is 0).
      - `cooldown_left`: `cooldown_days` minus the number of rows logged after
        the last ``EXIT`` row (floored at 0; 0 when no EXIT has ever fired).
    These three values represent the state AS OF the last logged row - i.e.
    exactly what policy_step's `state` argument expects (yesterday's closing
    state). policy_step itself folds in TODAY's bar: it recomputes streak
    from state["last_raw"] + today's raw signal, and decrements
    cooldown_left at the top of the call - so rebuilding through the last
    logged row (and no further) is correct, not off-by-one."""
    from core.timing_policy import FRESH_STATE
    _prepare()
    st = dict(FRESH_STATE)
    with _conn() as con:
        rows = con.execute(
            "SELECT date, timing_action, actual_next_ret, signal FROM prediction_log "
            "WHERE asset=? AND timing_action IS NOT NULL ORDER BY date",
            (asset,)).fetchall()
    for _day, action, ret, _signal in rows:
        if action.startswith("ENTER"):
            st.update(pos=1 if action.endswith("+1") else -1, days_held=0,
                      seg_peak=0.0, seg_ret=0.0)
        elif action == "EXIT":
            st.update(pos=0, days_held=0, seg_peak=0.0, seg_ret=0.0)
        elif st["pos"] != 0:
            st["days_held"] += 1
        if st["pos"] != 0 and ret is not None:
            st["seg_ret"] += st["pos"] * float(ret)
            st["seg_peak"] = max(st["seg_peak"], st["seg_ret"])

    def _raw_side(signal):
        if signal == "BUY":
            return 1
        if signal == "SELL":
            return -1
        return 0

    last_raw = _raw_side(rows[-1][3]) if rows else 0
    streak = 0
    if last_raw != 0:
        for _day, _action, _ret, signal in reversed(rows):
            if _raw_side(signal) == last_raw:
                streak += 1
            else:
                break

    cooldown_left = 0
    if cooldown_days:
        rows_after = 0
        found_exit = False
        for _day, action, _ret, _signal in reversed(rows):
            if action == "EXIT":
                found_exit = True
                break
            rows_after += 1
        if found_exit:
            cooldown_left = max(0, int(cooldown_days) - rows_after)

    st["last_raw"] = last_raw
    st["streak"] = streak
    st["cooldown_left"] = cooldown_left
    return st


def _load_bars(asset, cache):
    """Return a per-asset (Date-indexed, lowercase-columns) close series, cached
    for the duration of one reconcile pass. None if the price table is missing."""
    if asset in cache:
        return cache[asset]
    table = asset.lower()
    try:
        df = pd.read_sql(
            f'SELECT Date, Close FROM "{table}" ORDER BY Date',
            _engine(),
            index_col="Date",
        )
        # data_engine stores columns lowercase (close); normalize so df["close"]
        # never raises KeyError and silently skips the actual-vs-predicted check.
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        df = None
    cache[asset] = df
    return df


def update_actuals():
    """Reconcile logged predictions against the next trading bar. Returns counters
    so callers (the Web UI reconcile button) can report progress.

    A prediction is objectively verifiable only if its date is a real trading bar
    for that asset. Trading days are per-asset: a Saturday is a live bar for crypto
    but a closed market for a stock. So for each row:

      - date is an exact bar with a following bar  - score it (BUY correct if the
        next close rose, SELL if it fell, WAIT never scored).
      - date is an exact bar but the latest one    - stays pending (outcome not
        formed yet: today/yesterday before the next close lands).
      - date has no bar but a later bar exists      - the market was closed that day
        (weekend/holiday); the prediction just reused the prior close and cannot be
        verified, so it is dropped (excluded) rather than mapped onto a neighbouring
        day and double-counted. This is why non-trading days never reach the stats.
      - date has no bar and none follow yet         - a real trading day whose bar is
        not fetched yet - stays pending (never deleted).
    """
    reconciled = 0
    excluded = 0
    cache = {}
    with _conn() as con:
        cur = con.cursor()
        _ensure_table(cur)
        pending = cur.execute(
            "SELECT COUNT(*) FROM prediction_log WHERE actual_next_ret IS NULL"
        ).fetchone()[0]

        # Scan every row (not just pending) so historical phantom rows already
        # scored on a closed-market day get purged too - one reconcile self-heals.
        rows = cur.execute(
            "SELECT rowid, date, asset, signal, actual_next_ret FROM prediction_log"
        ).fetchall()

        for rowid, date_str, asset, signal, anr in rows:
            df = _load_bars(asset, cache)
            if df is None or len(df) == 0:
                continue
            D = pd.Timestamp(date_str).normalize()
            pos = int(df.index.searchsorted(D, side="right")) - 1
            exact = pos >= 0 and df.index[pos] == D
            has_future = pos + 1 < len(df)

            if not exact:
                if has_future:
                    # Non-trading day for this asset (market was closed), yet it has
                    # since traded: unverifiable stale duplicate - remove it.
                    cur.execute("DELETE FROM prediction_log WHERE rowid=?", (rowid,))
                    excluded += 1
                # else: bar simply not fetched yet - leave pending, never delete.
                continue

            if anr is not None:
                continue  # already scored on a real bar
            if not has_future:
                continue  # exact bar but latest - outcome not formed yet

            today_close = df["close"].iloc[pos]
            next_close = df["close"].iloc[pos + 1]
            if today_close == 0:
                continue
            ret = (next_close - today_close) / today_close

            if signal == "WAIT":
                correct = None
            elif signal == "BUY":
                correct = 1 if ret > 0 else 0
            else:  # SELL
                correct = 1 if ret < 0 else 0

            cur.execute(
                "UPDATE prediction_log SET actual_next_ret=?, correct=? WHERE rowid=?",
                (ret, correct, rowid),
            )
            reconciled += 1

        con.commit()
    return {"pending": pending, "reconciled": reconciled, "excluded": excluded}


def _date_filter(days):
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


def get_accuracy(asset=None, days=30, model_version=None):
    _prepare()
    cutoff = _date_filter(days)
    params = [cutoff]
    where = "WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL"
    if asset:
        where += " AND asset = ?"
        params.append(asset)
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)

    df = pd.read_sql(
        f"SELECT signal, correct FROM prediction_log {where}",
        _engine(),
        params=tuple(params),
    )

    if df.empty:
        return {"accuracy": None, "total_predictions": 0, "correct_count": 0, "by_signal": {}}

    total = len(df)
    correct = int(df["correct"].sum())
    accuracy = correct / total if total else None

    by_signal = {}
    for sig in ["BUY", "SELL"]:
        sub = df[df["signal"] == sig]
        count = len(sub)
        acc = sub["correct"].sum() / count if count else None
        by_signal[sig] = {"acc": acc, "count": count}

    return {
        "accuracy": accuracy,
        "total_predictions": total,
        "correct_count": correct,
        "by_signal": by_signal,
    }


def get_accuracy_history(asset=None, window=7, model_version=None):
    _prepare()
    params = []
    where = "WHERE signal != 'WAIT' AND correct IS NOT NULL"
    if asset:
        where += " AND asset = ?"
        params.append(asset)
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)

    df = pd.read_sql(
        f"SELECT date, correct FROM prediction_log {where} ORDER BY date",
        _engine(),
        params=tuple(params) if params else None,
    )

    if df.empty:
        return pd.DataFrame(columns=["date", "rolling_acc", "predictions_count"])

    daily = df.groupby("date").agg(correct_sum=("correct", "sum"), count=("correct", "count")).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["rolling_correct"] = daily["correct_sum"].rolling(window, min_periods=1).sum()
    daily["rolling_count"] = daily["count"].rolling(window, min_periods=1).sum()
    daily["rolling_acc"] = daily["rolling_correct"] / daily["rolling_count"]
    daily["predictions_count"] = daily["rolling_count"].astype(int)
    return daily[["date", "rolling_acc", "predictions_count"]]


def get_leaderboard(days=30, model_version=None):
    _prepare()
    cutoff = _date_filter(days)
    params = [cutoff]
    where = "WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL"
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)
    df = pd.read_sql(
        f"SELECT asset, correct FROM prediction_log {where}",
        _engine(),
        params=tuple(params),
    )

    if df.empty:
        return pd.DataFrame(columns=["Asset", "Accuracy", "Predictions", "Correct"])

    grp = df.groupby("asset").agg(
        Predictions=("correct", "count"),
        Correct=("correct", "sum"),
    ).reset_index()
    grp = grp[grp["Predictions"] >= 5].copy()
    grp["Accuracy"] = grp["Correct"] / grp["Predictions"]
    grp = grp.rename(columns={"asset": "Asset"})
    grp = grp.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return grp[["Asset", "Accuracy", "Predictions", "Correct"]]


def get_daily_stats(days=30, model_version=None):
    _prepare()
    cutoff = _date_filter(days)
    params = [cutoff]
    where = "WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL"
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)
    df = pd.read_sql(
        f"SELECT date, correct FROM prediction_log {where}",
        _engine(),
        params=tuple(params),
    )

    if df.empty:
        return pd.DataFrame(columns=["Date", "Predictions", "Correct", "Accuracy"])

    daily = df.groupby("date").agg(
        Predictions=("correct", "count"),
        Correct=("correct", "sum"),
    ).reset_index()
    daily["Accuracy"] = daily["Correct"] / daily["Predictions"]
    daily = daily.rename(columns={"date": "Date"})
    return daily.sort_values("Date").reset_index(drop=True)


META_SHADOW_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)


def meta_shadow_report(days=30, model_version=None):
    """Shadow evaluation of the meta-sizing gate. Over reconciled directional
    predictions that carry a meta_prob (logged while GTRADE_META_SIZING=shadow), it
    asks: if we gated - kept only signals with meta_prob >= thr and sent the rest to
    WAIT - would the acted-on accuracy beat acting on all of them? That is the
    shadow-active decision signal. Returns {"rows": 0} until shadow data exists."""
    _prepare()
    cutoff = _date_filter(days)
    params = [cutoff]
    where = ("WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL "
             "AND meta_prob IS NOT NULL")
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)
    df = pd.read_sql(
        f"SELECT correct, meta_prob FROM prediction_log {where}",
        _engine(),
        params=tuple(params),
    )
    if df.empty:
        return {"rows": 0}

    total = len(df)
    base_acc = float(df["correct"].mean())
    sweep = []
    for thr in META_SHADOW_THRESHOLDS:
        kept = df[df["meta_prob"] >= thr]
        n = len(kept)
        acc = float(kept["correct"].mean()) if n else None
        sweep.append({
            "thr": thr,
            "kept": n,
            "coverage": n / total,
            "accuracy": acc,
            "lift": (acc - base_acc) if acc is not None else None,
        })
    hi = df[df["meta_prob"] >= 0.5]["correct"]
    lo = df[df["meta_prob"] < 0.5]["correct"]
    return {
        "rows": total,
        "baseline_accuracy": base_acc,
        "sweep": sweep,
        "discrimination": {
            "high_meta_acc": float(hi.mean()) if len(hi) else None,
            "high_meta_n": len(hi),
            "low_meta_acc": float(lo.mean()) if len(lo) else None,
            "low_meta_n": len(lo),
        },
    }


if __name__ == "__main__":
    print("Updating actuals...")
    res = update_actuals()
    print("Reconciled %d of %d pending." % (res["reconciled"], res["pending"]))
    if res.get("excluded"):
        print("Excluded %d prediction(s) on non-trading days (market closed)." % res["excluded"])

    lb = get_leaderboard(days=30)
    if lb.empty:
        print("No leaderboard data (need >= 5 predictions per asset).")
    else:
        lb_display = lb.copy()
        lb_display["Accuracy"] = lb_display["Accuracy"].map("{:.1%}".format)
        print("\n=== LEADERBOARD (last 30 days) ===")
        print(lb_display.to_string(index=False))

    ver = current_model_version()
    for label, mv in (("ALL GENERATIONS", None), (f"CURRENT MODEL [{ver}]", ver)):
        overall = get_accuracy(days=30, model_version=mv)
        print(f"\n=== OVERALL ACCURACY (last 30 days) - {label} ===")
        if overall["accuracy"] is None:
            print("No data yet.")
        else:
            print(f"Accuracy : {overall['accuracy']:.1%}")
            print(f"Total    : {overall['total_predictions']}")
            print(f"Correct  : {overall['correct_count']}")
            for sig, stats in overall["by_signal"].items():
                acc_str = f"{stats['acc']:.1%}" if stats["acc"] is not None else "N/A"
                print(f"  {sig}: {acc_str} ({stats['count']} predictions)")

    ms = meta_shadow_report(days=30)
    print("\n=== META-SIZING SHADOW (last 30 days) ===")
    if ms["rows"] == 0:
        print("No shadow data yet (run predict with GTRADE_META_SIZING=shadow, then reconcile).")
    else:
        print(f"Signals with meta_prob : {ms['rows']}   baseline acc {ms['baseline_accuracy']:.1%}")
        d = ms["discrimination"]
        hi = f"{d['high_meta_acc']:.1%}" if d["high_meta_acc"] is not None else "N/A"
        lo = f"{d['low_meta_acc']:.1%}" if d["low_meta_acc"] is not None else "N/A"
        print(f"  meta>=0.5: {hi} ({d['high_meta_n']})   meta<0.5: {lo} ({d['low_meta_n']})")
        print(f"  {'thr':>5} {'kept':>6} {'coverage':>9} {'acc':>7} {'lift':>7}")
        for s in ms["sweep"]:
            acc = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "N/A"
            lift = f"{s['lift']:+.1%}" if s["lift"] is not None else "N/A"
            print(f"  {s['thr']:>5.2f} {s['kept']:>6} {s['coverage']:>8.0%} {acc:>7} {lift:>7}")


# --- issued trade levels and what they actually did --------------------------
#
# The same two-step shape as prediction_log: log what was ISSUED on the day, and
# reconcile it against later bars in a separate pass. Levels were the one thing
# the product tells a user to act on that nothing recorded, so "did the levels
# make money" could not be answered for a single day of history.
#
# What is stored is what was SHOWN, trailing stop included. A stop the user never
# saw would make the outcome a measurement of a different strategy.

def _ensure_level_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS level_log (
            date TEXT,
            asset TEXT,
            signal TEXT,
            close REAL,
            atr REAL,
            entry_low REAL,
            entry_high REAL,
            stop REAL,
            trailing INTEGER,
            entered INTEGER,
            entry_date TEXT,
            entry_price REAL,
            exit_date TEXT,
            exit_price REAL,
            exit_reason TEXT,
            bars_held INTEGER,
            ret_net REAL
        )
    """)


def _issued_levels(asset, signal):
    """The levels row this asset would show right now, trailing stop included."""
    from core import levels as levels_mod
    from core import positions as positions_mod
    from core import track_record

    track = track_record.asset_track(asset, limit=60)
    segment = None
    if track:
        segs = positions_mod.build_positions(list(reversed(track)))["segments"]
        if segs and segs[-1]["open"]:
            segment = segs[-1]
    return levels_mod.levels(track_record.ohlc_series(asset, days=60), signal,
                             segment=segment)


def log_levels(asset, signal, date=None, row=None):
    """Record the levels issued for `asset` today. One row per asset per day.

    Same guards as log_prediction and for the same reasons: a day with no real
    bar for this asset can never be reconciled, and a second call on one day
    must not create a second row. Only an actionable row is stored - WAIT has no
    side, so it has no entry zone and no stop, and a row of nulls would only
    dilute the outcome table.
    """
    today = date or datetime.utcnow().strftime("%Y-%m-%d")
    lv = row if row is not None else _issued_levels(asset, signal)
    if not lv or lv.get("status") != "ok":
        return False
    with _conn() as con:
        cur = con.cursor()
        _ensure_level_table(cur)
        if not _has_bar(cur, asset, today):
            return False
        if cur.execute("SELECT 1 FROM level_log WHERE date=? AND asset=?",
                       (today, asset)).fetchone():
            return False
        cur.execute(
            """INSERT INTO level_log
               (date, asset, signal, close, atr, entry_low, entry_high, stop,
                trailing, entered, entry_date, entry_price, exit_date,
                exit_price, exit_reason, bars_held, ret_net)
               VALUES (?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)""",
            (today, asset, signal, lv["close"], lv["atr"], lv["entry_low"],
             lv["entry_high"], lv["stop"], 1 if lv["trailing"] else 0))
        con.commit()
    return True


def _load_ohlc(asset, cache):
    """Date-indexed OHLC for one asset, cached for one reconcile pass."""
    if asset in cache:
        return cache[asset]
    try:
        df = pd.read_sql(f'SELECT Date, Open, High, Low, Close FROM "{asset.lower()}" '
                         f'ORDER BY Date', _engine(), index_col="Date")
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        df = None
    cache[asset] = df
    return df


def _shown_signals(cur, asset):
    """{date: signal the user was shown} for one asset, for the exit rule."""
    rows = cur.execute(
        "SELECT date, COALESCE(sig_shown, signal) FROM prediction_log WHERE asset=?",
        (asset,)).fetchall()
    return {d: s for d, s in rows}


def _resolve_level_row(row, df, shown, leg_cost):
    """What became of one issued level, or None while it is still undecided.

    Fills are limit-order conventions taken at the WORSE edge of the zone, and a
    gap through the stop fills at the gap, so nothing here flatters the result.
    The trade ends on the stop or on the signal turning away from the side it was
    issued for, which is the rule the product states in the asset card and the
    one core/positions.py already uses for a segment.
    """
    date_str, signal, entry_low, entry_high, stop = row
    side = 1 if signal == "BUY" else -1 if signal == "SELL" else 0
    if not side or df is None or len(df) == 0:
        return None
    start = int(df.index.searchsorted(pd.Timestamp(date_str).normalize(),
                                      side="right"))
    entry_date = entry_price = None
    held = 0
    for i in range(start, len(df)):
        bar = df.iloc[i]
        day = df.index[i].strftime("%Y-%m-%d")
        flipped = shown.get(day) not in (None, signal)
        if entry_price is None:
            touched = (bar["low"] <= entry_high) if side > 0 else (bar["high"] >= entry_low)
            if touched:
                entry_price = (min(bar["open"], entry_high) if side > 0
                               else max(bar["open"], entry_low))
                entry_date = day
                continue          # costs and stops are judged from the next bar
            if flipped:
                return {"entered": 0, "exit_reason": "no_entry", "ret_net": 0.0,
                        "bars_held": 0}
            continue
        held += 1
        hit = (bar["low"] <= stop) if side > 0 else (bar["high"] >= stop)
        if hit:
            exit_price = (min(bar["open"], stop) if side > 0
                          else max(bar["open"], stop))
            reason = "stop"
        elif flipped:
            exit_price = bar["close"]
            reason = "signal"
        else:
            continue
        gross = side * (exit_price - entry_price) / entry_price
        return {"entered": 1, "entry_date": entry_date, "entry_price": float(entry_price),
                "exit_date": day, "exit_price": float(exit_price),
                "exit_reason": reason, "bars_held": held,
                "ret_net": float(gross - 2 * leg_cost)}
    return None                    # still running, or not enough bars yet


def update_level_outcomes():
    """Score every issued level that has since resolved. Idempotent."""
    from core.backtesting import COMMISSION, SLIPPAGE

    leg_cost = COMMISSION + SLIPPAGE
    cache, shown_cache = {}, {}
    resolved = 0
    with _conn() as con:
        cur = con.cursor()
        _ensure_level_table(cur)
        _ensure_table(cur)
        rows = cur.execute(
            "SELECT rowid, date, asset, signal, entry_low, entry_high, stop "
            "FROM level_log WHERE exit_reason IS NULL").fetchall()
        pending = len(rows)
        for rowid, date_str, asset, signal, lo, hi, stop in rows:
            if asset not in shown_cache:
                shown_cache[asset] = _shown_signals(cur, asset)
            out = _resolve_level_row((date_str, signal, lo, hi, stop),
                                     _load_ohlc(asset, cache), shown_cache[asset],
                                     leg_cost)
            if out is None:
                continue
            cur.execute(
                "UPDATE level_log SET entered=?, entry_date=?, entry_price=?, "
                "exit_date=?, exit_price=?, exit_reason=?, bars_held=?, ret_net=? "
                "WHERE rowid=?",
                (out.get("entered"), out.get("entry_date"), out.get("entry_price"),
                 out.get("exit_date"), out.get("exit_price"), out["exit_reason"],
                 out.get("bars_held"), out.get("ret_net"), rowid))
            resolved += 1
        con.commit()
    return {"pending": pending, "resolved": resolved}
