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
    # Which policy decided. Without it the live timing number silently becomes a
    # blend of two policies over two date ranges the moment Stage B is switched
    # on, and the reason string is the wrong thing to infer it from.
    if cols and "timing_stage" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN timing_stage TEXT")
    # What the CHALLENGER would have done on the same bar, when it is only being
    # watched (GTRADE_TIMING_STAGE=shadow). Its own column, because the served
    # decision in timing_action is what the card shows, the levels follow and
    # timing_state rebuilds a position from: overwriting it with a policy nobody
    # acted on is how a watched challenger quietly becomes the live one.
    if cols and "shadow_action" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN shadow_action TEXT")


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
            timing_reason TEXT,
            timing_stage TEXT,
            shadow_action TEXT
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
                   timing_action=None, timing_reason=None,
                   timing_stage=None, shadow_action=None):
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
                timing_action, timing_reason, timing_stage, shadow_action)
               VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (today, asset, signal, probability, cb_prob, lstm_prob,
             model_version, meta_prob, sig_shown, gate_reason,
             timing_action, timing_reason, timing_stage, shadow_action),
        )
        con.commit()


def last_logged_prob(asset, db_path=None):
    """Yesterday's champion probability for `asset`, or None.

    Stage B's state carries prob_d1, the one-bar change in probability, and
    serving has only today's number. The log already holds every prior one, so
    it is read rather than recomputed: re-scoring the previous bar would mean
    a second champion forward pass per asset per day for one scalar.
    """
    import sqlite3

    path = db_path or DB_PATH
    try:
        with sqlite3.connect(path) as con:
            row = con.execute(
                "SELECT probability FROM prediction_log WHERE asset = ? "
                "AND probability IS NOT NULL ORDER BY date DESC LIMIT 1",
                (asset,)).fetchone()
    except Exception:
        return None
    return float(row[0]) if row and row[0] is not None else None


def timing_state(asset, cooldown_days=0, column="timing_action"):
    """Rebuild the timing policy's position state from the shadow log.

    `column` picks WHOSE history to rebuild: the served decisions in
    `timing_action`, or a watched challenger's in `shadow_action`. A challenger
    holds its own position - it enters and exits on different bars - so reading
    the served column for it would feed it someone else's state and make every
    comparison meaningless.

    Scans the asset's logged history oldest-first: an
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
        if column not in ("timing_action", "shadow_action"):
            raise ValueError("unknown timing column: %r" % (column,))
        rows = con.execute(
            f"SELECT date, {column}, actual_next_ret, signal FROM prediction_log "
            f"WHERE asset=? AND {column} IS NOT NULL ORDER BY date",
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
    from core import dashboard

    taleb_hi, risky = dashboard.regime_flags(asset)
    return levels_mod.levels(track_record.ohlc_series(asset, days=60), signal,
                             segment=segment, taleb_hi=taleb_hi, risky=risky)


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
    """Adapter: one level_log row plus this asset's bars -> core.levels.resolve_trade.

    The rule itself lives in core/levels.py so the fitter in train_levels.py
    optimises exactly what this journal scores.
    """
    from core.levels import resolve_trade

    date_str, signal, entry_low, entry_high, stop = row
    side = 1 if signal == "BUY" else -1 if signal == "SELL" else 0
    if not side or df is None or len(df) == 0:
        return None
    start = int(df.index.searchsorted(pd.Timestamp(date_str).normalize(),
                                      side="right"))
    after = df.iloc[start:]
    if len(after) == 0:
        return None
    bars = list(zip(after["open"], after["high"], after["low"], after["close"]))
    days = [d.strftime("%Y-%m-%d") for d in after.index]
    # The side in force on each later bar. A day with no logged prediction is
    # not a flip: absence of a row is absence of an instruction, and treating it
    # as one would close trades on days the pipeline simply did not run.
    sides = []
    for day in days:
        sig = shown.get(day)
        sides.append(side if sig is None else
                     (1 if sig == "BUY" else -1 if sig == "SELL" else 0))
    out = resolve_trade(side, entry_low, entry_high, stop, bars, sides, leg_cost)
    if out is None:
        return None
    if out.get("entered"):
        out["entry_date"] = days[out.pop("entry_index")]
        out["exit_date"] = days[out.pop("exit_index")]
    return out


def level_summary(days=None):
    """What the issued levels actually did, or as much of it as has resolved.

    The journal had no reader at all: it was written every day and scored on the
    next run, and the only way to see either was SQL by hand. So the one number
    the product tells a person to act on was measurable and unmeasured.

    Two populations, deliberately separate. `resolved` counts issues whose fate
    is known, and `entered` counts the subset that ever filled - a wide entry
    zone means many issues honestly end `no_entry`, and folding those into the
    return would report a strategy that took trades it never took. avg_ret and
    win_pct are over ENTERED rows only, for that reason.

    Everything is None when nothing has resolved yet, which is the truthful
    answer on a young journal rather than a zero that reads as a flat result.
    """
    with _conn() as con:
        cur = con.cursor()
        _ensure_level_table(cur)
        where, args = "", []
        if days:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            where, args = " WHERE date >= ?", [cutoff]
        row = cur.execute(
            "SELECT COUNT(*), "
            "       SUM(exit_reason IS NOT NULL), "
            "       SUM(entered = 1), "
            "       SUM(exit_reason = 'stop'), "
            "       SUM(exit_reason = 'signal'), "
            "       SUM(exit_reason = 'no_entry'), "
            "       AVG(CASE WHEN entered = 1 THEN ret_net END), "
            "       SUM(CASE WHEN entered = 1 AND ret_net > 0 THEN 1 ELSE 0 END), "
            "       MIN(date), MAX(date) "
            "FROM level_log" + where, args).fetchone()
        issued, resolved, entered, stopped, flipped, no_entry, avg_ret, wins, d0, d1 = row
    entered = entered or 0
    return {
        "issued": issued or 0, "resolved": resolved or 0, "entered": entered,
        "pending": (issued or 0) - (resolved or 0),
        "stopped": stopped or 0, "flipped": flipped or 0, "no_entry": no_entry or 0,
        "avg_ret": float(avg_ret) if avg_ret is not None else None,
        "win_pct": (100.0 * (wins or 0) / entered) if entered else None,
        "first": d0, "last": d1,
    }


def level_summary_lines(s=None):
    """level_summary as console text, for the __main__ block and for a log."""
    s = level_summary() if s is None else s
    if not s["issued"]:
        return ["=== TRADE LEVELS ===",
                ("No levels issued yet. predict.py writes one row per actionable "
                 "asset per day.")]
    out = ["=== TRADE LEVELS (%s to %s) ===" % (s["first"], s["last"]),
           "Issued   : %d  (%d resolved, %d still open)"
           % (s["issued"], s["resolved"], s["pending"])]
    if not s["resolved"]:
        out.append("Nothing has resolved yet, so there is no outcome to report. "
                   "A trade resolves when the stop is hit or the signal turns away.")
        return out
    out += ["Filled   : %d of %d resolved (%d never reached the entry zone)"
            % (s["entered"], s["resolved"], s["no_entry"]),
            "Exits    : %d on the stop, %d on the signal turning"
            % (s["stopped"], s["flipped"])]
    if s["entered"]:
        out += ["Net/trade: %+.4f  (both legs charged)" % s["avg_ret"],
                "Winners  : %.1f%% of filled trades" % s["win_pct"]]
    return out


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
            "SELECT rowid, date, asset, signal, entry_low, entry_high, stop, "
            "trailing FROM level_log WHERE exit_reason IS NULL").fetchall()
        pending = len(rows)
        skipped = 0
        for rowid, date_str, asset, signal, lo, hi, stop, trailing in rows:
            # A trailing row is not a setup. Its stop belongs to a position
            # opened days ago at a different price and has since ratcheted to
            # sit close to today's close - that is what a trailing stop is FOR.
            # Replaying it as "enter at the zone edge today, exit at that stop"
            # invents a trade nobody could take, with a stop a fraction of the
            # width the policy asks for. Measured 2026-08-22 on the first twelve
            # resolutions: all twelve were trailing rows, median distance from
            # entry to stop 0.56 ATR against the 2.99 the policy specifies, and
            # all twelve "lost" on bar one. The fitter scores fresh setups, so
            # the journal must score fresh setups, or the two numbers are not
            # the same number.
            if trailing:
                cur.execute(
                    "UPDATE level_log SET exit_reason=? WHERE rowid=?",
                    ("not a setup: position already open", rowid))
                skipped += 1
                continue
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
    return {"pending": pending, "resolved": resolved, "not_setups": skipped}


# `probability` is P(up), so a SELL's confidence is 1 - probability, not
# probability itself: bucketing on raw probability puts every SELL below
# 0.50 regardless of how confident it was, and a bucket floor of 0.50 then
# silently drops the whole SELL book (2769 of 4961 reconciled rows measured
# 2026-08-24, 56 percent of the sample). Buckets are over
# MAX(probability, 1 - probability), the axis that covers both books. The
# (0.50, 0.55) band is dropped: on the same measurement only 10 of 4961 rows
# land there, too few to read.
CALIBRATION_BUCKETS = ((0.55, 0.60), (0.60, 0.70), (0.70, 1.01))


def calibration(buckets=None, db_path=None):
    """Accuracy per confidence bucket over reconciled predictions.

    Confidence is MAX(probability, 1 - probability): the same axis for a BUY
    and a SELL. Bucketing on raw `probability` instead reads the SELL half of
    the table backwards, since a low `probability` is a CONFIDENT sell, not
    an unconfident one.
    """
    out = []
    try:
        con = sqlite3.connect("file:%s?mode=ro" % (db_path or DB_PATH), uri=True)
    except sqlite3.Error:
        return []
    try:
        for lo, hi in (buckets or CALIBRATION_BUCKETS):
            row = con.execute(
                "SELECT COUNT(*), AVG(correct) FROM prediction_log "
                "WHERE correct IS NOT NULL "
                "AND MAX(probability, 1 - probability) >= ? "
                "AND MAX(probability, 1 - probability) < ?", (lo, hi)).fetchone()
            out.append({"lo": lo, "hi": hi, "n": row[0], "accuracy": row[1]})
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return out


def accuracy_by_asset(db_path=None):
    """Per-asset accuracy over reconciled predictions, worst first."""
    try:
        con = sqlite3.connect("file:%s?mode=ro" % (db_path or DB_PATH), uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(
            "SELECT asset, COUNT(*), AVG(correct) FROM prediction_log "
            "WHERE correct IS NOT NULL GROUP BY asset "
            "ORDER BY AVG(correct) ASC").fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [{"asset": r[0], "n": r[1], "accuracy": r[2]} for r in rows]


def level_outcomes(db_path=None):
    """The per-trade rows behind the resolved levels, newest first.

    Counts belong to level_summary, the one place "resolved" (exit_reason IS
    NOT NULL) and "entered" are defined. This returns only what
    level_summary does not carry: the row list itself. An aggregate here
    under the same names but a different WHERE clause (ret_net IS NOT NULL)
    is a second, disagreeing definition of "resolved" on the same page.
    """
    try:
        con = sqlite3.connect("file:%s?mode=ro" % (db_path or DB_PATH), uri=True)
    except sqlite3.Error:
        return {"rows": []}
    try:
        rows = con.execute(
            "SELECT date, asset, signal, exit_reason, bars_held, ret_net "
            "FROM level_log WHERE ret_net IS NOT NULL "
            "ORDER BY date DESC").fetchall()
    except sqlite3.Error:
        return {"rows": []}
    finally:
        con.close()
    return {"rows": [{"date": r[0], "asset": r[1], "signal": r[2],
                      "exit_reason": r[3], "bars_held": r[4], "ret_net": r[5]}
                     for r in rows]}


if __name__ == "__main__":
    print("Updating actuals...")
    res = update_actuals()
    print("Reconciled %d of %d pending." % (res["reconciled"], res["pending"]))
    if res.get("excluded"):
        print("Excluded %d prediction(s) on non-trading days (market closed)." % res["excluded"])

    # Levels resolve on later bars exactly like predictions do, so they are
    # reconciled in the same pass. Only predict.py did this, which meant running
    # the tracker by hand reported stale level outcomes and never said so.
    lv = update_level_outcomes()
    print("Levels: scored %d of %d unresolved (%d were trailing rows, not "
          "setups)." % (lv["resolved"], lv["pending"], lv.get("not_setups", 0)))

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

    print()
    for line in level_summary_lines():
        print(line)

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
