"""Forex bars get the day's real close, and the same-bar leak goes with it.

2026-09-26: Yahoo `=X` daily bars stored the day's open as close while high/low
covered the whole day, and (close - low) / (high - low) predicted the next-bar
direction label at AUC 0.87-0.90 on every pair.
"""
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import data_engine as de
import repair_fx_close as fxc


def _yahoo_like(n=400, seed=0):
    """A random walk published the way Yahoo publishes FX: close = the day's
    open, high/low = the day's real range."""
    rng = np.random.default_rng(seed)
    opens = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, n + 1)))
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for i, d in enumerate(dates):
        o, c = opens[i], opens[i + 1]
        rows.append({"Date": d, "Open": o, "Close": o,
                     "High": max(o, c) * 1.001, "Low": min(o, c) * 0.999, "Volume": 0.0})
    return pd.DataFrame(rows)


def _leak_auc(df):
    x = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)
    m = df["Close"].shift(-1).notna()
    return roc_auc_score(y[m], -x[m])


def test_the_fetch_takes_close_from_the_next_open_and_the_leak_goes():
    raw = _yahoo_like()
    assert _leak_auc(raw) > 0.95, "positive control: the Yahoo shape leaks"
    fixed = de._fx_real_close(raw, {})
    assert abs(_leak_auc(fixed) - 0.5) < 0.1
    assert (fixed["Close"].values == raw["Open"].values[1:]).all()
    assert (fixed["High"] >= fixed["Close"]).all() and (fixed["Low"] <= fixed["Close"]).all()


def test_the_snapshot_and_the_bar_without_a_successor_are_not_stored():
    fri = pd.Timestamp("2026-09-25 02:00")
    snap = datetime(2026, 9, 26, 0, 29)
    raw = pd.DataFrame({"Date": [pd.Timestamp("2026-09-24 02:00"), fri, pd.Timestamp(snap)],
                        "Open": [1.0, 1.1, 1.2], "Close": [1.0, 1.1, 1.25],
                        "High": [1.2, 1.3, 1.3], "Low": [0.9, 1.0, 1.1], "Volume": [0.0] * 3})
    out = de._fx_real_close(raw, {"regularMarketTime": int(snap.timestamp())})
    assert list(out["Date"]) == [pd.Timestamp("2026-09-24 02:00")]
    assert out["Close"].iloc[0] == 1.1, "Thursday closes at Friday's open"


def _db(tmp_path, rows):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE eurusd (Date TEXT, open REAL, close REAL, high REAL, low REAL, volume REAL)')
    con.executemany("INSERT INTO eurusd VALUES (?,?,?,?,?,0)", rows)
    con.commit()
    con.close()
    return path


def test_the_repair_rewrites_the_table_and_a_second_run_changes_nothing(tmp_path):
    rows = [("2026-09-21", 1.00, 1.00, 1.05, 0.98),
            ("2026-09-22", 1.02, 1.02, 1.04, 1.00),
            ("2026-09-23", 1.03, 1.03, 1.06, 1.01),
            ("2026-09-24", 1.07, 1.07, 1.08, 1.04),   # next open is above this high
            ("2026-09-25", 1.06, 1.06, 1.09, 1.05),   # newest weekday: no successor
            ("2026-09-26", 1.06, 1.08, 1.09, 1.05)]   # Saturday snapshot
    path = _db(tmp_path, rows)
    # The every-run scan never judges the newest row: a repaired one can sit
    # near its open too, and it would be cut on every run.
    assert fxc.totals(fxc.scan(path, tables=["eurusd"])) == (4, 1, 1)
    plans = fxc.scan(path, tables=["eurusd"], newest=True)
    assert fxc.totals(plans) == (4, 2, 1)
    fxc.apply(path, plans)

    con = sqlite3.connect(path)
    got = con.execute("SELECT Date, close, high, low FROM eurusd ORDER BY Date").fetchall()
    con.close()
    assert [g[0] for g in got] == ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]
    assert [g[1] for g in got] == [1.02, 1.03, 1.07, 1.06]
    assert got[2][2] == 1.07, "high widened to take in the close"
    assert fxc.totals(fxc.scan(path, tables=["eurusd"])) == (0, 0, 0)
