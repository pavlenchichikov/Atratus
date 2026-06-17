"""update_actuals must reconcile logged predictions against the next-day close.

Regression test for the silent KeyError('Close') bug: price tables store columns
lowercase (data_engine lowercases them), so df["Close"] raised and every row was
skipped, leaving accuracy "0 verified" forever.
"""
import sqlite3


def _seed(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    # real data_engine schema: capital "Date" index column, lowercase OHLC
    cur.execute("CREATE TABLE btc (Date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    cur.executemany("INSERT INTO btc VALUES (?,?,?,?,?,?)", [
        ("2026-06-12", 1, 1, 1, 100.0, 1),
        ("2026-06-13", 1, 1, 1, 110.0, 1),   # +10% next day
    ])
    cur.execute(
        "CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, probability REAL, "
        "actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL)"
    )
    cur.execute("INSERT INTO prediction_log VALUES ('2026-06-12','BTC','BUY',0.7,NULL,NULL,NULL,NULL)")
    con.commit()
    con.close()


def test_update_actuals_reconciles_against_next_close(tmp_path, monkeypatch):
    import performance_tracker as pt
    db = str(tmp_path / "t.db")
    _seed(db)
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)  # rebuild engine against temp DB

    pt.update_actuals()

    con = sqlite3.connect(db)
    ret, correct = con.execute(
        "SELECT actual_next_ret, correct FROM prediction_log"
    ).fetchone()
    con.close()
    assert ret is not None and abs(ret - 0.10) < 1e-9   # 100 to 110
    assert correct == 1                                  # BUY and price rose
