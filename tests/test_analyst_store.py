"""core.analyst.store: schema, write path, and the outcome backfill."""

import sqlite3

import pytest

from core.analyst import store


@pytest.fixture()
def db(tmp_path):
    """A market.db stand-in with one asset's bars. Never the real database."""
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE sber (Date TEXT, Open REAL, Close REAL, '
                'High REAL, Low REAL)')
    # 20 flat bars so ATR exists, then a +2% day.
    rows = [(f"2026-01-{i + 1:02d}", 100.0, 100.0, 101.0, 99.0) for i in range(20)]
    rows.append(("2026-01-21", 100.0, 102.0, 102.5, 99.5))
    con.executemany('INSERT INTO sber VALUES (?,?,?,?,?)', rows)
    con.commit()
    con.close()
    store.ensure_table(path)
    return path


def _judgment(**over):
    row = {"date": "2026-01-20", "asset": "SBER", "horizon": 1,
           "direction": "up", "conviction": 3, "vol_regime": "normal",
           "key_risk": "none", "thesis": "flat tape", "evidence_json": "[]",
           "dossier_hash": "abc", "llm_model": "test",
           "forecast_pct": 0.01, "lo_pct": -0.01, "hi_pct": 0.03,
           "atr_at_signal": 2.0, "close_at_signal": 100.0}
    row.update(over)
    return row


@pytest.fixture()
def db_falling(tmp_path):
    """Same shape as `db`, but the last bar FALLS 2% instead of rising."""
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE sber (Date TEXT, Open REAL, Close REAL, '
                'High REAL, Low REAL)')
    rows = [(f"2026-01-{i + 1:02d}", 100.0, 100.0, 101.0, 99.0) for i in range(20)]
    rows.append(("2026-01-21", 100.0, 98.0, 100.5, 97.5))
    con.executemany('INSERT INTO sber VALUES (?,?,?,?,?)', rows)
    con.commit()
    con.close()
    store.ensure_table(path)
    return path


def _down_judgment(**over):
    # forecast_pct=0.02 is a +2% SHORT PAYOFF, not a +2% price move: direction
    # "down" pays off positively when the price falls (payoff = side * ret,
    # side=-1 for a short). See store._SIDE and payoff.ret_atr.
    row = {"date": "2026-01-20", "asset": "SBER", "horizon": 1,
           "direction": "down", "conviction": 3, "vol_regime": "normal",
           "key_risk": "none", "thesis": "downside", "evidence_json": "[]",
           "dossier_hash": "abc", "llm_model": "test",
           "forecast_pct": 0.02, "lo_pct": 0.0, "hi_pct": 0.04,
           "atr_at_signal": 2.0, "close_at_signal": 100.0}
    row.update(over)
    return row


def test_a_down_judgment_that_was_right_scores_a_low_error_and_lands_inside(db_falling):
    # The price fell exactly 2% - a short's payoff is +2%, which is exactly
    # what this "down" judgment forecast. The analyst was perfectly right, so
    # the score must show a near-zero error and an interval hit. Before the
    # Finding-1 fix, the direction-blind backfill compared this PAYOFF-space
    # forecast against the RAW (unflipped) return and produced the opposite:
    # abs_err_atr=2.0, inside_interval=0.
    store.write_judgment(_down_judgment(), db_path=db_falling)
    filled = store.backfill_outcomes(db_path=db_falling, today="2026-01-22")
    assert filled == 1

    r = store.scored_rows(db_path=db_falling)[0]
    # -2% raw return, side=-1 -> +1.0 ATR of short payoff, matching the
    # +2%-payoff forecast (0.02 / (2.0/100.0) = 1.0 ATR) almost exactly.
    assert r["realized_atr_units"] == pytest.approx(1.0)
    assert r["abs_err_atr"] == pytest.approx(0.0)
    assert r["inside_interval"] == 1


def test_a_down_judgment_that_was_wrong_scores_a_high_error(db):
    # Mirror of the test above: same "down" judgment (forecasting a +2%
    # short payoff), but on the `db` fixture where the price instead ROSE 2%
    # - the short lost money. The analyst was exactly wrong, so the score
    # must show a large error and a miss.
    store.write_judgment(_down_judgment(), db_path=db)
    filled = store.backfill_outcomes(db_path=db, today="2026-01-22")
    assert filled == 1

    r = store.scored_rows(db_path=db)[0]
    # +2% raw return, side=-1 -> -1.0 ATR of short payoff, as far as possible
    # from the +1.0 ATR (0.02 payoff) forecast.
    assert r["realized_atr_units"] == pytest.approx(-1.0)
    assert r["abs_err_atr"] == pytest.approx(2.0)
    assert r["inside_interval"] == 0


def test_ensure_table_is_idempotent(db):
    store.ensure_table(db)
    store.ensure_table(db)
    con = sqlite3.connect(db)
    cols = [d[1] for d in con.execute("pragma table_info(analyst_log)")]
    con.close()
    assert "realized_ret" in cols and "abs_err_atr" in cols


def test_a_written_judgment_starts_unscored(db):
    store.write_judgment(_judgment(), db_path=db)
    assert store.pending_count(db_path=db) == 1
    assert store.scored_rows(db_path=db) == []


def test_backfill_fills_the_next_bar_outcome(db):
    store.write_judgment(_judgment(), db_path=db)
    filled = store.backfill_outcomes(db_path=db, today="2026-01-22")
    assert filled == 1

    rows = store.scored_rows(db_path=db)
    assert len(rows) == 1
    r = rows[0]
    # close went 100.0 -> 102.0 on the bar after 2026-01-20.
    assert r["realized_ret"] == pytest.approx(0.02)
    # atr_at_signal 2.0 on close 100.0 is a 2% unit, so a 2% move is 1.0 ATR.
    assert r["realized_atr_units"] == pytest.approx(1.0)
    # forecast 0.01 is 0.5 ATR, realized is 1.0 ATR.
    assert r["abs_err_atr"] == pytest.approx(0.5)
    assert r["inside_interval"] == 1


def test_backfill_is_idempotent(db):
    store.write_judgment(_judgment(), db_path=db)
    assert store.backfill_outcomes(db_path=db, today="2026-01-22") == 1
    assert store.backfill_outcomes(db_path=db, today="2026-01-22") == 0


def test_a_judgment_with_no_next_bar_yet_stays_pending(db):
    store.write_judgment(_judgment(date="2026-01-21"), db_path=db)
    assert store.backfill_outcomes(db_path=db, today="2026-01-22") == 0
    assert store.pending_count(db_path=db) == 1


def test_a_realized_move_outside_the_interval_is_marked(db):
    store.write_judgment(_judgment(hi_pct=0.005), db_path=db)
    store.backfill_outcomes(db_path=db, today="2026-01-22")
    assert store.scored_rows(db_path=db)[0]["inside_interval"] == 0


def test_a_missing_asset_table_does_not_raise(db):
    store.write_judgment(_judgment(asset="NOSUCH"), db_path=db)
    assert store.backfill_outcomes(db_path=db, today="2026-01-22") == 0


def test_a_repeated_dossier_is_recognised(db):
    assert store.judged_with_hash("SBER", "abc", db_path=db) is False
    store.write_judgment(_judgment(), db_path=db)
    assert store.judged_with_hash("SBER", "abc", db_path=db) is True
    assert store.judged_with_hash("SBER", "different", db_path=db) is False


def test_today_in_the_future_holds_back_an_available_bar(db):
    # date="2026-01-20" has a target bar (2026-01-21) already in the series,
    # so only the `today` guard -- not the "no next bar yet" guard -- can
    # be the thing blocking this fill.
    store.write_judgment(_judgment(date="2026-01-20"), db_path=db)
    assert store.backfill_outcomes(db_path=db, today="2026-01-20") == 0
    assert store.pending_count(db_path=db) == 1
