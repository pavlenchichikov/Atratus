"""Unit tests for core.analyst.payoff (pure, no I/O)."""

import pytest

from core.analyst.payoff import K_SHRINK, ret_atr, shrink, to_pct


def test_return_in_atr_units_is_the_move_divided_by_the_atr_fraction():
    # A 2% move on an asset whose ATR is 1% of its close is a 2 ATR move.
    assert ret_atr(0.02, atr=1.0, close=100.0) == pytest.approx(2.0)


def test_side_flips_the_sign():
    # The same downward move is a gain for a short and a loss for a long.
    assert ret_atr(-0.02, atr=1.0, close=100.0, side=-1) == pytest.approx(2.0)
    assert ret_atr(-0.02, atr=1.0, close=100.0, side=1) == pytest.approx(-2.0)


@pytest.mark.parametrize("atr,close", [(None, 100.0), (1.0, None), (0.0, 100.0),
                                       (-1.0, 100.0), (1.0, 0.0)])
def test_degenerate_scale_returns_none_rather_than_dividing(atr, close):
    # A zero or missing ATR is an asset with no measurable volatility yet.
    # Returning None makes the caller drop the row; returning inf would put a
    # nonsense number on the card.
    assert ret_atr(0.02, atr=atr, close=close) is None


def test_to_pct_is_the_inverse_scaled_by_today():
    # 2 ATR on an asset whose ATR is 1.5% of today's close is a 3% move.
    assert to_pct(2.0, atr_today=1.5, close_today=100.0) == pytest.approx(0.03)


def test_to_pct_guards_the_same_degenerate_inputs():
    assert to_pct(2.0, atr_today=None, close_today=100.0) is None
    assert to_pct(2.0, atr_today=1.5, close_today=0.0) is None


def test_shrink_with_no_observations_is_the_prior():
    assert shrink(0, mean=5.0, prior=1.0) == pytest.approx(1.0)


def test_shrink_at_k_observations_is_the_midpoint():
    assert shrink(K_SHRINK, mean=3.0, prior=1.0) == pytest.approx(2.0)


def test_shrink_with_many_observations_approaches_the_mean():
    assert shrink(100000, mean=3.0, prior=1.0) == pytest.approx(3.0, abs=1e-3)


import sqlite3

import train_payoff


def _fixture_db(tmp_path, n_buy=60, move=0.02, n_wait=0, wait_move=0.0):
    """A market.db stand-in: one asset, flat bars, and n_buy BUY signals that
    each preceded the same +move day. The expected payoff is therefore exactly
    move / (atr/close), with no averaging noise to reason about.

    `n_wait` adds unsignalled bars that moved `wait_move`. They exist only to
    give the drift something to be measured on: it is the whole window, not the
    bars the ensemble spoke on, and a fixture with no WAIT rows cannot tell a
    payoff apart from the market that produced it."""
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE sber (Date TEXT, Open REAL, Close REAL, '
                'High REAL, Low REAL)')
    con.execute('CREATE TABLE prediction_log (date TEXT, asset TEXT, '
                'signal TEXT, actual_next_ret REAL)')
    bars, preds = [], []
    price = 100.0
    for i in range(200):
        day = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        bars.append((day, price, price, price + 1.0, price - 1.0))
        if 20 <= i < 20 + n_buy:
            preds.append((day, "SBER", "BUY", move))
        elif 20 + n_buy <= i < 20 + n_buy + n_wait:
            preds.append((day, "SBER", "WAIT", wait_move))
    con.executemany('INSERT INTO sber VALUES (?,?,?,?,?)', bars)
    con.executemany('INSERT INTO prediction_log VALUES (?,?,?,?)', preds)
    con.commit()
    con.close()
    return path


def test_table_reports_payoff_in_atr_units_not_percent(tmp_path):
    db = _fixture_db(tmp_path)
    table = train_payoff.build_table(db_path=db)
    cell = table["asset"]["SBER"]["BUY"]
    assert cell["n"] == 60
    # Flat bars with a constant 2.0 high-low range give ATR 2.0 on close 100,
    # a 2% unit, so a +2% move is exactly 1.0 ATR.
    assert cell["mean"] == pytest.approx(1.0, abs=1e-6)


def test_sell_rows_are_signed_by_side(tmp_path):
    db = _fixture_db(tmp_path)
    con = sqlite3.connect(db)
    con.execute("UPDATE prediction_log SET signal='SELL'")
    con.commit()
    con.close()
    table = train_payoff.build_table(db_path=db)
    # The same +2% move is a 1.0 ATR loss for a short.
    assert table["asset"]["SBER"]["SELL"]["mean"] == pytest.approx(-1.0, abs=1e-6)


def test_the_class_cell_pools_its_assets(tmp_path):
    db = _fixture_db(tmp_path)
    table = train_payoff.build_table(db_path=db)
    # config.radar_category maps SBER to the MOEX group, whose css name is "ru".
    assert table["class"]["ru"]["BUY"]["n"] == 60


def test_quantiles_bracket_the_mean(tmp_path):
    db = _fixture_db(tmp_path)
    cell = train_payoff.build_table(db_path=db)["asset"]["SBER"]["BUY"]
    assert cell["q10"] <= cell["mean"] <= cell["q90"]


def test_an_asset_with_no_signals_is_absent_rather_than_zero(tmp_path):
    db = _fixture_db(tmp_path, n_buy=0)
    table = train_payoff.build_table(db_path=db)
    # An absent key makes the caller fall back to the class prior. A zero cell
    # would look like a measured "no edge" and be indistinguishable from it.
    assert "SBER" not in table["asset"]


def test_a_call_that_only_matched_the_market_has_no_excess(tmp_path):
    """The 2026-09-03 finding, as a fixture.

    Every bar rose 1 ATR and the ensemble called BUY on half of them. The raw
    payoff is +1.0 and reads like an edge; the excess is zero, because holding
    the thing over the same window paid exactly the same. This is the shape the
    RU class was actually in: BUY -0.116 ATR raw, of which -0.101 was a market
    that fell on every bar, signalled or not."""
    db = _fixture_db(tmp_path, n_buy=60, move=0.02, n_wait=60, wait_move=0.02)
    cell = train_payoff.build_table(db_path=db)["asset"]["SBER"]["BUY"]
    assert cell["mean"] == pytest.approx(1.0, abs=1e-6)
    assert cell["drift"] == pytest.approx(1.0, abs=1e-6)
    assert cell["drift_n"] == 120
    assert cell["excess"] == pytest.approx(0.0, abs=1e-6)


def test_excess_keeps_the_part_of_the_payoff_the_market_did_not_give(tmp_path):
    # BUY bars rose 2 ATR, the unsignalled ones 1 ATR. The window average is
    # 1.5, so half of the raw +2.0 was the market and half was the call.
    db = _fixture_db(tmp_path, n_buy=60, move=0.04, n_wait=60, wait_move=0.02)
    cell = train_payoff.build_table(db_path=db)["asset"]["SBER"]["BUY"]
    assert cell["mean"] == pytest.approx(2.0, abs=1e-6)
    assert cell["drift"] == pytest.approx(1.5, abs=1e-6)
    assert cell["excess"] == pytest.approx(0.5, abs=1e-6)


def test_the_drift_is_signed_by_side(tmp_path):
    """A rising market is a cost to a short, so subtracting the same number
    from both sides would credit the short with the loss it just took."""
    db = _fixture_db(tmp_path, n_buy=60, move=0.02, n_wait=60, wait_move=0.02)
    con = sqlite3.connect(db)
    con.execute("UPDATE prediction_log SET signal='SELL' WHERE signal='BUY'")
    con.commit()
    con.close()
    cell = train_payoff.build_table(db_path=db)["asset"]["SBER"]["SELL"]
    assert cell["mean"] == pytest.approx(-1.0, abs=1e-6)
    assert cell["drift"] == pytest.approx(-1.0, abs=1e-6)
    assert cell["excess"] == pytest.approx(0.0, abs=1e-6)


def test_too_short_a_window_carries_no_drift_key_at_all(tmp_path):
    """Absent rather than zero, for the same reason an asset with no signals is
    absent: a zero would be read as a measured "the market gave nothing"."""
    db = _fixture_db(tmp_path, n_buy=train_payoff.MIN_DRIFT - 1)
    cell = train_payoff.build_table(db_path=db)["asset"]["SBER"]["BUY"]
    assert "drift" not in cell and "excess" not in cell


def test_wait_rows_never_become_a_payoff_cell(tmp_path):
    # They inform the drift and nothing else. A WAIT is not a recommendation.
    db = _fixture_db(tmp_path, n_buy=60, n_wait=60, wait_move=0.02)
    table = train_payoff.build_table(db_path=db)
    assert set(table["asset"]["SBER"]) == {"BUY"}
