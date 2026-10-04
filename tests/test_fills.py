"""The fills journal: what the account holds, and what the sheet does with it."""
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import fills
from core import levels as lv


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "market.db")


def test_a_fill_opens_and_closes(db):
    fills.record("SBER", "buy", 100, 315.4, entry_date="2026-09-01", db_path=db)
    assert list(fills.open_fills(db_path=db)) == ["SBER"]
    assert fills.close("SBER", 322.1, exit_date="2026-09-04", db_path=db)
    assert fills.open_fills(db_path=db) == {}
    assert fills.history(db_path=db)[0]["exit_price"] == 322.1


def test_two_open_positions_on_one_asset_are_refused(db):
    fills.record("SBER", "BUY", 100, 315.4, db_path=db)
    with pytest.raises(ValueError, match="already has an open fill"):
        fills.record("SBER", "BUY", 50, 318.0, db_path=db)
    # and the refusal did not eat the first one
    assert len(fills.open_fills(db_path=db)) == 1


def test_closing_nothing_says_so_rather_than_pretending(db):
    assert fills.close("GAZP", 100.0, db_path=db) is None


def test_nonsense_is_refused_at_the_boundary(db):
    with pytest.raises(ValueError):
        fills.record("SBER", "HOLD", 1, 1, db_path=db)
    with pytest.raises(ValueError):
        fills.record("SBER", "BUY", 0, 1, db_path=db)
    with pytest.raises(ValueError):
        fills.record("SBER", "BUY", 1, -5, db_path=db)


def test_no_fill_is_not_an_answer(db):
    """None means "nothing recorded", so the caller keeps its reconstruction."""
    assert fills.open_segment("SBER", db_path=db) is None


def test_the_segment_is_the_shape_the_levels_already_consume(db):
    fills.record("SBER", "SELL", 10, 300.0, entry_date="2026-09-01", db_path=db)
    seg = fills.open_segment("SBER", today="2026-09-05", db_path=db)
    assert seg["side"] == -1
    assert seg["start_date"] == "2026-09-01"
    assert seg["end_date"] is None
    assert seg["open"] is True
    assert seg["bars"] == 4
    assert seg["source"] == "fill"


def test_the_stop_moves_to_the_bar_you_actually_bought_on():
    """The whole point, priced.

    Same bars, same multiplier, two entry dates. The trailing stop is the best
    close - k*ATR seen SINCE entry, so a later entry cannot see the earlier
    highs and must sit lower for a long.
    """
    bars = [{"date": "2026-09-%02d" % d, "open": 100.0 + d, "high": 102.0 + d,
             "low": 99.0 + d, "close": 100.0 + d} for d in range(1, 11)]
    atrs = [1.0] * len(bars)
    early = lv._trailing_stop(bars, atrs, {"start_date": "2026-09-01",
                                           "end_date": None}, 1, 2.0)
    late = lv._trailing_stop(bars, atrs, {"start_date": "2026-09-08",
                                          "end_date": None}, 1, 2.0)
    assert early == pytest.approx(108.0)
    assert late == pytest.approx(108.0)

    # falling instead, so the reconstruction's earlier start is the one that
    # holds the high and the two answers genuinely differ
    bars = [{"date": "2026-09-%02d" % d, "open": 120.0 - d, "high": 122.0 - d,
             "low": 119.0 - d, "close": 120.0 - d} for d in range(1, 11)]
    early = lv._trailing_stop(bars, atrs, {"start_date": "2026-09-01",
                                           "end_date": None}, 1, 2.0)
    late = lv._trailing_stop(bars, atrs, {"start_date": "2026-09-08",
                                          "end_date": None}, 1, 2.0)
    assert early > late, "an entry three days later cannot claim the old high"


def test_reading_an_absent_journal_does_not_create_a_database(tmp_path):
    """The regression this module caused before it was caught.

    levels_sheet asks for an open fill on every row of every sheet. A reader
    that runs CREATE TABLE therefore conjures a market.db anywhere it is
    imported, and the test suite grew a 28 KB stub one exactly that way.
    """
    missing = str(tmp_path / "nothing.db")
    assert fills.open_fills(db_path=missing) == {}
    assert fills.history(db_path=missing) == []
    assert fills.open_segment("SBER", db_path=missing) is None
    assert not os.path.exists(missing)


def test_a_database_without_the_journal_is_not_an_error(tmp_path):
    """An existing market.db that predates this feature must read as empty."""
    import sqlite3
    other = str(tmp_path / "market.db")
    sqlite3.connect(other).execute("CREATE TABLE bars (x INT)")
    assert fills.open_fills(db_path=other) == {}
    assert fills.open_segment("SBER", db_path=other) is None


def _rising_bars(n=60, start=300.0, step=0.5, width=4.0):
    return [{"date": "2026-08-%02d" % (i % 28 + 1), "open": start + i * step,
             "high": start + i * step + width / 2, "low": start + i * step - width / 2,
             "close": start + i * step} for i in range(n)]


def test_my_positions_measures_a_held_trade_from_its_own_entry(db, monkeypatch):
    """The trader's question about a trade already placed: result, the money the
    stop would cost from here, how far that stop is in ordinary days, and
    whether the size fits risk_per_trade."""
    from core import dashboard, track_record
    from risk_manager import RISK_CONFIG

    monkeypatch.setattr(fills, "DB_PATH", db)
    bars = _rising_bars()
    monkeypatch.setattr(track_record, "ohlc_series", lambda asset, days=60: bars)
    monkeypatch.setattr(dashboard, "regime_flags", lambda asset, taleb=None: (False, False))
    monkeypatch.setattr(lv, "load_policy", lambda path=None: None)
    monkeypatch.setitem(RISK_CONFIG, "risk_per_trade", 0.01)
    fills.record("SBER", "BUY", 1000, 310.0, entry_date="2026-08-20", db_path=db)

    (p,) = dashboard.my_positions(equity=1_000_000, today="2026-08-25")
    close = bars[-1]["close"]
    assert p["asset"] == "SBER" and p["side"] == 1 and p["entry"] == 310.0
    assert p["pnl"] == pytest.approx((close - 310.0) * 1000)
    assert p["pnl_pct"] == pytest.approx(close / 310.0 - 1)
    gap = close - p["stop"]
    assert gap > 0
    assert p["risk_to_stop"] == pytest.approx(gap * 1000)
    assert p["risk_pct_equity"] == pytest.approx(gap * 1000 / 1_000_000)
    assert p["fit_qty"] == pytest.approx(1_000_000 * 0.01 / gap)
    assert p["stop_in_days"] == pytest.approx(gap / p["typical"])

    (p0,) = dashboard.my_positions(equity=0.0, today="2026-08-25")
    assert p0["risk_pct_equity"] is None and p0["fit_qty"] is None, "no equity, no money size"
    fills.close("SBER", 330.0, db_path=db)
    assert dashboard.my_positions(equity=1_000_000) == []


def test_a_short_that_went_against_it_risks_from_the_close(db, monkeypatch):
    from core import dashboard, track_record

    monkeypatch.setattr(fills, "DB_PATH", db)
    bars = _rising_bars()
    monkeypatch.setattr(track_record, "ohlc_series", lambda asset, days=60: bars)
    monkeypatch.setattr(dashboard, "regime_flags", lambda asset, taleb=None: (False, False))
    monkeypatch.setattr(lv, "load_policy", lambda path=None: None)
    fills.record("GAZP", "SELL", 10, 310.0, entry_date="2026-08-20", db_path=db)
    (p,) = dashboard.my_positions(equity=100_000, today="2026-08-25")
    assert p["pnl"] < 0, "price rose after the short"
    assert p["stop"] > p["close"] or p["status"] == "stop_breached"
