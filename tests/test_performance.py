"""core/performance.py: what an asset returned over a period.

Every number here is arithmetic on prices, so the fixtures are prices whose
answer can be worked out by hand rather than trusted from a run.
"""
import datetime
import sqlite3

import pytest

from core import performance


def _db(tmp_path, series, table="sber", extra=None):
    """series is [(date, close)]; extra adds a second table for benchmarks."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    for name, rows in [(table, series)] + list((extra or {}).items()):
        con.execute('CREATE TABLE "%s" (Date TEXT, close REAL)' % name)
        con.executemany('INSERT INTO "%s" VALUES (?,?)' % name, rows)
    con.commit()
    con.close()
    return path


def _daily(start, n, first=100.0, step=1.0):
    day = datetime.date.fromisoformat(start)
    return [((day + datetime.timedelta(days=i)).isoformat(), first + i * step)
            for i in range(n)]


def test_the_total_return_is_the_two_ends_of_the_window(tmp_path):
    db = _db(tmp_path, _daily("2026-01-01", 3, first=100.0, step=10.0))
    out = performance.summary("SBER", "MAX", db_path=db,
                              today=datetime.date(2026, 1, 3))
    assert out["total_return"] == pytest.approx(0.20)
    assert (out["start_price"], out["end_price"]) == (100.0, 120.0)
    assert out["max_drawdown"] == 0.0, "a straight line never drew down"


def test_a_window_shorter_than_a_year_is_not_annualised(tmp_path):
    """The oldest number in the business: a month that made 4% did not make
    60% a year, and printing it as a rate is how a table becomes a pitch."""
    db = _db(tmp_path, _daily("2026-01-01", 40, step=0.1))
    out = performance.summary("SBER", "1M", db_path=db,
                              today=datetime.date(2026, 2, 9))
    assert out["total_return"] > 0
    assert out["annualised"] is None

    long_db = _db(tmp_path / "l", _daily("2024-01-01", 800, step=0.1))
    out = performance.summary("SBER", "MAX", db_path=long_db,
                              today=datetime.date(2026, 3, 11))
    assert out["annualised"] is not None


def test_volatility_is_annualised_on_the_instrument_s_own_calendar(tmp_path):
    """Moscow trades weekends now, so a Moscow name prints about 334 bars a
    year against a US name's 252. One constant for both understates the
    weekend trader by 13%."""
    weekday = [(d, c) for d, c in _daily("2025-01-01", 700, step=0.1)
               if datetime.date.fromisoformat(d).weekday() < 5]
    db_us = _db(tmp_path, weekday)
    us = performance.summary("SBER", "MAX", db_path=db_us,
                             today=datetime.date(2026, 12, 1))
    assert 250 <= us["bars_per_year"] <= 262

    every_day = _daily("2025-01-01", 700, step=0.1)
    db_ru = _db(tmp_path / "ru", every_day)
    ru = performance.summary("SBER", "MAX", db_path=db_ru,
                             today=datetime.date(2026, 12, 1))
    assert ru["bars_per_year"] == pytest.approx(365, abs=2)


def test_the_benchmark_is_compared_on_the_dates_both_series_have(tmp_path):
    """A Moscow name measured over days the index was shut reads as
    outperformance it never earned."""
    asset = _daily("2026-01-01", 60, first=100.0, step=1.0)
    # the index is missing every tenth day, and jumps on the days it has
    bench = [(d, c) for i, (d, c) in enumerate(_daily("2026-01-01", 60,
                                                      first=100.0, step=1.0))
             if i % 10]
    db = _db(tmp_path, asset, extra={"imoex": bench})
    out = performance.summary("SBER", "MAX", db_path=db, benchmark="IMOEX",
                              today=datetime.date(2026, 3, 2))
    cross = out["benchmark"]
    assert cross["shared_bars"] == len(bench)
    # both series move identically on the shared days, so the excess is zero
    assert cross["excess"] == pytest.approx(0.0, abs=1e-9)
    assert cross["beta"] == pytest.approx(1.0, abs=1e-6)


def test_a_rewound_window_cannot_see_past_its_own_date(tmp_path):
    """The dossier builds this block for old judgments during a backfill.
    Bounded only at the start, a judgment dated in May would be handed the
    year that followed it - the exact trap the rewind machinery exists for."""
    rows = _daily("2026-01-01", 200, first=100.0, step=1.0)
    db = _db(tmp_path, rows)
    early = performance.summary("SBER", "MAX", db_path=db,
                                today=datetime.date(2026, 2, 1))
    late = performance.summary("SBER", "MAX", db_path=db,
                               today=datetime.date(2026, 7, 1))
    assert early["end"] <= "2026-02-01"
    assert late["end"] > early["end"]
    assert early["end_price"] < late["end_price"]


def test_a_window_with_nothing_in_it_says_so_rather_than_returning_zeroes():
    out = performance.summary("NOSUCHASSET", "1Y", db_path="nosuch.db")
    assert out["error"]
    assert "total_return" not in out, "a zero here would read as a flat year"


def test_a_day_count_works_where_a_name_would_not(tmp_path):
    db = _db(tmp_path, _daily("2026-01-01", 200))
    out = performance.summary("SBER", "45", db_path=db,
                              today=datetime.date(2026, 6, 1))
    assert out["bars"] == 46
    with pytest.raises(ValueError):
        performance.parse_window("last tuesday")


def test_the_printed_table_says_what_is_not_in_the_numbers(tmp_path):
    """Price return without dividends is a defensible choice and a silent one
    is not, so the caveat travels with the table."""
    db = _db(tmp_path, _daily("2026-01-01", 200))
    text = "\n".join(performance.table("SBER", windows=("1M", "MAX"),
                                       db_path=db))
    assert "dividends are not in any of these numbers" in text
    assert "own bar count rather than on 252" in text
