"""Hourly bars into exchange sessions. Synthetic bars only, no database."""
import pandas as pd

from core.intraday import session_table, session_zone, sessionize


def _bar(ts, o=1.0, h=1.0, l=1.0, c=1.0, v=1.0):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_the_session_is_the_exchange_local_date_not_the_utc_date():
    # 23:30 UTC on the 4th is 08:30 on the 5th in Tokyo.
    bars = [_bar("2026-09-04T23:30:00+00:00"), _bar("2026-09-05T00:30:00+00:00"),
            _bar("2026-09-05T01:30:00+00:00")]
    df = sessionize(bars, "Asia/Tokyo")
    assert set(df["session"]) == {"2026-09-05"}


def test_sessions_shorter_than_three_bars_are_dropped():
    bars = [_bar("2026-09-04T14:00:00+00:00"), _bar("2026-09-04T15:00:00+00:00"),
            _bar("2026-09-05T14:00:00+00:00"), _bar("2026-09-05T15:00:00+00:00"),
            _bar("2026-09-05T16:00:00+00:00")]
    df = sessionize(bars, "UTC")
    assert list(df["session"].unique()) == ["2026-09-05"]
    assert list(df["bar_idx"]) == [0, 1, 2]
    assert list(df["is_last"]) == [False, False, True]


def test_bars_are_sorted_and_deduplicated():
    bars = [_bar("2026-09-05T16:00:00+00:00"), _bar("2026-09-05T14:00:00+00:00"),
            _bar("2026-09-05T15:00:00+00:00"), _bar("2026-09-05T15:00:00+00:00")]
    df = sessionize(bars, "UTC")
    assert len(df) == 3 and df["ts"].is_monotonic_increasing


def test_round_the_clock_assets_use_the_utc_day():
    assert session_zone("BTC", "UTC") == "UTC"
    assert session_zone("EURUSD", "Europe/London") == "UTC"
    assert session_zone("AAPL", "America/New_York") == "America/New_York"


def test_session_table_aggregates_one_row_per_session():
    bars = [_bar("2026-09-05T14:00:00+00:00", 10, 12, 9, 11),
            _bar("2026-09-05T15:00:00+00:00", 11, 13, 10, 12),
            _bar("2026-09-05T16:00:00+00:00", 12, 12, 8, 9)]
    st = session_table(sessionize(bars, "UTC"))
    row = st.loc["2026-09-05"]
    assert (row["open"], row["high"], row["low"], row["close"], row["n"]) == (10, 13, 8, 9, 3)


def test_an_empty_input_gives_an_empty_frame_with_the_columns():
    df = sessionize([], "UTC")
    assert df.empty and {"session", "bar_idx", "is_last"} <= set(df.columns)
    assert isinstance(df, pd.DataFrame)
