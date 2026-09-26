"""Empty Yahoo bars are reported and never let a table step over them.

2026-09-26 06:31: Yahoo listed 09-25 with close = None for SAP and every field
None for BTC. dropna() threw 185 such bars away silently ("+1 bars", "0
errors"), and a later run storing 09-26 would have left 09-25 a hole no daily
run asks for again - the 09-22 vendor day did exactly that.
"""
from datetime import datetime

import pandas as pd

import data_engine as de

NOW = datetime(2026, 9, 26, 6, 31)


def _bars(rows):
    return pd.DataFrame(
        [{"Date": pd.Timestamp(d), "Open": v, "Close": v, "High": v, "Low": v,
          "Volume": None if v is None else 1.0} for d, v in rows])


def test_clean_bars_pass_exactly_as_dropna_kept_them():
    """The control: no empty bar, nothing held, nothing reported."""
    df = _bars([("2026-09-24", 1.0), ("2026-09-25", 2.0)])
    out, gaps = de._hold_at_vendor_gap(df, NOW)
    assert gaps == [] and out.equals(df.dropna())


def test_a_recent_empty_bar_holds_the_table_before_it():
    """BTC's shape: an empty 09-25 between two good bars. Storing 09-26 would
    move the table past the hole for good."""
    df = _bars([("2026-09-24", 1.0), ("2026-09-25", None), ("2026-09-26", 3.0)])
    out, gaps = de._hold_at_vendor_gap(df, NOW)
    assert gaps == [pd.Timestamp("2026-09-25")]
    assert list(out["Date"]) == [pd.Timestamp("2026-09-24")]


def test_a_single_empty_field_is_a_gap_too():
    """SAP's shape: open and volume present, close None."""
    df = _bars([("2026-09-24", 1.0), ("2026-09-25", 2.0)])
    df.loc[1, "Close"] = None
    out, gaps = de._hold_at_vendor_gap(df, NOW)
    assert gaps == [pd.Timestamp("2026-09-25")] and len(out) == 1


def test_an_old_empty_bar_is_stored_past_not_held_forever():
    """Deep in a backfill Yahoo has empty days it will never fill; holding
    there would freeze the table, so the rows after it are kept."""
    df = _bars([("2026-08-01", None), ("2026-08-03", 2.0), ("2026-09-25", 3.0)])
    out, gaps = de._hold_at_vendor_gap(df, NOW)
    assert gaps == [pd.Timestamp("2026-08-01")]
    assert list(out["Date"]) == [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-09-25")]


def test_assets_behind_the_common_last_bar_are_named_from_the_database():
    """SAP's case: Yahoo never showed the empty bar, so only the database can
    say Friday is missing. Assets ahead of the common date (a Saturday forex
    stamp) are not behind anything."""
    last = {"SPY": "2026-09-25", "MSFT": "2026-09-25", "DAX": "2026-09-25",
            "SAP": "2026-09-24", "BTC": "2026-09-24", "KOSPI": "2026-09-23",
            "EURUSD": "2026-09-26", "NEW": None}
    text = "\n".join(de.behind_report(last))
    assert "Most assets end at 2026-09-25; 3 end earlier" in text
    assert "2026-09-24     2  BTC, SAP" in text
    assert "2026-09-23     1  KOSPI" in text
    assert "EURUSD" not in text
    assert de.behind_report({"A": "2026-09-25", "B": "2026-09-25"}) == []


def test_the_report_says_which_dates_which_assets_and_what_to_run():
    gaps = {"BTC": [pd.Timestamp("2026-09-25")], "SAP": [pd.Timestamp("2026-09-25")],
            "OLD": [pd.Timestamp("2026-08-01")]}
    text = "\n".join(de.vendor_gap_report(gaps, NOW))
    assert "3 asset(s)" in text
    assert "2026-09-25     2  BTC, SAP" in text
    assert "2 held BEFORE the gap" in text and "run data_engine again later" in text
    assert "1 stored past an older gap" in text and "GTRADE_BACKFILL=1" in text
    assert de.vendor_gap_report({}, NOW) == []
