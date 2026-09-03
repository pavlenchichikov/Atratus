"""Unit tests for core.events (pure; the yfinance fetch is injected)."""

import datetime
import json

from core import events

TODAY = datetime.date(2026, 7, 26)


def test_days_until_counts_forward_and_backward():
    assert events.days_until("2026-07-29", TODAY) == 3
    assert events.days_until("2026-07-26", TODAY) == 0
    assert events.days_until("2026-07-20", TODAY) == -6


def test_a_single_date_reads_as_confirmed():
    def fetch(symbol, session):
        return {"Earnings Date": ["2026-08-05"]}

    got = events.earnings_for({"AAPL": "AAPL"}, fetch=fetch)
    assert got == {"AAPL": {"date": "2026-08-05", "confirmed": True}}


def test_a_date_range_reads_as_an_estimate():
    # yfinance reports an estimated date as a two-date range. Showing that as
    # fact would be a lie the user trades on.
    def fetch(symbol, session):
        return {"Earnings Date": ["2026-08-05", "2026-08-09"]}

    got = events.earnings_for({"AAPL": "AAPL"}, fetch=fetch)
    assert got["AAPL"]["confirmed"] is False
    assert got["AAPL"]["date"] == "2026-08-05"


def test_an_unparseable_bound_does_not_promote_a_range_to_confirmed():
    # The range is what makes it an estimate. Dropping a junk bound must not
    # turn the survivor into a confirmed date.
    def fetch(symbol, session):
        return {"Earnings Date": ["2026-08-05", "NaT"]}

    got = events.earnings_for({"AAPL": "AAPL"}, fetch=fetch)
    assert got["AAPL"]["date"] == "2026-08-05"
    assert got["AAPL"]["confirmed"] is False


def test_assets_without_earnings_are_absent():
    def fetch(symbol, session):
        return {"Earnings Date": []}

    assert events.earnings_for({"BTC": "BTC-USD"}, fetch=fetch) == {}


def test_one_failing_asset_does_not_sink_the_rest():
    def fetch(symbol, session):
        if symbol == "BAD":
            raise RuntimeError("yahoo said no")
        return {"Earnings Date": ["2026-08-05"]}

    got = events.earnings_for({"A": "BAD", "B": "GOOD"}, fetch=fetch)
    assert list(got) == ["B"]


def test_macro_loads_and_normalises(tmp_path):
    path = tmp_path / "macro.json"
    path.write_text(json.dumps([
        {"date": "2026-07-29", "name": "FOMC", "importance": "high"}]),
        encoding="utf-8")
    got = events.load_macro(str(path))
    # region is carried through so a consumer can filter on it; absent means
    # global, which is how every entry read before the field existed.
    assert got == [{"kind": "macro", "asset": None, "date": "2026-07-29",
                    "name": "FOMC", "importance": "high", "region": None,
                    "confirmed": True}]
    path.write_text(json.dumps([
        {"date": "2026-07-29", "name": "CBR", "region": " ru "}]),
        encoding="utf-8")
    assert events.load_macro(str(path))[0]["region"] == "RU"


def test_a_missing_macro_file_is_no_events(tmp_path):
    assert events.load_macro(str(tmp_path / "absent.json")) == []


def test_corrupt_macro_payloads_are_no_events(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert events.load_macro(str(bad)) == []
    obj = tmp_path / "obj.json"
    obj.write_text('{"date": "2026-07-29"}', encoding="utf-8")
    assert events.load_macro(str(obj)) == []


def test_macro_entries_without_a_date_or_name_are_dropped(tmp_path):
    path = tmp_path / "macro.json"
    path.write_text(json.dumps([
        {"date": "nonsense", "name": "FOMC"},
        {"date": "2026-07-29", "name": "  "},
        {"name": "no date"},
        "not a dict",
        {"date": "2026-07-30", "name": "CPI"}]), encoding="utf-8")
    assert [e["name"] for e in events.load_macro(str(path))] == ["CPI"]


def test_upcoming_keeps_the_forward_window_sorted():
    evs = [{"date": "2026-08-20", "name": "far"},
           {"date": "2026-07-28", "name": "soon"},
           {"date": "2026-07-20", "name": "past"},
           {"date": "2026-07-26", "name": "today"}]
    assert [e["name"] for e in events.upcoming(evs, TODAY)] == ["today", "soon"]


def test_upcoming_boundary_is_inclusive():
    evs = [{"date": "2026-08-09", "name": "edge"},
           {"date": "2026-08-10", "name": "over"}]
    assert [e["name"] for e in events.upcoming(evs, TODAY)] == ["edge"]


def test_event_id_is_stable_and_distinguishes_kinds():
    a = events.event_id("earnings", "AAPL", "2026-08-05", "Earnings")
    assert a == events.event_id("earnings", "AAPL", "2026-08-05", "Earnings")
    assert a != events.event_id("macro", None, "2026-08-05", "Earnings")
    assert len(a) == 16


def test_event_rows_merges_both_sources():
    earn = {"AAPL": {"date": "2026-08-05", "confirmed": False}}
    macro = [{"kind": "macro", "asset": None, "date": "2026-07-29",
              "name": "FOMC", "importance": "high", "confirmed": True}]
    rows = events.event_rows(earn, macro)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"earnings", "macro"}
    earn_row = next(r for r in rows if r["kind"] == "earnings")
    assert earn_row["asset"] == "AAPL"
    assert earn_row["confirmed"] is False
    assert earn_row["importance"] is None
    assert len({r["id"] for r in rows}) == 2


def test_a_moscow_listed_name_is_not_asked_for_earnings():
    # Its ticker is carried bare (VTBR, SBER), which Yahoo cannot resolve, so
    # asking bought one 404 per Russian asset and nothing else. The fundamentals
    # for these come from Smart-Lab on a different path and are unaffected.
    from core.events import can_have_earnings
    assert can_have_earnings("VTBR", "VTBR") is False
    assert can_have_earnings("SBER", "SBER") is False
    # A US name with the same bare shape must still be asked.
    assert can_have_earnings("AAPL", "AAPL") is True
    # And a foreign listing with a suffix.
    assert can_have_earnings("ASML.AS", "ASML") is True
    # Without the asset name the symbol-only rules still apply.
    assert can_have_earnings("^VIX") is False
    assert can_have_earnings("AAPL") is True


def test_the_scan_skips_moscow_names_without_calling_out():
    from core import events
    called = []

    def fetch(symbol, session=None):
        called.append(symbol)
        return {}

    events.earnings_for({"VTBR": "VTBR", "SBER": "SBER", "AAPL": "AAPL"},
                        fetch=fetch)
    assert called == ["AAPL"], "a Moscow-listed name was sent to the source"
