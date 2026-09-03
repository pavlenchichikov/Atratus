"""core/macro.py: the calendar fetched instead of typed.

The parsers are fed saved page fragments rather than the live sites, so a
layout change fails here loudly instead of quietly returning nothing on a run.
"""
from core import macro

CBR_HTML = """
<div class="main-events_day">
  <div class="date col-md-5">11&nbsp;сентября 2026 года</div>
  <div class="info"><span>Заседание Совета директоров Банка России по ключевой ставке</span></div>
</div>
<div class="main-events_day">
  <div class="date col-md-5">23&nbsp;сентября 2026 года</div>
  <div class="info"><span>Резюме обсуждения ключевой ставки</span></div>
</div>
<div class="main-events_day">
  <div class="date col-md-5">1&nbsp;октября 2026 года</div>
  <div class="info"><span>Отчёт о чём-то ещё</span></div>
</div>
"""

FOMC_HTML = """
<h4>2026 FOMC Meetings</h4>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month"><strong>January</strong></div>
  <div class="fomc-meeting__date">27-28</div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month"><strong>April</strong></div>
  <div class="fomc-meeting__date">28-29</div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month"><strong>December</strong></div>
  <div class="fomc-meeting__date">31-1</div>
</div>
<h4>2027 FOMC Meetings</h4>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month"><strong>March</strong></div>
  <div class="fomc-meeting__date">16-17</div>
</div>
"""


def test_a_rate_decision_and_its_minutes_are_not_the_same_event():
    """A decision moves the rate; a `Резюме обсуждения` explains one taken six
    weeks ago. Folded together, an analyst reads minutes as a rate risk."""
    got = macro.cbr_events(CBR_HTML)
    assert [(e["date"], e["importance"]) for e in got] == [
        ("2026-09-11", "high"), ("2026-09-23", "medium")]
    assert "decision" in got[0]["name"] and "minutes" in got[1]["name"]
    # and a calendar row that is neither is not an event
    assert all("2026-10-01" != e["date"] for e in got)


def test_every_meeting_on_the_page_is_read_not_one_in_three():
    """Splitting the page into blocks looked tidier and kept 27 of 57
    meetings: a chunk holding two rows was searched once."""
    got = macro.fomc_events(FOMC_HTML)
    assert len(got) == 4, "one per row, across both year sections"
    assert got[0]["date"] == "2026-01-28"
    assert got[-1]["date"] == "2027-03-17", "the second section's year is used"


def test_a_two_day_meeting_is_dated_on_the_day_it_decides():
    """The statement lands on the second day. Filed under the first, the event
    sits a day early - inside a one-day horizon that is the whole event."""
    got = macro.fomc_events(FOMC_HTML)
    assert got[0]["date"] == "2026-01-28", "not the 27th"
    # and a meeting across a month boundary rolls forward, including a year end
    assert got[2]["date"] == "2027-01-01"


def test_a_dead_source_contributes_nothing_rather_than_a_guess(monkeypatch):
    def boom(url):
        raise OSError("connection reset")

    monkeypatch.setattr(macro, "_get", boom)
    events, failed = macro.fetch(sources=(("CBR", "url", macro.cbr_events),))
    assert events == []
    assert "connection reset" in failed["CBR"]


def test_a_reachable_page_that_parses_to_nothing_is_reported_as_a_failure(
        monkeypatch):
    """Silence from a layout change looks exactly like a quiet week, and a
    calendar that is quietly empty is worse than one that is missing."""
    monkeypatch.setattr(macro, "_get", lambda url: "<html>redesigned</html>")
    events, failed = macro.fetch(sources=(("CBR", "url", macro.cbr_events),))
    assert events == [] and "layout" in failed["CBR"]

    # positive control: the same plumbing DOES pass a page it can read
    monkeypatch.setattr(macro, "_get", lambda url: CBR_HTML)
    events, failed = macro.fetch(sources=(("CBR", "url", macro.cbr_events),))
    assert len(events) == 2 and failed == {}


def test_a_hand_written_event_survives_a_fetch():
    """The file was designed to be maintained by hand and may carry an OPEC
    meeting or an election that neither central bank publishes."""
    existing = [{"date": "2026-10-01", "name": "OPEC+ meeting",
                 "importance": "high", "region": ""}]
    fetched = [{"date": "2026-09-11", "name": "CBR key rate decision",
                "importance": "high", "region": "RU"}]
    merged = macro.merge(existing, fetched)
    assert [e["name"] for e in merged] == ["CBR key rate decision",
                                           "OPEC+ meeting"]
    # and re-running does not duplicate what it already wrote
    assert len(macro.merge(merged, fetched)) == 2


def test_the_file_round_trips(tmp_path):
    path = str(tmp_path / "macro_calendar.json")
    events = macro.cbr_events(CBR_HTML)
    assert macro.save(events, path=path) == len(events)
    assert macro.load(path=path) == events
    assert macro.load(path=str(tmp_path / "absent.json")) == []
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert macro.load(path=str(tmp_path / "bad.json")) == []


def test_upcoming_is_bounded_at_both_ends():
    events = [{"date": "2020-01-01", "name": "old"},
              {"date": "2026-09-11", "name": "soon"},
              {"date": "2030-01-01", "name": "far"}]
    got = macro.upcoming(events, today="2026-09-03", days=30)
    assert [e["name"] for e in got] == ["soon"]
