"""Unit tests for core.news_link (pure; no network)."""

import pytest

from core import news_link


def bars(closes, start_day=1):
    return [{"asset": "X", "date": f"2026-07-{start_day + i:02d}", "close": c}
            for i, c in enumerate(closes)]


def item(published, weighted=0.5):
    return {"title": "t", "published": published, "weighted_score": weighted}


def test_daily_move_is_the_last_two_closes():
    assert news_link.daily_move(bars([100.0, 110.0])) == pytest.approx(0.1)


def test_daily_move_needs_two_usable_closes():
    assert news_link.daily_move(bars([100.0])) is None
    assert news_link.daily_move([]) is None
    assert news_link.daily_move(bars([0.0, 110.0])) is None


def test_sigma_needs_enough_returns():
    assert news_link.move_sigma(bars([100.0] * 10)) is None
    assert news_link.move_sigma(bars([100.0] * 40)) == 0.0


def test_notability_is_volatility_relative():
    # The same 3 percent move is notable for a calm asset and not for a wild one.
    assert news_link.is_notable(0.03, 0.01) is True
    assert news_link.is_notable(0.03, 0.05) is False


def test_notability_refuses_missing_or_zero_sigma():
    assert news_link.is_notable(0.03, None) is False
    assert news_link.is_notable(None, 0.01) is False
    assert news_link.is_notable(0.03, 0.0) is False


def test_same_day_matches_a_parsed_rss_timestamp():
    items = [item("Fri, 24 Jul 2026 10:00:00 GMT"),
             item("Sat, 25 Jul 2026 10:00:00 GMT")]
    assert len(news_link.same_day(items, "2026-07-24")) == 1


def test_same_day_excludes_an_unparseable_timestamp():
    # Guessing "today" would park unrelated news beside a move.
    assert news_link.same_day([item("last tuesday"), item("")],
                              "2026-07-24") == []


def test_mean_sentiment_ignores_missing_scores():
    items = [item("", weighted=0.2), item("", weighted=0.4),
             {"title": "t", "published": ""}]
    assert news_link.mean_sentiment(items) == pytest.approx(0.3)
    assert news_link.mean_sentiment([]) is None


def test_consistency_labels():
    assert news_link.consistency(-0.04, -0.5) == "consistent"
    assert news_link.consistency(-0.04, 0.5) == "conflicting"
    assert news_link.consistency(0.04, 0.5) == "consistent"


def test_consistency_is_unclear_when_measured_but_flat():
    assert news_link.consistency(-0.04, 0.02) == "unclear"
    assert news_link.consistency(0.0, 0.5) == "unclear"
    assert news_link.consistency(None, 0.5) == "unclear"


def test_absent_sentiment_is_its_own_answer():
    # "no news that day" is a data gap, not a measured neutral. Reporting it as
    # "unclear" would dress an absence up as a finding.
    assert news_link.consistency(-0.04, None) == "no_news"


def test_context_row_uses_the_last_bar_date_not_today():
    # A stale asset must be matched against the day it last moved, not today.
    # 25 closes keeps every generated date real (July has 31 days) and still
    # leaves the 20 returns move_sigma needs.
    stale = bars([100.0] * 24 + [130.0], start_day=1)
    date = stale[-1]["date"]
    assert date == "2026-07-25"
    # Weekday name omitted on purpose: RFC 2822 allows it, and inventing one
    # that disagrees with the date is a trap for the next reader.
    items = [item("25 Jul 2026 09:00:00 GMT", weighted=0.6)]
    row = news_link.context_row("X", stale, items)
    assert row["date"] == date
    assert row["notable"] is True
    assert row["consistency"] == "consistent"


def test_context_row_is_none_without_bars():
    assert news_link.context_row("X", [], []) is None
