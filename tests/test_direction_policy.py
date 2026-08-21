"""The direction override: the widest authority, and what it is allowed to say.

Fitted and gated on LIVE rows on purpose. The reconstructed backtest and the
live stream disagreed about the sign of the relationship this rule conditions
on, so history is the wrong place to ask.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import train_direction as td
from core import direction_policy as dp


def _rows(n=10, prob=0.8, signal="BUY"):
    return [{"date": "2026-08-%02d" % (1 + i), "asset": "A",
             "signal": signal, "probability": prob, "actual_next_ret": 0.01}
            for i in range(n)]


def test_follow_changes_nothing():
    rows = _rows()
    assert [r["signal"] for r in dp.apply_direction(rows, "follow")] \
        == ["BUY"] * 10


def test_aside_silences_only_what_clears_the_threshold():
    rows = _rows(prob=0.8) + _rows(prob=0.52)
    out = dp.apply_direction(rows, "aside", thr=0.20)
    assert [r["signal"] for r in out[:10]] == ["WAIT"] * 10
    assert [r["signal"] for r in out[10:]] == ["BUY"] * 10


def test_invert_takes_the_other_side_of_the_confident_ones():
    out = dp.apply_direction(_rows(prob=0.9), "invert", thr=0.20)
    assert all(r["signal"] == "SELL" for r in out)


def test_a_row_with_no_probability_is_left_alone():
    """An override needs the confidence it is conditioned on. Inventing one
    would put the rule in charge of rows it cannot read."""
    rows = [{"date": "2026-08-01", "asset": "A", "signal": "BUY",
             "probability": None, "actual_next_ret": 0.01}]
    assert dp.apply_direction(rows, "invert", 0.0)[0]["signal"] == "BUY"


def test_the_incumbent_is_in_the_search_space():
    """`follow` has to be able to win, or the fit is guaranteed to find an
    override whether or not one exists."""
    assert "follow" in dp.MODES
    rows = _rows(prob=0.9)          # confident longs that all pay
    assert td.fit(rows, forex=())["mode"] == "follow"


def test_a_rule_that_silences_everything_is_named_as_one():
    """The gate can pass "stand aside on everything", and on the live window it
    did. That is not a clever override, it is an off switch, and the report has
    to say the word."""
    rows = _rows(prob=0.9)
    assert td.suppression(rows, {"thr": 0.20}) == 1.0
    assert td.suppression(rows, {"thr": 0.45}) == 0.0
    assert td.SILENCE_SHARE <= 1.0


def test_the_split_is_on_the_date_and_refuses_when_it_cannot_be():
    rows = _rows(n=12)
    early, late, cut = td.split_in_time(rows)
    assert early and late and cut
    assert all(r["date"] < cut for r in early)
    assert td.split_in_time(_rows(n=3)) is None
