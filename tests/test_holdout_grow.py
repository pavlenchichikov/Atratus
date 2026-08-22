"""Growing the gate, not redrawing it.

Two holdouts decide things here: the search's own, which picks what is even
offered for adoption, and the A/B's, which judges it. Both were fourteen
assets, resolving nothing smaller than +2.80 dScore - while the one genome ever
adopted measured +1.63. Sizing them is now a launcher question, so the pick has
to be deterministic, balanced, and above all additive: a replaced holdout makes
every earlier measurement incomparable.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import holdout as H

GROUPS = {
    "crypto": ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT"],
    "forex": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"],
    "us": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
    "eu": ["SAP", "ASML", "LVMH"],
}
ALL = [a for members in GROUPS.values() for a in members]


def test_every_asset_already_held_is_kept():
    """The property the whole design rests on: a superset stays comparable."""
    current = ["BTC", "EURUSD", "AAPL"]
    out = H.grow(current, ALL, GROUPS, 10)
    assert set(current) <= set(out)
    assert out[:3] == current, "and in the same order, so a diff reads cleanly"


def test_it_reaches_the_size_asked_for():
    out = H.grow(["BTC"], ALL, GROUPS, 12)
    assert len(out) == 12
    assert len(set(out)) == 12, "no asset twice"


def test_additions_fill_the_thinnest_classes_first():
    """A holdout that came out all crypto would measure one regime and call it
    the asset list."""
    out = H.grow(["BTC", "ETH", "SOL", "ADA"], ALL, GROUPS, 8)
    added = out[4:]
    assert "crypto" not in [g for g in GROUPS if set(added) <= set(GROUPS[g])]
    groups_hit = {g for a in added for g in GROUPS if a in GROUPS[g]}
    assert len(groups_hit) >= 3, "four additions spread over at least three classes"


def test_it_is_deterministic():
    a = H.grow(["BTC"], ALL, GROUPS, 10)
    b = H.grow(["BTC"], ALL, GROUPS, 10)
    assert a == b


def test_a_holdout_already_big_enough_is_returned_untouched():
    current = ["BTC", "ETH", "SOL"]
    assert H.grow(current, ALL, GROUPS, 2) == current
    assert H.grow(current, ALL, GROUPS, 3) == current


def test_it_stops_when_the_pool_runs_out_rather_than_repeating():
    out = H.grow(["BTC"], ["ETH", "SOL"], GROUPS, 99)
    assert sorted(out) == ["BTC", "ETH", "SOL"]


def test_an_asset_already_held_is_never_added_twice():
    out = H.grow(["BTC", "ETH"], ALL, GROUPS, 6)
    assert len(out) == len(set(out))


def test_the_size_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("GTRADE_AB_HOLDOUT_N", "60")
    assert H._env_n("GTRADE_AB_HOLDOUT_N", 40) == 60


def test_a_size_below_the_floor_is_refused_not_obeyed(monkeypatch):
    """A gate too small to resolve anything is worse than the default, and a
    typo should not quietly shrink it."""
    monkeypatch.setenv("GTRADE_AB_HOLDOUT_N", "2")
    assert H._env_n("GTRADE_AB_HOLDOUT_N", 40) == 40
    monkeypatch.setenv("GTRADE_AB_HOLDOUT_N", "not a number")
    assert H._env_n("GTRADE_AB_HOLDOUT_N", 40) == 40
