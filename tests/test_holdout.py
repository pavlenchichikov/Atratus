"""Unit tests for core.holdout (pure; no database, no network)."""

from core import holdout

GROUPS = {
    "CRYPTO": ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE"],
    "US TECH": ["NVDA", "AAPL", "MSFT", "AMZN"],
    "FOREX": ["EURUSD", "GBPUSD", "USDJPY"],
    "TOP SIGNALS": ["BTC", "NVDA"],          # deliberate overlap
}
ALL = sorted({a for v in GROUPS.values() for a in v})
BARS = dict.fromkeys(ALL, 3000)


def test_excluded_is_the_union_of_every_source():
    got = holdout.excluded(["BTC,ETH", "NVDA"], [["EURUSD", "AAPL"]])
    assert got == {"BTC", "ETH", "NVDA", "EURUSD", "AAPL"}


def test_excluded_accepts_both_strings_and_lists():
    # SELECTION_ASSETS and HELDOUT_ASSETS are comma strings; tier_assets and a
    # previous run's holdout arrive as lists.
    assert holdout.excluded(["A,B"], [["C"]]) == {"A", "B", "C"}
    assert holdout.excluded([["A", "B"]], ["C,D"]) == {"A", "B", "C", "D"}


def test_eligible_drops_the_excluded_and_the_thin():
    bars = dict(BARS, SOL=100)
    got = holdout.eligible(ALL, bars, {"BTC", "ETH"})
    assert "BTC" not in got and "ETH" not in got
    assert "SOL" not in got, "an asset with 100 bars cannot carry a measurement"
    assert "NVDA" in got


def test_eligible_treats_a_missing_bar_count_as_thin():
    got = holdout.eligible(["AAA"], {}, set())
    assert got == []


def test_suggest_returns_the_requested_size():
    assert len(holdout.suggest(ALL, GROUPS, n=6, seed=1)) == 6


def test_suggest_is_deterministic_for_a_seed():
    a = holdout.suggest(ALL, GROUPS, n=6, seed=7)
    b = holdout.suggest(ALL, GROUPS, n=6, seed=7)
    assert a == b
    assert holdout.suggest(ALL, GROUPS, n=6, seed=8) != a


def test_suggest_spreads_across_groups():
    # The failure this prevents: a holdout that is all crypto measures one market
    # regime and calls it the whole asset list.
    got = holdout.suggest(ALL, GROUPS, n=6, seed=3)
    hit = {g for g, members in GROUPS.items() if any(a in got for a in members)}
    assert len(hit) >= 3


def test_an_asset_in_two_groups_is_counted_once():
    # BTC is in CRYPTO and TOP SIGNALS. Counting it twice would overstate crypto.
    got = holdout.suggest(ALL, GROUPS, n=len(ALL), seed=0)
    assert len(got) == len(set(got)) == len(ALL)


def test_suggest_never_returns_an_ineligible_asset():
    elig = holdout.eligible(ALL, BARS, {"BTC", "NVDA"})
    got = holdout.suggest(elig, GROUPS, n=5, seed=2)
    assert "BTC" not in got and "NVDA" not in got


def test_validate_accepts_a_good_holdout():
    assert holdout.validate(ALL[:10], ALL) == []


def test_validate_names_an_ineligible_asset():
    problems = holdout.validate(["BTC", "NVDA", "AAPL", "MSFT", "AMZN",
                                 "EURUSD", "GBPUSD", "USDJPY", "SOL"],
                                [a for a in ALL if a != "BTC"])
    assert any("BTC" in p for p in problems)


def test_validate_rejects_an_underpowered_size():
    # Genome B was once rejected on a six-asset sign test that could not reach
    # significance at all, whichever way the deltas fell.
    problems = holdout.validate(ALL[:5], ALL)
    assert any("8" in p for p in problems)


def test_validate_rejects_a_duplicate():
    problems = holdout.validate(["AAPL"] * 9, ALL)
    assert any("duplicate" in p.lower() for p in problems)
