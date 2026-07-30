"""Which assets may form a FRESH holdout, and a balanced suggestion.

Freshness is the only thing that makes an A/B number honest, so the rules here
are exclusions rather than taste: an asset the search screened on, or that the
research loop gates on, or that a previous A/B already measured, has been seen.

Pure on purpose. The caller reads the static sets from auto_research and the bar
counts from market.db once and passes them in, so this logic tests in
milliseconds and never touches a database.
"""

import random

MIN_BARS = 2000   # below this an asset is too thin to carry a measurement
# Below MIN_N a one-sided Wilcoxon reaches significance on a handful of mildly
# positive deltas, so a pass would mean little.
MIN_N = 8
DEFAULT_N = 14    # what the previous A/B used


def _as_set(value):
    """Accept a comma string or a list; the sources use both."""
    if isinstance(value, str):
        return {a.strip() for a in value.split(",") if a.strip()}
    return {str(a).strip() for a in (value or []) if str(a).strip()}


def excluded(static_sets, previous_holdouts):
    """Every asset that a fresh holdout must avoid.

    static_sets: SELECTION_ASSETS, HELDOUT_ASSETS, tier_assets().
    previous_holdouts: the holdout of every earlier A/B run. Reuse erodes a
    holdout - a re-gate on an already-used one once compared a result with
    itself and produced dScore 0.00, p=1.000.
    """
    out = set()
    for source in list(static_sets or []) + list(previous_holdouts or []):
        out |= _as_set(source)
    return out


def eligible(all_assets, bar_counts, excluded_set, min_bars=MIN_BARS):
    """Assets that may be drawn: not excluded, and with enough history.

    A missing bar count reads as too thin rather than as unknown: guessing would
    put an asset with no data into a measurement.
    """
    return [a for a in all_assets
            if a not in (excluded_set or set())
            and bar_counts.get(a, 0) >= min_bars]


def _first_group_of(asset, groups):
    for name, members in groups.items():
        if asset in members:
            return name
    return "OTHER"


def suggest(eligible_assets, groups, n=DEFAULT_N, seed=0):
    """A sample spread across asset classes, stable for a seed.

    Each asset counts under the FIRST group that claims it, because the groups
    overlap and counting an asset twice would overstate that class. Drawing is
    round-robin over the groups so a holdout cannot come out as all crypto,
    which would measure one market regime and call it the whole asset list.
    """
    rng = random.Random(seed)
    buckets = {}
    for asset in eligible_assets:
        buckets.setdefault(_first_group_of(asset, groups), []).append(asset)
    for members in buckets.values():
        rng.shuffle(members)
    order = sorted(buckets)
    rng.shuffle(order)
    out = []
    while len(out) < n and any(buckets[g] for g in order):
        for g in order:
            if buckets[g] and len(out) < n:
                out.append(buckets[g].pop())
    return sorted(out)


def validate(chosen, eligible_assets, min_n=MIN_N):
    """Problems with a holdout, in plain words. Empty means it is usable."""
    problems = []
    chosen = list(chosen or [])
    if len(chosen) != len(set(chosen)):
        dupes = sorted({a for a in chosen if chosen.count(a) > 1})
        problems.append("duplicate assets: %s" % ", ".join(dupes))
    allowed = set(eligible_assets or [])
    bad = sorted(set(chosen) - allowed)
    if bad:
        problems.append(
            "not eligible (already seen, or too little history): %s"
            % ", ".join(bad))
    if len(set(chosen)) < min_n:
        problems.append(
            "only %d assets; below %d the test cannot reach significance"
            % (len(set(chosen)), min_n))
    return problems
