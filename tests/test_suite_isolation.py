"""The suite must not read the developer's own adopted state.

Every one of these held before too - by accident, on a machine that had never
fitted anything. On 2026-08-21 nine tests across four files failed locally and
passed in CI, because config.py applies .env and adopted_genome.json to
os.environ at IMPORT time (during collection, before any fixture exists) and
core.levels reads levels_policy.json on every call. None of those three files is
in git, so CI never had them and the suite there was measuring the shipped
defaults while the same code measured a fitted policy locally.

These tests fail on any machine if the isolating fixtures in conftest.py are
removed, which is the point: an assertion that only fires on one developer's
laptop is how this went unnoticed in the first place.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_repo_root(path):
    return os.path.dirname(os.path.abspath(path)) == REPO_ROOT


def test_fitted_levels_policy_is_not_read_from_the_repo():
    from core import levels

    assert not _in_repo_root(levels.POLICY_PATH), (
        "core.levels.POLICY_PATH still points into the checkout, so a fitted "
        "levels_policy.json would decide what these tests measure")
    assert levels.load_policy() is None
    assert levels.policy_evidence() is None


def test_levels_under_test_use_the_shipped_multipliers():
    """What the four zone/stop tests silently assume, said out loud."""
    from core.levels import K_ENTRY, K_STOP, levels

    bars = [{"date": "d%d" % i, "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.0} for i in range(30)]
    row = levels(bars, "BUY")
    assert row["status"] == "ok"
    assert abs(row["atr"] - 2.0) < 1e-9
    assert abs(row["entry_low"] - (100.0 - K_ENTRY * 2.0)) < 1e-9
    assert abs(row["stop"] - (100.0 - K_STOP * 2.0)) < 1e-9


def test_the_adopted_genome_is_not_in_force():
    from core import adopted
    from core.feature_dsl import load_dsl_specs

    assert not _in_repo_root(adopted.PATH)
    assert adopted.load() is None
    # the fallback path inside load_dsl_specs, which no env var covers
    assert load_dsl_specs() == []
    for key in adopted.env_overrides({"drops": ["x"], "label_mode": "rel_median"}):
        assert os.getenv(key) is None, "%s survived into the test session" % key


def test_local_env_flags_do_not_reach_the_code_under_test():
    """.env is a developer's local configuration; CI has no such file."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for key in dotenv_values(os.path.join(REPO_ROOT, ".env")):
        if key.startswith("GTRADE_"):
            assert os.getenv(key) is None, (
                "%s came from .env and changes what the code under test does" % key)
