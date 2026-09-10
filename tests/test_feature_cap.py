"""The feature cap, and the measurement that says the default is the worse one."""

import numpy as np
import pandas as pd

import train_hybrid as T


def test_zero_means_keep_every_feature(monkeypatch):
    monkeypatch.setenv("GTRADE_TOP_K_FEATURES", "0")
    assert T.top_k_features(12) is None


def test_the_default_is_untouched_without_the_variable(monkeypatch):
    """Changing it invalidates every champion on disk, so it stays opt-in."""
    monkeypatch.delenv("GTRADE_TOP_K_FEATURES", raising=False)
    assert T.top_k_features(12) == 12


def test_garbage_does_not_silently_change_the_feature_space(monkeypatch):
    monkeypatch.setenv("GTRADE_TOP_K_FEATURES", "twelve")
    assert T.top_k_features(12) == 12


def _frame(n=400):
    rng = np.random.default_rng(0)
    d = {c: rng.normal(size=n) for c in ("a", "b", "c", "d")}
    d["target"] = (rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(d)


def test_none_returns_the_candidates_untouched_and_trains_nothing():
    """The all-features path must not fit a ranking model just to discard it."""
    df = _frame()
    got = T.derive_feature_set(df, slice(0, 300), ["a", "b", "c", "d"], None)
    assert got == ["a", "b", "c", "d"]


def test_a_cap_still_selects_and_still_shortens():
    df = _frame()
    got = T.derive_feature_set(df, slice(0, 300), ["a", "b", "c", "d"], 2)
    assert len(got) == 2 and set(got) <= {"a", "b", "c", "d"}
