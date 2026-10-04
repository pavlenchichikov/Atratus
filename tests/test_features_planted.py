"""The RS positive control's planted feature, and the big_move label."""
import numpy as np
import pandas as pd

from core import features


def test_the_planted_leak_agrees_with_the_label_about_60_percent():
    df = pd.DataFrame({"target": np.random.default_rng(0).integers(0, 2, 5000).astype(float)})
    out = features.add_planted_leak(df.copy(), 0.4, seed=7)
    agree = (out["planted_leak"] == out["target"]).mean()
    assert 0.57 < agree < 0.63
    again = features.add_planted_leak(df.copy(), 0.4, seed=7)
    assert (again["planted_leak"] == out["planted_leak"]).all(), "seeded"


def test_without_the_variable_there_is_no_leak(monkeypatch):
    monkeypatch.delenv("GTRADE_PLANTED_LEAK", raising=False)
    assert "planted_leak" not in features.active_candidate_features()


def test_big_move_marks_the_day_before_a_wider_than_usual_bar():
    n = 120
    close = pd.Series(np.full(n, 100.0))
    high = pd.Series(np.full(n, 101.0))
    low = pd.Series(np.full(n, 99.0))
    high[100] = 105.0                        # bar 100 is a wide day
    t = features.make_target(close, "big_move", high=high, low=low)
    assert t[99] == 1.0, "the day BEFORE the wide bar is labelled"
    assert t[98] == 0.0
    assert np.isnan(t[10]) and np.isnan(t[n - 1])
    assert features.label_footprint() == 1
