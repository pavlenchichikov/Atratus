"""Training must be able to stop short of today, or nothing can be measured.

On 2026-08-22, after a retrain of 165 assets, 163 of 317 champions had ZERO bars
dated after their own updated_at: there was no ground left in the book on which
a champion had not been fitted. The same day's measurement put the reconstructed
champion at 66.8% over its own history and 49.1% on days it had not seen. A
blanket retrain buys accuracy on paper by spending the ability to check it.
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import train_hybrid as th


def _frame(n=100):
    return pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=n),
                         "close": range(n)})


def test_off_by_default_leaves_the_frame_untouched():
    """The control. Every champion in the book was trained without this."""
    df = _frame()
    out = th.apply_train_embargo(df, "AAA", bars=0)
    assert out is df
    assert th._TRAIN_EMBARGO_BARS == 0, "the default must stay off"


def test_it_removes_exactly_the_last_n_bars():
    df = _frame(100)
    out = th.apply_train_embargo(df, "AAA", bars=30)
    assert len(out) == 70
    assert out["Date"].iloc[-1] == df["Date"].iloc[69]
    assert df["Date"].iloc[70] not in set(out["Date"]), "bar 71 must be unseen"


def test_a_frame_no_longer_than_the_embargo_is_kept_whole():
    """Cutting it to nothing would hide the asset instead of reporting it."""
    df = _frame(20)
    assert len(th.apply_train_embargo(df, "AAA", bars=20)) == 20
    assert len(th.apply_train_embargo(df, "AAA", bars=50)) == 20


def test_it_says_what_it_did():
    said = []
    th.apply_train_embargo(_frame(100), "AAA", bars=10, say=said.append)
    assert said and "EMBARGO" in said[0] and "10 bars" in said[0]


def test_the_environment_is_what_turns_it_on(monkeypatch):
    monkeypatch.setenv("GTRADE_TRAIN_EMBARGO_BARS", "45")
    assert th._env_int("GTRADE_TRAIN_EMBARGO_BARS", 0) == 45


def test_the_fold_training_window_is_recorded_as_a_date():
    """Without it nothing can tell which bars a champion had already seen."""
    df = _frame(100)
    assert th._fold_train_end(df, slice(0, 60)) == "2020-02-29"
    assert th._fold_train_end(df, None) is None
