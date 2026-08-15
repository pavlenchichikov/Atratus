"""The sequence window must end ON the labelled bar.

`target[i]` is `close[i+1] > close[i]`, decided by information available at bar i.
CatBoost trains on X[i] -> target[i] and serving feeds `X_all[-lookback:]`, a
window ending on the latest bar. A training window that stops at i-1 hands the
nets a strictly harder problem than CatBoost solves AND a different one than
serving asks - which is what kept the neural members at coin-flip accuracy while
five separate experiments looked for the cause in the features.
"""

import numpy as np

from train_hybrid import build_sequences

LOOKBACK = 5


def _rows(n=30):
    """Row i carries the value i, so a window's contents name their own rows."""
    return np.arange(n, dtype="float64").reshape(n, 1), np.arange(n, dtype="float64")


def test_window_ends_on_the_labelled_row():
    X, y = _rows()
    X_seq, y_seq = build_sequences(X, y, LOOKBACK)
    for k in range(len(X_seq)):
        assert X_seq[k][-1][0] == y_seq[k], (
            "window %d ends on row %s but is labelled with row %s: the nets would "
            "have to predict a move starting from a bar they never see"
            % (k, X_seq[k][-1][0], y_seq[k]))


def test_window_covers_exactly_lookback_bars_back_to_back():
    X, y = _rows()
    X_seq, _ = build_sequences(X, y, LOOKBACK)
    first = X_seq[0].flatten()
    assert list(first) == [1, 2, 3, 4, 5], first
    assert X_seq.shape[1] == LOOKBACK


def test_index_mapping_is_unchanged():
    """The caller slices train/val/test out of the sequence array by counting
    rows, and y_mag / span come through this same function - so row k must keep
    corresponding to source row k+lookback."""
    X, y = _rows(40)
    X_seq, y_seq = build_sequences(X, y, LOOKBACK)
    assert len(X_seq) == len(X) - LOOKBACK
    assert list(y_seq) == list(range(LOOKBACK, len(X)))


def test_a_label_decided_by_the_last_bar_is_recoverable():
    """The property the whole thing exists for: when y[i] is a function of row i,
    the window must actually contain the row it needs. This fails on a window
    that stops at i-1 - that is the positive control, in one assert."""
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (200, 3))
    y = (X[:, 0] > 0).astype(float)
    X_seq, y_seq = build_sequences(X, y, LOOKBACK)
    decided_by = (X_seq[:, -1, 0] > 0).astype(float)
    assert np.array_equal(decided_by, y_seq)
