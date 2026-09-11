"""Which fold becomes the champion. The trade-count floor says whether a fold's
trading Score can be trusted; it must not leave an asset on a champion fitted on
features that no longer exist, which is what force-promote is for."""

import train_hybrid as T


def _fold(k, score, trades):
    return {"fold": k, "score": score, "test_trades": trades, "test_profit": 0.0}


UNRELIABLE = -299.7


def test_the_best_admitted_fold_wins_as_before():
    folds = [_fold(1, 2.0, 30), _fold(2, 5.0, 40), _fold(3, UNRELIABLE, 3)]
    best, valid = T.choose_champion_fold(folds, fold_floor=10, force=False)
    assert best["fold"] == 2 and [f["fold"] for f in valid] == [1, 2]


def test_no_admitted_fold_without_force_keeps_the_old_champion():
    folds = [_fold(1, UNRELIABLE, 4), _fold(2, UNRELIABLE, 6)]
    assert T.choose_champion_fold(folds, fold_floor=10, force=False) == (None, [])


def test_no_admitted_fold_under_force_promotes_the_latest_fold():
    """After a feature change the old champion reads inputs that no longer
    exist; the newest fold is fitted on the freshest data."""
    folds = [_fold(1, UNRELIABLE, 4), _fold(2, UNRELIABLE, 9), _fold(3, UNRELIABLE, 2)]
    best, valid = T.choose_champion_fold(folds, fold_floor=10, force=True)
    assert best["fold"] == 3 and valid == []


def test_force_does_not_override_an_admitted_fold():
    folds = [_fold(1, 1.0, 12), _fold(2, UNRELIABLE, 2)]
    best, _ = T.choose_champion_fold(folds, fold_floor=10, force=True)
    assert best["fold"] == 1


def test_the_champion_is_a_copy():
    """Writing the median into the champion must not rewrite that fold's own
    entry in fold_metrics (Fold_Scores and the journal read it)."""
    folds = [_fold(1, 2.0, 30)]
    best, _ = T.choose_champion_fold(folds, fold_floor=10, force=False)
    best["score"] = 99.0
    assert folds[0]["score"] == 2.0


def test_nothing_trained_gives_nothing():
    assert T.choose_champion_fold([], fold_floor=10, force=True) == (None, [])
