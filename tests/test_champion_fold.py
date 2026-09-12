"""Which fold becomes the champion. The trade-count floor says whether a fold's
trading Score can be trusted; it must not leave an asset on a champion fitted on
features that no longer exist, which is what force-promote is for."""

import train_hybrid as T


def _fold(k, score, trades):
    return {"fold": k, "score": score, "test_trades": trades, "test_profit": 0.0}


# The marker score_strategy returns for a fold with too few trades to judge.
# It reads -999.0 and not -299.7 since 2026-09-12: the fold score used to be
# multiplied by adv_weight (which floors at 0.3) AFTER the sentinel was
# assigned, so the marker reached the registry as -999 * 0.3, and this file's
# own constant was a copy of that artefact rather than of the contract.
UNRELIABLE = -999.0


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


def test_the_unreliable_sentinel_is_never_scaled():
    """adv_weight penalises a fold whose train and test distributions diverge.
    Applying it to the "too few trades to judge" marker turned -999 into
    -299.7, a value that reads like a real, merely terrible, score."""
    assert T.weighted_score(T.UNRELIABLE_SCORE, 0.3) == T.UNRELIABLE_SCORE
    assert T.weighted_score(T.UNRELIABLE_SCORE, 1.0) == T.UNRELIABLE_SCORE


def test_a_real_score_is_still_penalised():
    """The weight must keep working on everything that is not the sentinel,
    or the fix would quietly switch adversarial validation off."""
    assert T.weighted_score(4.0, 0.5) == 2.0
    assert T.weighted_score(-4.0, 0.5) == -2.0
    assert T.weighted_score(4.0, 1.0) == 4.0


def test_the_scaled_sentinel_would_have_passed_the_guard():
    """The positive control for the bug: show the OLD value defeating the very
    clause meant to catch it, so a regression cannot pass unnoticed.

    With the trade floor at its production value the fold is still refused, by
    the trade count rather than by the sentinel clause - which is why champion
    selection survived the bug while every mean over Score did not."""
    scaled = T.UNRELIABLE_SCORE * 0.3
    assert scaled > -999, "the scaled sentinel is what slipped past every guard"
    folds = [_fold(1, scaled, 3)]
    assert T.choose_champion_fold(folds, fold_floor=10, force=False) == (None, [])
    promoted, valid = T.choose_champion_fold(folds, fold_floor=1, force=False)
    assert promoted["fold"] == 1 and len(valid) == 1, (
        "a missing measurement was admitted as a champion once the floor let it by")
