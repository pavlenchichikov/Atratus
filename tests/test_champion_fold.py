import os

"""Which fold becomes the champion.

Two bases. Under `score` the trade-count floor says whether a fold's trading
Score can be trusted, and the best admitted fold wins; that is what every
champion before 2026-09-12 was, and the tests for it now name the basis
explicitly rather than relying on the default. Under `acc` (the default since
then) the floor plays no part, because a trade count says nothing about whether
a model predicts the next bar, and the LATEST measurable fold wins rather than
the most accurate one.

Either way the rule must not leave an asset on a champion fitted on features
that no longer exist, which is what force-promote is for.
"""

import train_hybrid as T


def _fold(k, score, trades, acc=None):
    return {"fold": k, "score": score, "test_trades": trades, "test_profit": 0.0,
            "ens_acc": acc}


# The marker score_strategy returns for a fold with too few trades to judge.
# It reads -999.0 and not -299.7 since 2026-09-12: the fold score used to be
# multiplied by adv_weight (which floors at 0.3) AFTER the sentinel was
# assigned, so the marker reached the registry as -999 * 0.3, and this file's
# own constant was a copy of that artefact rather than of the contract.
UNRELIABLE = -999.0


def test_the_best_admitted_fold_wins_as_before():
    folds = [_fold(1, 2.0, 30), _fold(2, 5.0, 40), _fold(3, UNRELIABLE, 3)]
    best, valid = T.choose_champion_fold(folds, fold_floor=10, force=False,
                                         basis="score")
    assert best["fold"] == 2 and [f["fold"] for f in valid] == [1, 2]


def test_no_admitted_fold_without_force_keeps_the_old_champion():
    folds = [_fold(1, UNRELIABLE, 4), _fold(2, UNRELIABLE, 6)]
    assert T.choose_champion_fold(folds, fold_floor=10, force=False,
                                  basis="score") == (None, [])


def test_no_admitted_fold_under_force_promotes_the_latest_fold():
    """After a feature change the old champion reads inputs that no longer
    exist; the newest fold is fitted on the freshest data."""
    folds = [_fold(1, UNRELIABLE, 4), _fold(2, UNRELIABLE, 9), _fold(3, UNRELIABLE, 2)]
    best, valid = T.choose_champion_fold(folds, fold_floor=10, force=True,
                                         basis="score")
    assert best["fold"] == 3 and valid == []


def test_force_does_not_override_an_admitted_fold():
    folds = [_fold(1, 1.0, 12), _fold(2, UNRELIABLE, 2)]
    best, _ = T.choose_champion_fold(folds, fold_floor=10, force=True,
                                     basis="score")
    assert best["fold"] == 1


def test_the_champion_is_a_copy():
    """Writing the median into the champion must not rewrite that fold's own
    entry in fold_metrics (Fold_Scores and the journal read it)."""
    folds = [_fold(1, 2.0, 30)]
    best, _ = T.choose_champion_fold(folds, fold_floor=10, force=False,
                                     basis="score")
    best["score"] = 99.0
    assert folds[0]["score"] == 2.0


def test_nothing_trained_gives_nothing():
    assert T.choose_champion_fold([], fold_floor=10, force=True,
                                  basis="score") == (None, [])
    assert T.choose_champion_fold([], fold_floor=10, force=True,
                                  basis="acc") == (None, [])


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
    assert T.choose_champion_fold(folds, fold_floor=10, force=False,
                                  basis="score") == (None, [])
    promoted, valid = T.choose_champion_fold(folds, fold_floor=1, force=False,
                                             basis="score")
    assert promoted["fold"] == 1 and len(valid) == 1, (
        "a missing measurement was admitted as a champion once the floor let it by")


# --- the accuracy basis ------------------------------------------------------

def test_accuracy_is_the_default_basis(monkeypatch):
    """The design decision of 2026-09-12: accuracy of the next-bar prediction
    decides the champion, and profitability is left to the backtest."""
    monkeypatch.delenv("GTRADE_CHAMPION_BASIS", raising=False)
    assert T.champion_basis() == "acc"
    monkeypatch.setenv("GTRADE_CHAMPION_BASIS", "score")
    assert T.champion_basis() == "score"
    monkeypatch.setenv("GTRADE_CHAMPION_BASIS", "bogus")
    assert T.champion_basis() == "acc", "an unknown basis must not silently trade"


def test_the_trade_floor_does_not_apply_to_accuracy():
    """A trade count says whether a Score can be trusted, never whether a model
    predicts the next bar. Under `score` this fold is refused and the asset ends
    up with no champion at all, which is what happened to 383 of them."""
    folds = [_fold(1, UNRELIABLE, 0, acc=0.55)]
    assert T.choose_champion_fold(folds, fold_floor=10, force=False,
                                  basis="score") == (None, [])
    best, valid = T.choose_champion_fold(folds, fold_floor=10, force=False,
                                         basis="acc")
    assert best["fold"] == 1 and len(valid) == 1


def test_the_latest_fold_wins_not_the_most_accurate_one():
    """The anti-argmax rule, and the reason for it: picking the best fold is a
    selected maximum. Measured 2026-09-12, the top quartile of assets by
    champion-fold accuracy read 0.6356 offline and 0.4820 live."""
    folds = [_fold(1, 1.0, 40, acc=0.71), _fold(2, 1.0, 40, acc=0.52),
             _fold(3, 1.0, 40, acc=0.54)]
    best, valid = T.choose_champion_fold(folds, fold_floor=10, force=False,
                                         basis="acc")
    assert best["fold"] == 3, "the champion must not be chosen by argmax"
    assert [f["fold"] for f in valid] == [1, 2, 3]


def test_a_fold_with_no_measurable_accuracy_is_not_admitted():
    """_safe_acc returns None on a window too short to measure. Admitting it
    would score a champion on nothing; force still promotes the latest fold so
    a changed feature chain cannot strand the asset on an old model."""
    folds = [_fold(1, 1.0, 40), _fold(2, 1.0, 40)]
    assert T.choose_champion_fold(folds, fold_floor=10, force=False,
                                  basis="acc") == (None, [])
    best, valid = T.choose_champion_fold(folds, fold_floor=10, force=True,
                                         basis="acc")
    assert best["fold"] == 2 and valid == []


def test_the_accuracy_champion_is_a_copy():
    folds = [_fold(1, 2.0, 30, acc=0.53)]
    best, _ = T.choose_champion_fold(folds, fold_floor=10, force=False,
                                     basis="acc")
    best["ens_acc"] = 0.99
    assert folds[0]["ens_acc"] == 0.53


def test_the_baseline_takes_the_better_of_majority_and_persistence():
    import numpy as np

    from train_hybrid import _baseline_acc
    y = np.array([1, 1, 0, 0] * 5, dtype=float)          # runs of two
    prev = np.concatenate([[np.nan], y[:-1]])
    # majority of an all-ones train set scores 0.5; persistence scores 10/19
    assert _baseline_acc(np.ones(20), y, prev, 1) == 10 / 19
    # a multi-bar label cannot see yesterday's label, so only the majority counts
    assert _baseline_acc(np.ones(20), y, prev, 5) == 0.5
    assert _baseline_acc(np.ones(20), y[:5], prev[:5], 1) is None


def test_the_label_signature_moves_with_every_label_setting(monkeypatch):
    for k in [k for k in os.environ if k.startswith("GTRADE_LABEL_")]:
        monkeypatch.delenv(k)
    assert T._label_sig() == ""
    monkeypatch.setenv("GTRADE_LABEL_MODE", "rel_median")
    monkeypatch.setenv("GTRADE_LABEL_WINDOW", "30")
    a = T._label_sig()
    monkeypatch.setenv("GTRADE_LABEL_WINDOW", "20")
    assert a == "GTRADE_LABEL_MODE=rel_median;GTRADE_LABEL_WINDOW=30" != T._label_sig()


def test_the_combiner_defaults_to_the_fixed_mix_and_can_be_set_back(monkeypatch):
    monkeypatch.delenv("GTRADE_COMBINER", raising=False)
    assert T._combiner() == "fixed"
    monkeypatch.setenv("GTRADE_COMBINER", "stack")
    assert T._combiner() == "stack"
    monkeypatch.setenv("GTRADE_COMBINER", "nonsense")
    assert T._combiner() == "fixed"


def test_dir_metrics_score_the_next_bars_direction_against_the_validation_majority():
    import numpy as np
    prob = np.array([0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4, 0.6, 0.6])
    ret = np.array([0.02, -0.001, -0.02, 0.001, 0.03, 0.02, -0.01, -0.02, -0.03, 0.002])
    val = np.array([0.01, 0.02, -0.01, 0.015, 0.02, -0.005, 0.01, 0.012, -0.02, 0.01])
    m = T._dir_metrics(prob, ret, val)
    # calls: up up dn dn up up dn dn up up ; outcomes: up dn dn up up up dn dn dn up
    assert m["dir_acc"] == 0.7 and m["dir_n"] == 10
    assert m["dir_base"] == 0.5                      # validation mostly up: "always up"
    # band = 0.5 x median |val| = 0.5 x 0.011; the +-0.001/0.002 rows are noise
    assert m["dir_n_clean"] == 7 and abs(m["dir_acc_clean"] - 6 / 7) < 1e-12
    assert T._dir_metrics(prob[:5], ret[:5], val[:5]) is None
