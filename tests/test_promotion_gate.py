"""The gate reports the decision production will actually make.

Every other statistic in the adoption path reduces a holdout to a mean.
train_hybrid never sees that mean: it walks the assets one at a time and keeps
the champion unless the challenger beats it by 0.2. On 2026-08-18 an A/B passed
on a mean of +0.036 while the same rows carried 3 promotions against 10
demotions, and the ten-hour retrain it authorised then rediscovered that asset
by asset.
"""

import ab_build
import auto_research as ar


def _rows(**scores):
    return [{"Asset": a, "Score": s} for a, s in scores.items()]


def test_a_mean_carried_by_one_asset_is_not_a_promotion():
    base = _rows(A=5.0, B=5.0, C=5.0, D=5.0, E=5.0)
    var = _rows(A=25.0, B=4.0, C=4.0, D=4.0, E=4.0)
    st = ar.promotion_stats(base, var)
    assert (st["promoted"], st["demoted"]) == (1, 4)
    assert st["p"] > 0.5, "four losses against one win is not evidence of a gain"


def test_a_real_across_the_board_gain_reads_as_one():
    base = _rows(A=5.0, B=5.0, C=5.0, D=5.0, E=5.0)
    var = _rows(A=6.0, B=6.0, C=6.0, D=6.0, E=6.0)
    st = ar.promotion_stats(base, var)
    assert (st["promoted"], st["demoted"]) == (5, 0)
    assert st["p"] < 0.05


def test_assets_inside_the_margin_are_excluded_not_counted():
    """train_hybrid keeps the champion inside the margin, so those assets are
    not evidence either way and must not dilute the sign test."""
    base = _rows(A=5.0, B=5.0, C=5.0)
    var = _rows(A=5.1, B=4.9, C=6.0)
    st = ar.promotion_stats(base, var)
    assert (st["promoted"], st["demoted"], st["n"]) == (1, 0, 1)


def test_the_margin_is_the_one_train_hybrid_uses():
    assert ar.PROMOTION_MARGIN == 0.2


def test_an_asset_missing_from_one_side_is_skipped():
    st = ar.promotion_stats(_rows(A=5.0), _rows(A=6.0, B=9.9))
    assert st["n"] == 1


def test_nothing_comparable_is_not_a_win():
    st = ar.promotion_stats([], [])
    assert st == {"promoted": 0, "demoted": 0, "n": 0, "p": 1.0}
    assert ar.promotion_tag(st) == ""


def test_the_tag_states_both_sides():
    tag = ar.promotion_tag(ar.promotion_stats(_rows(A=5.0, B=5.0),
                                              _rows(A=6.0, B=4.0)))
    assert "promote 1" in tag and "demote 1" in tag


def test_the_ab_stats_dict_carries_the_counts_for_the_report():
    st = {"promoted": 3, "demoted": 10, "p_promotion": 0.9539}
    assert "promote 3" in ab_build.ar_promotion_tag(st)
    assert "demote 10" in ab_build.ar_promotion_tag(st)


def test_a_passing_mean_is_vetoed_when_it_would_demote_more_than_it_promotes():
    """The 2026-08-18 shape: significant on the basis, negative for production."""
    passing = {"p": 0.0067, "value": 0.036, "n": 14, "promoted": 3, "demoted": 10}
    assert ab_build.verdict(passing, floor=0.005, alpha=0.05) == "FAILED"


def test_the_veto_does_not_touch_a_candidate_that_wins_on_both():
    both = {"p": 0.0067, "value": 0.036, "n": 14, "promoted": 9, "demoted": 4}
    assert ab_build.verdict(both, floor=0.005, alpha=0.05) == "PASSED"


def test_an_arm_with_no_promotion_data_is_judged_as_before():
    """Older result files carry no counts; absent must not read as a veto."""
    old = {"p": 0.0067, "value": 0.036, "n": 14}
    assert ab_build.verdict(old, floor=0.005, alpha=0.05) == "PASSED"
