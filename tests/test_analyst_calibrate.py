"""core.analyst.calibrate: judgment cells, their priors, and the cold start."""

import pytest

from core.analyst import calibrate


def _table():
    """A payoff table shaped like train_payoff.build_table output."""
    return {"asset": {"SBER": {"BUY": {"n": 60, "mean": 0.30, "q10": -1.2, "q90": 1.8},
                               "SELL": {"n": 40, "mean": -0.10, "q10": -1.5, "q90": 1.4}}},
            "class": {"ru": {"BUY": {"n": 900, "mean": 0.10, "q10": -1.4, "q90": 1.6},
                             "SELL": {"n": 800, "mean": 0.05, "q10": -1.5, "q90": 1.5}}}}


def _judgment(direction="up", conviction=3, vol_regime="normal"):
    return {"direction": direction, "conviction": conviction,
            "vol_regime": vol_regime}


def test_with_an_empty_log_the_forecast_is_the_prior(tmp_path):
    # Day one. Every cell has zero observations, so the number has to come from
    # the empirical payoff table or it is invented.
    cells = calibrate.fit([], _table(), lambda a: "ru")
    out = calibrate.forecast(_judgment(), cells, "SBER", "ru",
                             atr_today=2.0, close_today=100.0,
                             payoff_table=_table())
    # SBER BUY mean 0.30 ATR shrunk toward class 0.10 with n=60, k=50:
    #   (60*0.30 + 50*0.10) / 110 = 0.2091 ATR, and ATR is 2% of close.
    assert out["pct"] == pytest.approx(0.2091 * 0.02, abs=1e-5)
    assert out["source"] == "prior"
    assert out["n"] == 0


def test_a_down_judgment_uses_the_sell_side_of_the_table():
    cells = calibrate.fit([], _table(), lambda a: "ru")
    out = calibrate.forecast(_judgment(direction="down"), cells, "SBER", "ru",
                             atr_today=2.0, close_today=100.0,
                             payoff_table=_table())
    # A short's payoff is stated for the short. The card shows the position's
    # return, so a profitable short is a positive number.
    assert out["pct"] < 0   # SBER SELL mean is negative, shrunk toward +0.05


def test_a_filled_cell_outweighs_its_prior():
    # Controller correction (2026-08-25): 500 rows was not enough evidence to
    # clear the 2e-3 tolerance against the 0.2091 ATR prior at k=50 -
    # (500*2.0 + 50*0.2091)/550 = 1.8372 ATR = 0.03674, which misses 0.04 by
    # 3.3e-3. 2000 rows clears it: (2000*2.0 + 50*0.2091)/2050 = 1.9563 ATR =
    # 0.03913, within 2e-3 of 0.04.
    rows = [{"direction": "up", "conviction": 5, "vol_regime": "normal",
             "asset": "SBER", "realized_atr_units": 2.0} for _ in range(2000)]
    cells = calibrate.fit(rows, _table(), lambda a: "ru")
    out = calibrate.forecast(_judgment(conviction=5), cells, "SBER", "ru",
                             atr_today=2.0, close_today=100.0,
                             payoff_table=_table())
    assert out["pct"] == pytest.approx(2.0 * 0.02, abs=2e-3)
    assert out["source"] == "measured"
    assert out["n"] == 2000


def test_conviction_levels_are_calibrated_separately():
    rows = ([{"direction": "up", "conviction": 1, "vol_regime": "normal",
              "asset": "SBER", "realized_atr_units": 0.0} for _ in range(500)]
            + [{"direction": "up", "conviction": 5, "vol_regime": "normal",
                "asset": "SBER", "realized_atr_units": 2.0} for _ in range(500)])
    cells = calibrate.fit(rows, _table(), lambda a: "ru")
    weak = calibrate.forecast(_judgment(conviction=1), cells, "SBER", "ru",
                              2.0, 100.0, _table())
    strong = calibrate.forecast(_judgment(conviction=5), cells, "SBER", "ru",
                                2.0, 100.0, _table())
    assert strong["pct"] > weak["pct"]


def test_an_anti_informative_conviction_is_reported_not_hidden():
    # If conviction 5 has historically LOST money, the calibration says so and
    # the card prints a negative number. core/calibration.py already documents
    # this exact inversion in the ensemble's own probabilities, so it is a
    # live possibility here and not a hypothetical.
    rows = [{"direction": "up", "conviction": 5, "vol_regime": "normal",
             "asset": "SBER", "realized_atr_units": -1.5} for _ in range(500)]
    cells = calibrate.fit(rows, _table(), lambda a: "ru")
    out = calibrate.forecast(_judgment(conviction=5), cells, "SBER", "ru",
                             2.0, 100.0, _table())
    assert out["pct"] < 0


def test_the_interval_brackets_the_point():
    cells = calibrate.fit([], _table(), lambda a: "ru")
    out = calibrate.forecast(_judgment(), cells, "SBER", "ru", 2.0, 100.0, _table())
    assert out["lo"] <= out["pct"] <= out["hi"]


def test_an_unknown_asset_falls_back_to_its_class():
    cells = calibrate.fit([], _table(), lambda a: "ru")
    out = calibrate.forecast(_judgment(), cells, "NOSUCH", "ru", 2.0, 100.0, _table())
    assert out["pct"] == pytest.approx(0.10 * 0.02, abs=1e-6)


def test_a_flat_judgment_forecasts_no_move():
    cells = calibrate.fit([], _table(), lambda a: "ru")
    out = calibrate.forecast(_judgment(direction="flat"), cells, "SBER", "ru",
                             2.0, 100.0, _table())
    assert out["pct"] == pytest.approx(0.0)
    assert out["lo"] < 0 < out["hi"]


def _rows(n, conviction=5, realized=2.0):
    return [{"direction": "up", "conviction": conviction, "vol_regime": "normal",
             "asset": "SBER", "realized_atr_units": realized} for _ in range(n)]


def test_source_is_prior_at_zero_blended_below_min_measured_at_min():
    # The label's own boundary is MIN_CELL_OWN, same threshold that decides
    # whether the interval comes from the cell or the prior. n=1 and n=99 are
    # both overwhelmingly prior-driven at k=50 but carry some evidence, hence
    # "blended" rather than "prior" or "measured".
    cells0 = calibrate.fit([], _table(), lambda a: "ru")
    out0 = calibrate.forecast(_judgment(conviction=5), cells0, "SBER", "ru",
                              2.0, 100.0, _table())
    assert out0["source"] == "prior"
    assert out0["n"] == 0

    cells1 = calibrate.fit(_rows(1), _table(), lambda a: "ru")
    out1 = calibrate.forecast(_judgment(conviction=5), cells1, "SBER", "ru",
                              2.0, 100.0, _table())
    assert out1["source"] == "blended"
    assert out1["n"] == 1

    cells99 = calibrate.fit(_rows(99), _table(), lambda a: "ru")
    out99 = calibrate.forecast(_judgment(conviction=5), cells99, "SBER", "ru",
                               2.0, 100.0, _table())
    assert out99["source"] == "blended"
    assert out99["n"] == 99

    cells100 = calibrate.fit(_rows(100), _table(), lambda a: "ru")
    out100 = calibrate.forecast(_judgment(conviction=5), cells100, "SBER", "ru",
                                2.0, 100.0, _table())
    assert out100["source"] == "measured"
    assert out100["n"] == 100


def test_a_flat_judgments_interval_is_pinned_to_the_buy_side():
    # Deliberate: a flat call claims the raw return is small, and the BUY
    # side's quantiles ARE the raw return's distribution in ATR units (BUY
    # payoff = +ret / (atr/close)); SELL's are that return's negation, the
    # short's payoff, and would be the wrong band for a no-move claim.
    # Asset-level n is put at MIN_CELL_OWN on both sides so _prior reads the
    # asset's own (visibly different) quantiles rather than falling back to
    # class, and a regression that reads SELL instead of BUY is caught here
    # rather than eyeballed.
    table = {"asset": {"SBER": {
                 "BUY": {"n": 150, "mean": 0.30, "q10": -1.2, "q90": 1.8},
                 "SELL": {"n": 150, "mean": -0.10, "q10": -1.5, "q90": 1.4}}},
             "class": {"ru": {
                 "BUY": {"n": 900, "mean": 0.10, "q10": -1.4, "q90": 1.6},
                 "SELL": {"n": 800, "mean": 0.05, "q10": -1.5, "q90": 1.5}}}}
    cells = calibrate.fit([], table, lambda a: "ru")
    out = calibrate.forecast(_judgment(direction="flat"), cells, "SBER", "ru",
                             2.0, 100.0, table)
    assert out["lo"] == pytest.approx(-1.2 * 0.02, abs=1e-9)
    assert out["hi"] == pytest.approx(1.8 * 0.02, abs=1e-9)


def test_conviction_does_not_move_the_number_until_a_cell_has_outcomes():
    # Surprising, and true by design: with no scored judgment in a cell, every
    # conviction falls back to the same prior, so 1/5 and 5/5 produce an
    # identical figure. Pinned because the asset card shows the conviction
    # right beside that number, and a reader - or a later contributor - would
    # otherwise assume the one drives the other. If this test ever fails,
    # someone has made conviction synthesise a number it has not earned.
    cells = calibrate.fit([], _table(), lambda a: "ru")
    out = [calibrate.forecast(_judgment(conviction=c), cells, "SBER", "ru",
                              2.0, 100.0, _table())["pct"]
           for c in (1, 2, 3, 4, 5)]
    assert len(set(out)) == 1, "conviction moved the number with no evidence"


def test_an_up_call_can_carry_a_negative_number_and_that_is_not_a_bug():
    # The figure is what the DIRECTION has historically been worth, not what
    # the analyst asserted. On a class whose longs lost money it is negative
    # under a bullish call, and the card has to be able to say so.
    table = {"asset": {}, "class": {"ru": {
        "BUY": {"n": 700, "mean": -0.23, "q10": -1.3, "q90": 0.8},
        "SELL": {"n": 800, "mean": 0.19, "q10": -0.8, "q90": 1.2}}}}
    cells = calibrate.fit([], table, lambda a: "ru")
    up = calibrate.forecast(_judgment(direction="up", conviction=5), cells,
                            "SBER", "ru", 2.0, 100.0, table)
    assert up["pct"] < 0
    assert up["source"] == "prior"


def _drifting_table():
    """The RU class, exaggerated: the real drift is -0.10 ATR and the real
    excess a fifth of the numbers here. Rounded up so the assertions below read
    as arithmetic rather than as noise."""
    return {"asset": {},
            "class": {"ru": {"BUY": {"n": 730, "mean": -0.23, "drift": -0.30,
                                     "drift_n": 3755, "excess": 0.07,
                                     "q10": -1.28, "q90": 0.78},
                             "SELL": {"n": 839, "mean": 0.20, "drift": 0.30,
                                      "drift_n": 3755, "excess": -0.10,
                                      "q10": -0.79, "q90": 1.21}}}}


def test_the_prior_is_net_of_the_market_it_was_measured_in():
    """A negative raw payoff on a falling market is not a verdict on the call.

    Before this, every RU long inherited the class's raw -0.23 ATR and the card
    stamped "[the payoff table disagrees]" on all of them - a flag that was
    reporting the direction of the Russian market, not the analyst."""
    out = calibrate.forecast(_judgment(), {}, "SBER", "ru", atr_today=2.0,
                             close_today=100.0, payoff_table=_drifting_table())
    assert out["pct"] == pytest.approx(0.07 * 0.02, abs=1e-6)
    assert out["pct"] > 0, "the long survives once the market is taken out"


def test_the_short_loses_the_credit_the_falling_market_gave_it():
    # The mirror of the test above, and the reason the drift is signed: +0.20
    # raw was less than the -(-0.30) a short got for free.
    out = calibrate.forecast(_judgment(direction="down"), {}, "SBER", "ru",
                             atr_today=2.0, close_today=100.0,
                             payoff_table=_drifting_table())
    assert out["pct"] == pytest.approx(-0.10 * 0.02, abs=1e-6)


def test_the_band_moves_with_the_point_estimate():
    """Shifting the mean and leaving the quantiles would let the forecast sit
    outside its own 80% band."""
    out = calibrate.forecast(_judgment(), {}, "SBER", "ru", atr_today=2.0,
                             close_today=100.0, payoff_table=_drifting_table())
    assert out["lo"] == pytest.approx((-1.28 + 0.30) * 0.02, abs=1e-6)
    assert out["hi"] == pytest.approx((0.78 + 0.30) * 0.02, abs=1e-6)
    assert out["lo"] <= out["pct"] <= out["hi"]


def test_a_table_written_before_drift_existed_still_reads():
    """payoff_stats.json is gitignored and the one on disk predates the change,
    so a missing `drift` key has to mean "no adjustment", not a KeyError."""
    out = calibrate.forecast(_judgment(), {}, "SBER", "ru", atr_today=2.0,
                             close_today=100.0, payoff_table=_table())
    assert out["pct"] == pytest.approx(0.2091 * 0.02, abs=1e-5)
