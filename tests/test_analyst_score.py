"""core.analyst.score: coverage, error, and the shuffle control.

The control is the point of this file. A scorer that cannot be made to fail is
not measuring the analyst, it is measuring the asset's volatility.
"""

import random
import sqlite3

import pytest

from core.analyst import score, store
from core.analyst.payoff import ret_atr
from core.analyst.score import coverage, mae_atr, shuffle_control, standings


def _rows(n=200, skill=1.0, seed=7):
    """n rows where the forecast tracks the outcome with strength `skill`.

    skill=1.0 is a perfect forecaster, skill=0.0 is noise uncorrelated with the
    outcome. The interval is a fixed plus or minus 1.5 ATR around the forecast.
    """
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        realized = rnd.gauss(0.0, 1.0)
        forecast = skill * realized + (1 - skill) * rnd.gauss(0.0, 1.0)
        out.append({"realized_atr_units": realized, "forecast_atr": forecast,
                    "inside_interval": 1 if abs(realized - forecast) <= 1.5 else 0})
    return out


def test_coverage_counts_the_marked_rows():
    rows = [{"inside_interval": 1}, {"inside_interval": 0}, {"inside_interval": 1}]
    assert coverage(rows) == {"n": 3, "inside": 2, "rate": pytest.approx(2 / 3)}


def test_coverage_of_nothing_is_none_not_zero():
    # Zero would read as "the interval never contained the outcome", which is a
    # measurement. No rows is the absence of one.
    assert coverage([])["rate"] is None


def test_coverage_ignores_unscored_rows():
    rows = [{"inside_interval": 1}, {"inside_interval": None}]
    assert coverage(rows)["n"] == 1


def test_mae_is_zero_for_a_perfect_forecaster():
    assert mae_atr(_rows(skill=1.0)) == pytest.approx(0.0, abs=1e-9)


def test_mae_is_worse_for_noise_than_for_skill():
    assert mae_atr(_rows(skill=0.0)) > mae_atr(_rows(skill=0.8))


def test_mae_of_nothing_is_none():
    assert mae_atr([]) is None


def test_the_shuffle_control_destroys_a_real_advantage():
    # A skilled forecaster's error must rise sharply when its forecasts are
    # detached from the outcomes they were made for.
    rows = _rows(skill=0.9)
    result = shuffle_control(rows, seed=3)
    assert result["mae"] == pytest.approx(mae_atr(rows))
    assert result["mae_shuffled"] > result["mae"] * 1.5
    assert result["survives_shuffle"] is False


def test_the_shuffle_control_flags_a_scorer_that_measures_nothing():
    # Forecasts uncorrelated with outcomes score the same shuffled or not. That
    # is the failure this control exists to catch, and it is reported as a
    # survival rather than as a pass.
    rows = _rows(skill=0.0)
    result = shuffle_control(rows, seed=3)
    assert result["survives_shuffle"] is True


def test_a_perfect_forecaster_does_not_read_as_an_unmeasurable_one():
    # base MAE is exactly 0. Shuffling must break it, so the control has to
    # report survives_shuffle False. Before this was pinned, the `base > 0`
    # guard skipped the comparison and left the default True, which by this
    # module's own contract means "the measurement showed nothing" — the
    # opposite of the truth for the strongest signal there is.
    rows = _rows(skill=1.0)
    result = shuffle_control(rows, seed=1)
    assert result["mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["mae_shuffled"] > 0
    assert result["survives_shuffle"] is False


def _aligned(n=300, seed=11):
    rnd = random.Random(seed)
    rows, zero, emp, ens = [], [], [], []
    for i in range(n):
        realized = rnd.gauss(0.0, 1.0)
        rows.append({"realized_atr_units": realized,
                     "forecast_atr": 0.7 * realized + rnd.gauss(0, 0.3),
                     "inside_interval": 1,
                     "agent_direction": "up" if realized > 0 else "down",
                     "ensemble_direction": "up" if i % 2 else "down"})
        zero.append(0.0)
        emp.append(0.05)
        ens.append(0.05 if i % 2 else -0.05)
    return rows, {"zero": zero, "empirical": emp, "ensemble": ens}


def test_standings_reports_the_agent_against_every_baseline():
    rows, base = _aligned()
    s = standings(rows, base)
    assert set(s["baselines"]) == {"zero", "empirical", "ensemble"}
    assert s["agent"]["mae"] < s["baselines"]["zero"]["mae"]
    assert s["agent"]["beats"] == ["zero", "empirical", "ensemble"]


def test_standings_breaks_out_the_disagreement_subset():
    # Baselines 2 and 3 are identical wherever the agent and the ensemble
    # agree. The disagreement rows are the only place a second opinion can
    # earn its cost, so they get their own line instead of dissolving into
    # the average.
    rows, base = _aligned()
    s = standings(rows, base)
    assert 0 < s["disagreement"]["n"] < len(rows)
    assert s["disagreement"]["agent_mae"] is not None


def test_standings_carries_the_shuffle_control():
    rows, base = _aligned()
    assert standings(rows, base)["control"]["survives_shuffle"] is False


def test_a_worthless_agent_beats_nothing():
    rows, base = _aligned()
    for r in rows:
        r["forecast_atr"] = 5.0        # constant, wildly wrong
    assert standings(rows, base)["agent"]["beats"] == []


# --- The forecast_atr blind spot -------------------------------------------
#
# core.analyst.store.scored_rows() rows carry forecast_pct, atr_at_signal and
# close_at_signal - never a "forecast_atr" key. Every function above reads
# "forecast_atr". A test that hand-builds a row with "forecast_atr" already
# present would never see that gap, so this one goes through the real
# write path and reads the row back the way cmd_score does.

def _db_with_one_scored_judgment(tmp_path):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE sber (Date TEXT, Open REAL, Close REAL, '
                'High REAL, Low REAL)')
    rows = [(f"2026-01-{i + 1:02d}", 100.0, 100.0, 101.0, 99.0) for i in range(20)]
    rows.append(("2026-01-21", 100.0, 102.0, 102.5, 99.5))
    con.executemany('INSERT INTO sber VALUES (?,?,?,?,?)', rows)
    con.commit()
    con.close()
    store.ensure_table(path)
    store.write_judgment({
        "date": "2026-01-20", "asset": "SBER", "horizon": 1,
        "direction": "up", "conviction": 3, "vol_regime": "normal",
        "key_risk": "none", "thesis": "flat tape", "evidence_json": "[]",
        "dossier_hash": "abc", "llm_model": "test",
        "forecast_pct": 0.01, "lo_pct": -0.01, "hi_pct": 0.03,
        "atr_at_signal": 2.0, "close_at_signal": 100.0,
    }, db_path=path)
    store.backfill_outcomes(db_path=path, today="2026-01-22")
    return path


def test_scored_rows_from_the_store_have_no_forecast_atr_key_before_derivation(tmp_path):
    db = _db_with_one_scored_judgment(tmp_path)
    rows = store.scored_rows(db_path=db)
    assert len(rows) == 1
    assert "forecast_atr" not in rows[0]
    # This is the bug the controller flagged: every score.py function reads
    # "forecast_atr", so scoring the store's own rows unmodified reports None
    # rather than an error - a silent hole, not a loud one.
    assert mae_atr(rows) is None


def test_deriving_forecast_atr_from_the_stored_columns_fixes_the_mae(tmp_path):
    db = _db_with_one_scored_judgment(tmp_path)
    rows = store.scored_rows(db_path=db)
    for r in rows:
        r["forecast_atr"] = ret_atr(r["forecast_pct"], r["atr_at_signal"],
                                     r["close_at_signal"])
    # forecast 0.01 at 2.0 ATR / 100.0 close is 0.5 ATR; realized is 1.0 ATR
    # (see test_analyst_store.py:test_backfill_fills_the_next_bar_outcome).
    assert rows[0]["forecast_atr"] == pytest.approx(0.5)
    assert mae_atr(rows) == pytest.approx(0.5)


def _row(direction, conviction, realized, forecast=1.0):
    return {"direction": direction, "conviction": conviction,
            "realized_ret": realized, "forecast_atr": forecast,
            "realized_atr_units": realized}


class TestConvictionCalibration:
    def test_it_refuses_to_call_a_small_sample_informative(self):
        """The live log's own numbers: rho -0.125 at p=0.543 over 26 rows. The
        table looks monotone and means nothing, and reporting that as an effect
        is exactly how a belief gets manufactured."""
        rows = ([_row("up", 2, 1.0)] * 3 + [_row("up", 2, -1.0)] * 2
                + [_row("up", 3, 1.0)] * 8 + [_row("up", 3, -1.0)] * 6
                + [_row("up", 4, 1.0)] * 3 + [_row("up", 4, -1.0)] * 4)
        c = score.conviction_calibration(rows)
        assert c["n"] == 26
        assert c["by_conviction"][4]["rate"] < c["by_conviction"][2]["rate"]
        assert c["informative"] == "unknown", c

    def test_it_names_a_real_inversion_when_there_is_one(self):
        """The positive control. A monitor that always says "unknown" would
        pass the test above on its own."""
        rows = ([_row("up", 1, 1.0)] * 30 + [_row("up", 5, -1.0)] * 30)
        c = score.conviction_calibration(rows)
        assert c["informative"] == "INVERTED" and c["rho"] < 0

        rows = ([_row("up", 1, -1.0)] * 30 + [_row("up", 5, 1.0)] * 30)
        assert score.conviction_calibration(rows)["informative"] == "yes"

    def test_flat_calls_are_not_scored_as_directions(self):
        """A flat call is a claim about the SIZE of a move; scoring it as a
        direction needs a band this function has no business picking."""
        assert score.conviction_calibration(
            [_row("flat", 3, 0.001)] * 10)["n"] == 0


class TestPayoffAgreement:
    def test_it_splits_on_the_sign_of_the_forecast(self):
        """Direction is the model's, the forecast is the empirical payoff of
        that side. Ten of the first 33 judgments disagreed with themselves."""
        rows = [_row("up", 3, 1.0, forecast=+0.5),
                _row("up", 3, 1.0, forecast=+0.5),
                _row("down", 3, -1.0, forecast=-0.5)]
        a = score.payoff_agreement(rows)
        assert a["n"] == 3 and a["agree"] == 2
        assert a["agree_rate"] == pytest.approx(2 / 3)
        assert a["agree_mae"] is not None and a["disagree_mae"] is not None

    def test_a_row_without_a_forecast_is_not_counted_either_way(self):
        rows = [_row("up", 3, 1.0, forecast=None), _row("up", 3, 1.0, 0.5)]
        assert score.payoff_agreement(rows)["n"] == 1


class TestFieldUsage:
    @staticmethod
    def _rows(field, hits_with, n_with, hits_without, n_without):
        out = []
        for i in range(n_with):
            out.append({"direction": "up", "realized_ret": 1.0 if i < hits_with
                        else -1.0, "evidence_json": '["%s"]' % field})
        for i in range(n_without):
            out.append({"direction": "up",
                        "realized_ret": 1.0 if i < hits_without else -1.0,
                        "evidence_json": '["other"]'})
        return out

    def test_a_thinly_cited_field_is_not_given_a_verdict(self):
        """The live log's shape: gap_open cited once, hit rate 1.00. A table
        that prints that as an effect manufactures knowledge out of one row."""
        u = score.field_usage(self._rows("gap_open", 1, 1, 13, 25))
        assert u["fields"]["gap_open"]["verdict"] == "thin"
        assert u["fields"]["gap_open"]["hit_with"] == 1.0
        assert u["measurable"] == 0

    def test_a_field_with_both_sides_populated_gets_tested(self):
        """The positive control. Without it, a function that always answers
        "thin" would pass the test above."""
        u = score.field_usage(self._rows("ret_20", 28, 30, 6, 30))
        e = u["fields"]["ret_20"]
        assert e["verdict"] == "helps" and e["p"] < 0.05

        u = score.field_usage(self._rows("ret_20", 6, 30, 28, 30))
        assert u["fields"]["ret_20"]["verdict"] == "hurts"

        u = score.field_usage(self._rows("ret_20", 15, 30, 15, 30))
        assert u["fields"]["ret_20"]["verdict"] == "no effect"

    def test_unreadable_evidence_does_not_stop_the_report(self):
        rows = [{"direction": "up", "realized_ret": 1.0,
                 "evidence_json": "not json"}]
        assert score.field_usage(rows)["n"] == 1
