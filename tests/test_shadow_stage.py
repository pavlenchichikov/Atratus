"""Watching a challenger must not hand it the wheel.

One setting used to do two jobs: `GTRADE_TIMING_STAGE=b` was the only way to
get the Q into the live log, and it also made the Q the SERVED decision - the
card's badge, the timing column, the side the levels are drawn and fitted on.
So collecting a comparison silently replaced the policy being compared against.
`shadow` serves the rules and records the Q beside them, on the same bars.
"""
import sqlite3

import pytest

from core import policy_report as pr
from core import timing_fqi as fq


class TestTheFlag:
    def test_serving_and_watching_are_different_settings(self, monkeypatch):
        cases = {
            "": (False, False),
            "a": (False, False),
            "b": (True, False),
            "B": (True, False),
            " b ": (True, False),
            "shadow": (False, True),
            "SHADOW": (False, True),
            "1": (False, False),
        }
        for value, (serve, watch) in cases.items():
            monkeypatch.setenv("GTRADE_TIMING_STAGE", value)
            assert fq.stage_b_on() is serve, value
            assert fq.stage_b_shadow_on() is watch, value

    def test_neither_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("GTRADE_TIMING_STAGE", raising=False)
        assert fq.stage_b_on() is False
        assert fq.stage_b_shadow_on() is False

    def test_the_two_are_never_both_on(self, monkeypatch):
        """The pair has to be exclusive, or 'served' stops having one answer."""
        for value in ("", "a", "b", "shadow", "junk"):
            monkeypatch.setenv("GTRADE_TIMING_STAGE", value)
            assert not (fq.stage_b_on() and fq.stage_b_shadow_on()), value


class TestTheChallengerKeepsItsOwnPosition:
    """A watched policy enters and exits on different bars from the served one,
    so rebuilding it from the served column would hand it a position it never
    took, and every comparison after that is between two fictions."""

    def _db(self, tmp_path, rows):
        import performance_tracker as pt
        db = tmp_path / "p.db"
        con = sqlite3.connect(db)
        cur = con.cursor()
        pt._ensure_table(cur)
        cur.executemany(
            "INSERT INTO prediction_log (date, asset, signal, probability,"
            " actual_next_ret, timing_action, shadow_action)"
            " VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
        return str(db)

    def test_each_column_rebuilds_its_own_history(self, tmp_path, monkeypatch):
        import performance_tracker as pt
        # Served went long and is still in. Watched went long, then flattened.
        rows = [
            ("2026-08-01", "SP500", "BUY", 0.6, 0.01, "ENTER:+1", "ENTER:+1"),
            ("2026-08-02", "SP500", "BUY", 0.6, 0.01, "HOLD", "EXIT"),
        ]
        monkeypatch.setattr(pt, "DB_PATH", self._db(tmp_path, rows))
        assert pt.timing_state("SP500")["pos"] == 1
        assert pt.timing_state("SP500", column="shadow_action")["pos"] == 0

    def test_an_unknown_column_is_refused_not_interpolated(self):
        import performance_tracker as pt
        with pytest.raises(ValueError):
            pt.timing_state("SP500", column="signal")


class TestTheReport:
    def _row(self, date, action, shadow, ret):
        return {"asset": "SP500", "date": date, "signal": "BUY",
                "probability": 0.6, "actual_next_ret": ret,
                "timing_action": action, "timing_stage": "A",
                "shadow_action": shadow}

    def test_the_watched_arm_is_scored_from_its_own_column(self):
        # The rules hold a loser all week; the Q left after one bar.
        rows = [self._row("2026-08-%02d" % d,
                          "ENTER:+1" if d == 1 else "HOLD",
                          "ENTER:+1" if d == 1 else ("EXIT" if d == 2 else "STAY_OUT"),
                          -0.01)
                for d in range(1, 8)]
        out = pr.reconcile(rows)
        assert out["timing A"]["rows"] == 7
        assert out["timing B (watched)"]["rows"] == 7
        assert out["timing B (watched)"]["profit"] > out["timing A"]["profit"], (
            "leaving early beat holding a loser, and the arms must show it")

    def test_no_shadow_column_means_no_watched_arm(self):
        rows = [{"asset": "SP500", "date": "2026-08-%02d" % d, "signal": "BUY",
                 "probability": 0.6, "actual_next_ret": 0.01,
                 "timing_action": "HOLD", "timing_stage": "A"}
                for d in range(1, 8)]
        out = pr.reconcile(rows)
        assert out["timing B (watched)"]["status"] == "no data"

    def test_the_watched_arm_never_feeds_the_served_one(self):
        """Both are present on the same bars; A's number must not move."""
        with_shadow = [self._row("2026-08-%02d" % d, "HOLD", "STAY_OUT", 0.01)
                       for d in range(1, 8)]
        without = [dict(r, shadow_action=None) for r in with_shadow]
        assert (pr.reconcile(with_shadow)["timing A"]["profit"]
                == pr.reconcile(without)["timing A"]["profit"])


def test_an_old_database_gains_the_column_without_losing_its_rows(tmp_path):
    """The column arrives by migration, so a log written before today keeps
    every row it had and simply reads NULL for the challenger."""
    import performance_tracker as pt
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE prediction_log (
        date TEXT, asset TEXT, signal TEXT, probability REAL,
        actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL,
        model_version TEXT, meta_prob REAL, sig_shown TEXT, gate_reason TEXT,
        timing_action TEXT, timing_reason TEXT, timing_stage TEXT)""")
    con.execute("INSERT INTO prediction_log (date, asset, signal, timing_action)"
                " VALUES ('2026-08-01','SP500','BUY','HOLD')")
    con.commit()
    cur = con.cursor()
    pt._ensure_table(cur)
    con.commit()
    cols = {r[1] for r in con.execute("PRAGMA table_info(prediction_log)")}
    assert "shadow_action" in cols
    row = con.execute("SELECT timing_action, shadow_action FROM prediction_log"
                      " WHERE asset='SP500'").fetchone()
    con.close()
    assert row == ("HOLD", None)


class TestSeeingItInTheUi:
    """A watched policy nobody can see is not being watched."""

    def test_the_wording_says_would_not_did(self):
        from core import timing_policy as tp
        assert tp.watched_label("ENTER:+1") == "watched Q: would enter long"
        assert tp.watched_label("ENTER:-1") == "watched Q: would enter short"
        assert tp.watched_label("EXIT") == "watched Q: would exit"
        assert tp.watched_label("STAY_OUT") == "watched Q: would stay out"
        assert tp.watched_label(None) is None
        for label in ("ENTER:+1", "EXIT", "STAY_OUT", "HOLD"):
            assert "would" in tp.watched_label(label), label

    def test_the_column_reaches_the_card(self, tmp_path):
        import performance_tracker as pt
        from core import track_record as tr
        db = tmp_path / "t.db"
        con = sqlite3.connect(db)
        cur = con.cursor()
        pt._ensure_table(cur)
        cur.execute("INSERT INTO prediction_log (date, asset, signal, sig_shown,"
                    " timing_action, timing_reason, shadow_action)"
                    " VALUES ('2026-08-01','SP500','BUY','BUY','HOLD','ok','EXIT')")
        con.commit()
        con.close()
        out = tr.latest_gated("SP500", db_path=str(db))
        assert out["timing_action"] == "HOLD"
        assert out["shadow_action"] == "EXIT"

    def test_a_database_without_the_column_still_renders(self, tmp_path):
        from core import track_record as tr
        db = tmp_path / "old.db"
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE prediction_log (
            date TEXT, asset TEXT, signal TEXT, sig_shown TEXT,
            gate_reason TEXT, timing_action TEXT, timing_reason TEXT)""")
        con.execute("INSERT INTO prediction_log (date, asset, signal, sig_shown)"
                    " VALUES ('2026-08-01','SP500','BUY','BUY')")
        con.commit()
        con.close()
        out = tr.latest_gated("SP500", db_path=str(db))
        assert out["signal"] == "BUY"
        assert out["shadow_action"] is None


class TestCheckingItAgainstFact:
    """The point of watching: not just what Q would have done, but whether it
    would have been right, on the same terms the signal's own row is judged."""

    def _rows(self, spec):
        """spec: (signal, shadow_action, actual_next_ret) oldest first."""
        return [{"asset": "SP500", "date": "2026-08-%02d" % (i + 1),
                 "signal": sig, "actual_next_ret": ret,
                 "timing_action": "HOLD", "shadow_action": act}
                for i, (sig, act, ret) in enumerate(spec)]

    def test_a_held_position_is_right_when_the_bar_goes_its_way(self):
        out = pr.live_timing_hits(
            self._rows([("BUY", "ENTER:+1", 0.02), ("BUY", "HOLD", 0.01)]),
            "shadow_action")
        assert list(out["verdicts"]) == [True, True]
        assert out["decided"] == 2 and out["hits"] == 2

    def test_a_held_position_is_wrong_when_it_does_not(self):
        out = pr.live_timing_hits(
            self._rows([("BUY", "ENTER:+1", -0.02), ("BUY", "HOLD", -0.01)]),
            "shadow_action")
        assert list(out["verdicts"]) == [False, False]

    def test_staying_out_of_a_loser_counts_as_right(self):
        """A timing policy does not call direction, so avoiding a bad trade is
        the other half of being correct."""
        out = pr.live_timing_hits(
            self._rows([("BUY", "STAY_OUT", -0.02)]), "shadow_action")
        assert list(out["verdicts"]) == [True]

    def test_staying_out_of_a_winner_counts_as_wrong(self):
        out = pr.live_timing_hits(
            self._rows([("BUY", "STAY_OUT", 0.02)]), "shadow_action")
        assert list(out["verdicts"]) == [False]

    def test_flat_on_both_sides_is_not_a_decision_at_all(self):
        out = pr.live_timing_hits(
            self._rows([("WAIT", "STAY_OUT", 0.02)]), "shadow_action")
        assert list(out["verdicts"]) == [None]
        assert out["decided"] == 0

    def test_an_unreconciled_bar_is_not_scored_either_way(self):
        out = pr.live_timing_hits(
            self._rows([("BUY", "ENTER:+1", None)]), "shadow_action")
        assert list(out["verdicts"]) == [None]

    def test_a_bar_the_policy_never_spoke_on_is_not_a_decision(self):
        rows = self._rows([("BUY", "ENTER:+1", 0.02), ("BUY", None, 0.02)])
        out = pr.live_timing_hits(rows, "shadow_action")
        assert out["verdicts"][1] is None, "no action logged is not a call"

    def test_an_entry_against_a_quiet_signal_keeps_its_own_side(self):
        """The log only ever writes ENTER:+1 / ENTER:-1. Matching a bare
        "ENTER" never fired, so an entry fell through to "the raw signal
        stands" and a policy entering while the signal was flat recorded no
        position: 24 of 637 logged entries as of 2026-08-23."""
        out = pr.live_timing_hits(
            self._rows([("WAIT", "ENTER:-1", -0.02), ("WAIT", "HOLD", -0.01)]),
            "shadow_action")
        assert list(out["verdicts"]) == [True, True], (
            "short into a falling bar is right, and used to read as flat")

    def test_the_live_reading_and_the_fit_share_one_definition(self):
        """Two implementations would let the card and a fit report different
        accuracies for the same decisions."""
        import numpy as np

        import train_timing as tt
        series = {"probs": np.array([0.9, 0.9, 0.1]),
                  "buy_thr": 0.6, "sell_thr": 0.4,
                  "next_ret": np.array([0.01, -0.01, -0.01])}
        sides = np.array([1, 1, 0])
        offline = tt.hit_stats(series, sides)
        rows = self._rows([("BUY", "ENTER:+1", 0.01), ("BUY", "HOLD", -0.01),
                           ("SELL", "EXIT", -0.01)])
        live = pr.live_timing_hits(rows, "shadow_action")
        assert offline["accuracy"] == live["accuracy"]
        assert list(offline["verdicts"]) == list(live["verdicts"])
