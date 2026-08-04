"""The live recalibration layer must never ship a constant.

A flat isotonic fit is what anti-calibrated data produces, it is a perfectly
valid IsotonicRegression, and it maps every asset to one probability - which on
2026-08-01 turned all 208 signals into WAIT.
"""

import sqlite3

import pytest

import recalibrate_live as rl


def _db(tmp_path, rows):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, "
                "probability REAL, actual_next_ret REAL, correct INTEGER)")
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def _rows(n, informative):
    """`informative` = a stream the layer can learn from (high probability really
    does mean up); otherwise the anti-calibrated stream that flattens the fit."""
    out = []
    for i in range(n):
        p = 0.05 + 0.9 * (i % 20) / 19.0
        up = (p > 0.5) if informative else (p < 0.5)
        out.append(("2026-07-%02d" % (i % 28 + 1), "BTC", "BUY", p, 0.01,
                    int(up)))
    return out


def test_a_constant_fit_is_refused(tmp_path, capsys):
    path = _db(tmp_path, _rows(600, informative=False))
    out = rl.main(days=3650, model_dir=str(tmp_path), db_path=path)
    assert out is None                       # nothing written
    assert "collapsed to a constant" in capsys.readouterr().out
    assert not (tmp_path / "live_calib_global.pkl").exists()


def test_a_usable_fit_is_still_written(tmp_path):
    # The positive control: the guard must not block a layer that does its job.
    path = _db(tmp_path, _rows(600, informative=True))
    out = rl.main(days=3650, model_dir=str(tmp_path), db_path=path)
    assert out is not None
    assert (tmp_path / "live_calib_global.pkl").exists()


def test_spread_helper_reports_a_flat_map_as_zero():
    class Flat:
        def predict(self, xs):
            return [0.4653] * len(xs)

    assert rl._fit_spread(Flat(), [0.1, 0.9]) == pytest.approx(0.0)
