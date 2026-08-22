"""Live accuracy, split by which champion generation wrote the row.

The live log needs no correction to be honest: every row was written on the day
by the champion in force then, and scored against the bar that followed. It is
out of sample by construction.

An earlier version of this split the same rows into "in sample" and "out of
sample" against the CURRENT champion's training window, which was simply wrong -
3364 of the 4717 rows predate their asset's current champion and were written by
an earlier one, honestly. The split that means something is by generation, and
it answers the question a retrain actually raises: is the new model better than
the one it replaced.
"""
import json
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import model_health as mh


def _setup(tmp_path, registry, rows):
    models = tmp_path / "models"
    models.mkdir()
    (models / "champion_registry.json").write_text(json.dumps(registry),
                                                   encoding="utf-8")
    con = sqlite3.connect(tmp_path / "market.db")
    con.execute("CREATE TABLE prediction_log (asset TEXT, date TEXT, correct INT)")
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?)", rows)
    con.commit()
    con.close()
    return str(models)


def _point(monkeypatch, tmp_path, models):
    monkeypatch.setattr(mh, "MODEL_DIR", models)
    monkeypatch.setattr(mh, "BASE_DIR", str(tmp_path))


def test_rows_are_attributed_to_the_generation_that_wrote_them(monkeypatch,
                                                               tmp_path):
    models = _setup(
        tmp_path,
        {"AAA": {"updated_at": "2026-08-10"}},
        [("AAA", "2026-08-0%d" % d, 0) for d in (1, 2, 3)]      # old champion
        + [("AAA", "2026-08-1%d" % d, 1) for d in (1, 2, 3)])   # new one
    _point(monkeypatch, tmp_path, models)
    out = mh.print_generations(min_bars=1)
    assert out["earlier"] == 3 and out["earlier_hit"] == 0
    assert out["current"] == 3 and out["current_hit"] == 3


def test_an_old_row_is_never_called_in_sample(monkeypatch, tmp_path, capsys):
    """The bug this replaced: a July row written by July's champion is out of
    sample whatever August's champion has since been fitted on."""
    models = _setup(tmp_path, {"AAA": {"updated_at": "2026-08-10"}},
                    [("AAA", "2026-08-01", 1)])
    _point(monkeypatch, tmp_path, models)
    mh.print_generations(min_bars=1)
    text = capsys.readouterr().out.lower()
    assert "out of sample" in text
    assert "in sample" not in text and "in-sample" not in text


def test_a_champion_with_no_date_is_counted_apart(monkeypatch, tmp_path):
    models = _setup(tmp_path, {"AAA": {"score": 1.0}},
                    [("AAA", "2026-08-01", 1), ("AAA", "2026-08-02", 0)])
    _point(monkeypatch, tmp_path, models)
    out = mh.print_generations(min_bars=1)
    assert out["unknown"] == 2
    assert out["current"] == 0 and out["earlier"] == 0


def test_the_training_day_itself_belongs_to_the_old_generation(monkeypatch,
                                                               tmp_path):
    models = _setup(tmp_path, {"AAA": {"updated_at": "2026-08-10"}},
                    [("AAA", "2026-08-10", 1), ("AAA", "2026-08-11", 1)])
    _point(monkeypatch, tmp_path, models)
    out = mh.print_generations(min_bars=1)
    assert out["earlier"] == 1 and out["current"] == 1


def test_too_few_rows_reads_n_a_rather_than_a_percentage(monkeypatch, tmp_path,
                                                         capsys):
    models = _setup(tmp_path, {"AAA": {"updated_at": "2026-08-01"}},
                    [("AAA", "2026-08-02", 1)])
    _point(monkeypatch, tmp_path, models)
    mh.print_generations(min_bars=5)
    assert "n/a" in capsys.readouterr().out
