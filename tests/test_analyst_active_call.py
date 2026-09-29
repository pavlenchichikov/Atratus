import sqlite3

import pytest

from core.analyst import score, store


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE "sber" (Date TEXT, open REAL, high REAL, low REAL, close REAL)')
    for i in range(1, 31):
        con.execute('INSERT INTO "sber" VALUES (?,1,1,1,1)', ("2026-09-%02d" % i,))
    con.commit()
    con.close()
    monkeypatch.setattr(store, "DB_PATH", path)
    return path


def test_a_long_horizon_call_stays_active_until_it_resolves(db):
    store.write_judgment({"date": "2026-09-03", "asset": "SBER", "horizon": 20,
                          "direction": "up"}, db_path=db)
    store.write_judgment({"date": "2026-09-03", "asset": "SBER", "horizon": 1,
                          "direction": "up"}, db_path=db)
    act = store.active_call("SBER", 20, "2026-09-10", db_path=db)
    assert act is not None and act["date"] == "2026-09-03"
    # 20 bars after 09-03 is 09-23: from then on nothing is active
    assert store.active_call("SBER", 20, "2026-09-24", db_path=db) is None
    # the next day's question is new every day: never "active"
    assert store.active_call("SBER", 1, "2026-09-03", db_path=db) is None


def test_a_revision_names_the_call_it_replaces_and_becomes_the_active_one(db):
    store.write_judgment({"date": "2026-09-03", "asset": "SBER", "horizon": 20,
                          "direction": "up"}, db_path=db)
    store.write_judgment({"date": "2026-09-08", "asset": "SBER", "horizon": 20,
                          "direction": "down", "revision_of": "2026-09-03"}, db_path=db)
    act = store.active_call("SBER", 20, "2026-09-10", db_path=db)
    assert act["date"] == "2026-09-08" and act["revision_of"] == "2026-09-03"


def test_flip_rate_counts_rejudgments_made_before_the_previous_call_resolved():
    rows = [{"asset": "SBER", "horizon": 20, "date": "2026-09-18", "direction": "up"},
            {"asset": "SBER", "horizon": 20, "date": "2026-09-23", "direction": "up"},
            {"asset": "SBER", "horizon": 20, "date": "2026-09-28", "direction": "down"},
            {"asset": "SBER", "horizon": 1, "date": "2026-09-28", "direction": "down"},
            {"asset": "GAZP", "horizon": 20, "date": "2026-01-05", "direction": "up"},
            {"asset": "GAZP", "horizon": 20, "date": "2026-03-05", "direction": "down"}]
    fr = score.flip_rate(rows)
    # SBER: two re-judgments inside the window, one of them flipped; GAZP's second
    # call came after the first resolved, so it is not a re-judgment; horizon 1 never is
    assert fr == {"rejudged": 2, "flipped": 1, "rate": 0.5}


def test_run_skips_an_active_call_unless_asked_to_revise(db):
    import analyst
    store.write_judgment({"date": "2026-09-03", "asset": "SBER", "horizon": 20,
                          "direction": "up"}, db_path=db)
    assert analyst._active_gate("SBER", 20, "2026-09-10", revise=False) == (True, None)
    assert analyst._active_gate("SBER", 20, "2026-09-10", revise=True) == (False, "2026-09-03")
    assert analyst._active_gate("SBER", 1, "2026-09-10", revise=False) == (False, None)
    assert analyst._active_gate("SBER", 20, "2026-09-25", revise=False) == (False, None)


def test_the_card_names_the_call_a_revision_replaced(db):
    import webapp
    store.write_judgment({"date": "2026-09-03", "asset": "SBER", "horizon": 20,
                          "direction": "up"}, db_path=db)
    row = {"date": "2026-09-08", "asset": "SBER", "horizon": 20, "direction": "down",
           "revision_of": "2026-09-03"}
    out = webapp._decorate_judgment(row)
    assert out["revises"] == {"date": "2026-09-03", "direction": "up"}
    assert webapp._decorate_judgment({"date": "2026-09-08", "asset": "SBER",
                                      "horizon": 20, "direction": "up"})["revises"] is None
