import sqlite3

import pytest

from core.analyst import lessons, store


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    for t in ("sp500", "nasdaq"):
        con.execute(f'CREATE TABLE "{t}" (Date TEXT, Open REAL, High REAL, Low REAL, Close REAL)')
        for i in range(1, 29):
            con.execute(f'INSERT INTO "{t}" VALUES (?,1,1,1,1)', ("2026-08-%02d" % i,))
    con.commit()
    con.close()
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setattr(lessons, "_klass", lambda a: "us")
    for asset, date, h, ret in (("SP500", "2026-08-03", 1, 0.02),
                                ("SP500", "2026-08-10", 5, -0.01),
                                ("NASDAQ", "2026-08-04", 1, 0.01)):
        store.write_judgment({"date": date, "asset": asset, "horizon": h,
                              "direction": "down", "conviction": 3, "thesis": "t",
                              "key_risk": "k"}, db_path=path)
        c = sqlite3.connect(path)
        c.execute("UPDATE analyst_log SET realized_ret=? WHERE asset=? AND date=?",
                  (ret, asset, date))
        c.commit()
        c.close()
    return path


def test_resolved_date_is_the_horizonth_bar_after(db):
    assert lessons.resolved_date("SP500", "2026-08-10", 5) == "2026-08-15"
    assert lessons.resolved_date("SP500", "2026-08-27", 5) is None


def test_learn_writes_one_lesson_per_row_once(db):
    calls = []
    n = lessons.learn(lambda p: calls.append(p) or "Do not fight a trend.", db_path=db)
    assert n == 3 and any("moved +2.00%" in c for c in calls)
    assert lessons.learn(lambda p: calls.append(p) or "x", db_path=db) == 0
    assert len(calls) == 3


def test_a_lesson_shows_only_after_its_outcome_resolved(db):
    lessons.learn(lambda p: "Lesson.", db_path=db)
    early = [g for g in lessons.lessons_for("SP500", "us", today="2026-08-12", db_path=db)
             if " SP500:" in g]
    assert len(early) == 1 and early[0].startswith("2026-08-04 SP500")
    late = [g for g in lessons.lessons_for("SP500", "us", today="2026-08-20", db_path=db)
            if " SP500:" in g]
    assert len(late) == 2


def test_class_lessons_fill_in_after_the_asset_own(db):
    lessons.learn(lambda p: "Lesson.", db_path=db)
    got = lessons.lessons_for("SP500", "us", today=None, k=3, db_path=db)
    assert [g.split()[1].rstrip(":") for g in got] == ["SP500", "SP500", "NASDAQ"]


def test_a_timeout_stops_learn(db):
    from core.llm_proposer import CallTimedOut

    def dead(p):
        raise CallTimedOut("slow")

    with pytest.raises(CallTimedOut):
        lessons.learn(dead, db_path=db)


def test_the_dossier_carries_lessons_and_the_prompt_names_them(db, monkeypatch):
    from core.analyst import agent, dossier
    lessons.learn(lambda p: "Respect the trend.", db_path=db)
    monkeypatch.setattr(lessons, "_connect", lambda db_path: lessons.sqlite3.connect(db))
    got = dossier._lessons("SP500", "2026-08-20")
    assert got and "Respect the trend." in got[0]
    assert "lessons" in agent.prompt_for({"asset": "SP500", "lessons": got})


def test_the_learn_command_uses_the_memory_brain(db, monkeypatch, capsys):
    import analyst
    roles = []
    monkeypatch.setattr("core.analyst.brains.call_for",
                        lambda role: roles.append(role) or (lambda p: "Lesson."))
    assert analyst.main(["learn", "--limit", "2"]) == 0
    assert roles == ["memory"] and "wrote 2 lesson(s)" in capsys.readouterr().out


def test_lessons_do_not_move_the_dossier_hash():
    from core.analyst import dossier
    d = {"asset": "SP500", "close": 1.0}
    assert dossier.dossier_hash(d) == dossier.dossier_hash(dict(d, lessons=["x"]))


def test_rows_that_cannot_resolve_or_fail_do_not_block_newer_ones(db):
    calls = []

    def flaky(p):
        calls.append(p)
        return "" if len(calls) == 1 else "Lesson."

    assert lessons.learn(flaky, limit=2, db_path=db) == 1
    # the failed row is recorded as tried and not asked again
    assert lessons.learn(flaky, limit=2, db_path=db) == 1
    assert len(calls) == 3
    assert all(": " in g and not g.endswith(": None")
               for g in lessons.lessons_for("SP500", "us", db_path=db))
