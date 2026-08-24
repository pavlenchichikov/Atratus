"""Read-only joins over the research journals. No live files, no market.db."""

import json
import sqlite3

import pytest

from core import experience


def _write(tmp_path, name, obj):
    (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")


GENOME_A = {"drops": ["vol_z"], "extra": [], "label_mode": "rel_median",
            "label_window": 30, "thr_margin": 0.02, "regime_mode": "off"}
GENOME_B = {"drops": ["vol_z", "rsi"], "extra": [], "label_mode": "direction",
            "label_window": 1, "thr_margin": 0.0, "regime_mode": "off"}


@pytest.fixture
def tree(tmp_path):
    """A miniature version of the real journal set."""
    _write(tmp_path, "_ar_findings.json", [
        {"ts": "2026-08-01T00:00:00", "mode": "axes", "axes": ["pruning"],
         "winners": [{"genome": GENOME_A, "axis": "pruning", "p": 0.01,
                      "value": 1.5, "tag": "ok", "adoptable": True,
                      "replicated": False, "clears": 1, "neural_lift": -0.3}]},
        {"ts": "2026-08-02T00:00:00", "mode": "axes", "axes": ["hyper"],
         "winners": []},
        {"ts": "2026-08-03T00:00:00", "mode": "axes", "axes": ["pruning"],
         "winners": [{"genome": GENOME_B, "axis": "pruning", "p": 0.4,
                      "value": 0.1, "tag": "no", "adoptable": False,
                      "replicated": False, "clears": 0, "neural_lift": 0.2}]},
    ])
    _write(tmp_path, "_ar_tried.json", {"genome": ["s1", "s2", "s3", "s4"],
                                        "genome@net_auc": ["s5"]})
    _write(tmp_path, "_ar_replication.json", {"sigA": ["t1", "t2"], "sigB": ["t1"]})
    return tmp_path


def test_journalled_sigs_maps_each_genome_to_the_records_that_flagged_it(tree):
    sigs = experience.journalled_sigs(base=str(tree))
    assert len(sigs) == 2
    assert all(len(v) == 1 for v in sigs.values())


def test_funnel_counts_every_stage(tree):
    f = experience.funnel(base=str(tree))
    assert f["tried"] == 5           # 4 + 1 across both buckets, deduplicated
    assert f["journalled"] == 2
    assert f["cleared_once"] == 2    # both signatures in the replication file
    assert f["cleared_twice"] == 1   # only sigA has two stamps
    assert f["adopted"] == 0         # no adopted_genome.json in the fixture


def test_a_missing_source_reads_as_empty_not_an_error(tmp_path):
    f = experience.funnel(base=str(tmp_path))
    assert f == {"tried": 0, "journalled": 0, "cleared_once": 0,
                 "cleared_twice": 0, "adopted": 0}


def test_a_corrupt_source_reads_as_empty_not_an_error(tmp_path):
    (tmp_path / "_ar_tried.json").write_text("{not json", encoding="utf-8")
    assert experience.funnel(base=str(tmp_path))["tried"] == 0


def test_levers_of_names_every_gene_that_differs_from_the_bare_base():
    got = experience.levers_of(GENOME_A)
    assert "drop:vol_z" in got
    assert "label:rel_median/30" in got
    assert "thr:0.02" in got
    # regime_mode "off" differs from the dataclass default "both", so it is
    # a lever this genome chose
    assert "regimemode:off" in got


def test_a_genome_that_chose_nothing_has_no_levers():
    import dataclasses

    import auto_research as ar
    assert experience.levers_of(dataclasses.asdict(ar.Genome())) == []


def test_a_dsl_operation_is_one_lever():
    g = dict(GENOME_A, extra=['["lag", ["ret_10"], [["k", 10]]]'])
    got = experience.levers_of(g)
    assert 'op:["lag", ["ret_10"], [["k", 10]]]' in got


def test_lever_yield_counts_genomes_flags_and_replications(tree):
    rows = {r["lever"]: r for r in experience.levers(base=str(tree))}
    # vol_z is dropped by both fixture genomes, only one of which was flagged
    assert rows["drop:vol_z"]["genomes"] == 2
    assert rows["drop:vol_z"]["flagged"] == 1
    # rsi is dropped by GENOME_B alone, which was never flagged
    assert rows["drop:rsi"]["genomes"] == 1
    assert rows["drop:rsi"]["flagged"] == 0


def test_lever_yield_carries_the_median_neural_lift(tree):
    rows = {r["lever"]: r for r in experience.levers(base=str(tree))}
    # -0.3 from GENOME_A and +0.2 from GENOME_B
    assert rows["drop:vol_z"]["neural_lift"] == pytest.approx(-0.05)


def test_lever_yield_is_empty_without_sources(tmp_path):
    assert experience.levers(base=str(tmp_path)) == []


def test_verdicts_are_collected_per_signature(tmp_path):
    _write(tmp_path, "_ab_genomes_20260801-0000.json", {
        "reference": "adopted:A", "holdout": "BTC,ETH",
        "results": {"c1": {"sig": "sigA", "value_raw": 1.2, "p_raw": 0.02,
                           "n_raw": 14, "sd_raw": 3.0, "mde": 2.4,
                           "powered": False, "promoted": 3, "demoted": 10,
                           "p_promotion": 0.9}},
    })
    v = experience.verdicts(base=str(tmp_path))
    assert list(v) == ["sigA"]
    assert v["sigA"][0]["promoted"] == 3
    assert v["sigA"][0]["powered"] is False
    assert v["sigA"][0]["file"] == "_ab_genomes_20260801-0000.json"


def test_genomes_lists_every_journalled_genome_with_its_levers(tree):
    import auto_research as ar
    sig_a = ar.genome_sig(ar.Genome(**GENOME_A))
    sig_b = ar.genome_sig(ar.Genome(**GENOME_B))
    rows = {r["sig"]: r for r in experience.genomes(base=str(tree))}
    assert set(rows) == {sig_a, sig_b}
    assert "drop:vol_z" in rows[sig_a]["levers"]


def test_genomes_is_empty_without_sources(tmp_path):
    assert experience.genomes(base=str(tmp_path)) == []


def test_genome_record_gathers_levers_findings_and_verdicts(tree):
    import auto_research as ar
    sig = ar.genome_sig(ar.Genome(**GENOME_A))
    rec = experience.genome(sig, base=str(tree))
    assert rec["sig"] == sig
    assert "drop:vol_z" in rec["levers"]
    assert len(rec["findings"]) == 1
    assert rec["findings"][0]["adoptable"] is True
    assert rec["verdicts"] == []
    assert rec["clears"] == 0


def test_an_unknown_signature_returns_an_empty_record_not_an_error(tree):
    rec = experience.genome("nosuchsig", base=str(tree))
    assert rec["sig"] == "nosuchsig"
    assert rec["levers"] == [] and rec["findings"] == []


def test_similar_ranks_neighbours_by_shared_levers(tree):
    import auto_research as ar
    sig = ar.genome_sig(ar.Genome(**GENOME_A))
    near = experience.similar(sig, base=str(tree))
    assert [n["sig"] for n in near] == [ar.genome_sig(ar.Genome(**GENOME_B))]
    # both fixture genomes set regime_mode "off", so that lever is shared too
    assert near[0]["shared"] == ["drop:vol_z", "regimemode:off"]
    assert 0 < near[0]["overlap"] < 1


def test_similar_never_returns_the_genome_itself(tree):
    import auto_research as ar
    sig = ar.genome_sig(ar.Genome(**GENOME_A))
    assert sig not in [n["sig"] for n in experience.similar(sig, base=str(tree))]


def test_unresolved_signatures_are_named_not_dropped(tmp_path):
    """A clear or a verdict whose genome is not in the journal is evidence
    about the join, and hiding it hides the composed-genome case."""
    _write(tmp_path, "_ar_findings.json", [])
    _write(tmp_path, "_ar_replication.json", {"ghost": ["t1"]})
    _write(tmp_path, "_ab_genomes_20260801-0000.json",
           {"results": {"c1": {"sig": "phantom", "value_raw": 1.0}}})
    u = experience.unresolved(base=str(tmp_path))
    assert u["replication"] == ["ghost"]
    assert u["verdicts"] == ["phantom"]


def test_nothing_is_unresolved_when_every_signature_is_journalled(tree):
    import auto_research as ar
    sig = ar.genome_sig(ar.Genome(**GENOME_A))
    _write(tree, "_ar_replication.json", {sig: ["t1"]})
    u = experience.unresolved(base=str(tree))
    assert u["replication"] == [] and u["verdicts"] == []


def test_verdicts_survives_a_results_field_that_is_not_a_dict(tmp_path):
    _write(tmp_path, "_ab_genomes_20260801-0000.json",
           {"results": ["not", "a", "dict"]})
    assert experience.verdicts(base=str(tmp_path)) == {}


@pytest.fixture
def plog(tmp_path):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, "
                "signal TEXT, probability REAL, correct INTEGER, "
                "model_version TEXT)")
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?)", [
        ("2026-06-01", "BTC", "BUY", 0.7, 1, "aaaa1111"),
        ("2026-06-02", "BTC", "BUY", 0.6, 0, "aaaa1111"),
        ("2026-06-03", "ETH", "SELL", 0.8, None, "aaaa1111"),
        ("2026-07-01", "BTC", "BUY", 0.9, 1, "bbbb2222"),
    ])
    con.commit()
    con.close()
    return path


def test_generations_report_accuracy_over_reconciled_rows_only(plog):
    rows = {r["model_version"]: r for r in experience.generations(db_path=plog)}
    assert rows["aaaa1111"]["n"] == 3
    assert rows["aaaa1111"]["reconciled"] == 2
    assert rows["aaaa1111"]["accuracy"] == pytest.approx(0.5)
    assert rows["aaaa1111"]["first"] == "2026-06-01"
    assert rows["aaaa1111"]["last"] == "2026-06-03"


def test_generations_are_ordered_newest_first(plog):
    assert [r["model_version"] for r in experience.generations(db_path=plog)] == \
        ["bbbb2222", "aaaa1111"]


def test_generations_degrade_to_empty_without_a_database(tmp_path):
    assert experience.generations(db_path=str(tmp_path / "missing.db")) == []
