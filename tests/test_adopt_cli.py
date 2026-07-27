"""Unit tests for the adoption CLI (pure; reads fixture files from tmp_path)."""

import io
import json
import os

import adopt_genome

ARCHIVE = {
    "3_4_5": {"fitness": 5.3,
              "genome": {"drops": ["vol_z"], "extra": [],
                         "label_mode": "rel_median", "label_window": 30,
                         "thr_margin": 0.02, "regime_mode": "off"}},
    "0_0_1": {"fitness": 2.1,
              "genome": {"drops": ["rsi"], "extra": [],
                         "label_mode": "direction", "label_window": 30}},
}


def _ab_for(sig):
    return {"holdout": "AMZN,JPM", "objective": "mean", "floor": 0.5,
            "results": {"A": {"sig": sig, "value_raw": 1.63, "p_raw": 0.0067,
                              "n_raw": 14, "value_neural": -0.65,
                              "p_neural": 0.948, "label": "A"}}}


def _fixtures(tmp_path):
    import auto_research as ar
    io.open(str(tmp_path / "_qd_archive.json"), "w", encoding="utf-8").write(
        json.dumps(ARCHIVE))
    sig = ar.genome_sig(ar.Genome(**ARCHIVE["3_4_5"]["genome"]))
    io.open(str(tmp_path / "_ab_genomes_20260101-0000.json"), "w",
            encoding="utf-8").write(json.dumps(_ab_for(sig)))
    return str(tmp_path)


def test_candidates_marks_measured_and_search_stage(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    kinds = {c["kind"] for c in got}
    assert kinds == {"measured", "search"}
    measured = [c for c in got if c["kind"] == "measured"][0]
    assert measured["value"] == 1.63
    assert measured["p"] == 0.0067
    assert measured["validated"] is True


def test_a_measured_candidate_carries_the_full_genome_from_the_archive(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    measured = [c for c in got if c["kind"] == "measured"][0]
    # The A/B file stores only a signature; the specs and their names come from
    # the archive join.
    assert measured["genome"]["label_mode"] == "rel_median"
    assert measured["genome"]["thr_margin"] == 0.02


def test_a_search_candidate_is_not_validated_and_carries_no_p(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    search = [c for c in got if c["kind"] == "search"][0]
    assert search["validated"] is False
    assert search["p"] is None
    # Its number is a search fitness and must never be labelled a gain.
    assert search["value"] == 2.1
    assert "NOT validated" in adopt_genome.describe(search)


def test_a_measured_candidate_is_described_as_measured(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    measured = [c for c in got if c["kind"] == "measured"][0]
    line = adopt_genome.describe(measured)
    assert "MEASURED" in line and "p=0.0067" in line


def test_write_adoption_records_the_evidence(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    measured = [c for c in got if c["kind"] == "measured"][0]
    dest = str(tmp_path / "adopted_genome.json")
    adopt_genome.write_adoption(measured, dest)
    rec = json.loads(io.open(dest, encoding="utf-8").read())
    assert rec["genome"]["label_mode"] == "rel_median"
    assert rec["evidence"]["p"] == 0.0067
    assert rec["evidence"]["n"] == 14
    assert "holdout" in rec["evidence"]


def test_adopting_twice_keeps_the_first_as_the_previous(tmp_path):
    dest = str(tmp_path / "adopted_genome.json")
    io.open(dest, "w", encoding="utf-8").write(json.dumps({"label": "OLD",
                                                           "genome": {}}))
    got = adopt_genome.candidates(_fixtures(tmp_path))
    adopt_genome.write_adoption([c for c in got if c["validated"]][0], dest)
    prev = json.loads(io.open(str(tmp_path / "adopted_genome.prev.json"),
                              encoding="utf-8").read())
    assert prev["label"] == "OLD"


def test_revert_restores_the_previous_adoption(tmp_path):
    dest = str(tmp_path / "adopted_genome.json")
    prev = str(tmp_path / "adopted_genome.prev.json")
    io.open(prev, "w", encoding="utf-8").write(json.dumps({"label": "OLD",
                                                           "genome": {}}))
    io.open(dest, "w", encoding="utf-8").write(json.dumps({"label": "NEW",
                                                           "genome": {}}))
    assert adopt_genome.revert(dest) is True
    assert json.loads(io.open(dest, encoding="utf-8").read())["label"] == "OLD"
    assert not os.path.exists(prev)


def test_revert_with_no_previous_removes_the_file(tmp_path):
    dest = str(tmp_path / "adopted_genome.json")
    io.open(dest, "w", encoding="utf-8").write(json.dumps({"label": "NEW",
                                                           "genome": {}}))
    assert adopt_genome.revert(dest) is True
    assert not os.path.exists(dest)


def test_revert_with_nothing_adopted_reports_false(tmp_path):
    assert adopt_genome.revert(str(tmp_path / "absent.json")) is False


def test_adopting_resets_the_chunk_progress(tmp_path):
    # Without this, a second adoption finds every asset already recorded done and
    # silently trains nothing.
    prog = tmp_path / "_chunk_progress.txt"
    qual = tmp_path / "_chunk_quality.json"
    io.open(str(prog), "w", encoding="utf-8").write("BTC\nETH\n")
    io.open(str(qual), "w", encoding="utf-8").write("[]")
    removed = adopt_genome.reset_chunk_progress(str(tmp_path))
    assert not os.path.exists(str(prog))
    assert not os.path.exists(str(qual))
    assert len(removed) == 2


def test_resetting_when_there_is_no_progress_is_harmless(tmp_path):
    assert adopt_genome.reset_chunk_progress(str(tmp_path)) == []
