"""Unit tests for the adoption CLI (pure; reads fixture files from tmp_path)."""

import json
import os

import pytest

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
    open(str(tmp_path / "_qd_archive.json"), "w", encoding="utf-8").write(
        json.dumps(ARCHIVE))
    sig = ar.genome_sig(ar.Genome(**ARCHIVE["3_4_5"]["genome"]))
    open(str(tmp_path / "_ab_genomes_20260101-0000.json"), "w",
            encoding="utf-8").write(json.dumps(_ab_for(sig)))
    return str(tmp_path)


def test_candidates_marks_measured_and_search_stage(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    kinds = {c["kind"] for c in got}
    assert kinds == {"measured", "search"}
    measured = next(c for c in got if c["kind"] == "measured")
    assert measured["value"] == 1.63
    assert measured["p"] == 0.0067
    assert measured["validated"] is True


def test_a_measured_candidate_carries_the_full_genome_from_the_archive(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    measured = next(c for c in got if c["kind"] == "measured")
    # The A/B file stores only a signature; the specs and their names come from
    # the archive join.
    assert measured["genome"]["label_mode"] == "rel_median"
    assert measured["genome"]["thr_margin"] == 0.02


def test_a_search_candidate_is_not_validated_and_carries_no_p(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    search = next(c for c in got if c["kind"] == "search")
    assert search["validated"] is False
    assert search["p"] is None
    # Its number is a search fitness and must never be labelled a gain.
    assert search["value"] == 2.1
    assert "NOT validated" in adopt_genome.describe(search)


def test_a_passing_candidate_is_described_as_passed(tmp_path):
    # The verdict, not just the provenance: "measured" would be true of a failed
    # A/B row too, and that distinction is the whole point of the listing.
    got = adopt_genome.candidates(_fixtures(tmp_path))
    measured = next(c for c in got if c["kind"] == "measured")
    line = adopt_genome.describe(measured)
    assert "PASSED" in line and "p=0.0067" in line
    assert "FAILED" not in line


def test_write_adoption_records_the_evidence(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    measured = next(c for c in got if c["kind"] == "measured")
    dest = str(tmp_path / "adopted_genome.json")
    adopt_genome.write_adoption(measured, dest)
    rec = json.loads(open(dest, encoding="utf-8").read())
    assert rec["genome"]["label_mode"] == "rel_median"
    assert rec["evidence"]["p"] == 0.0067
    assert rec["evidence"]["n"] == 14
    assert "holdout" in rec["evidence"]


def test_adopting_twice_keeps_the_first_as_the_previous(tmp_path):
    dest = str(tmp_path / "adopted_genome.json")
    open(dest, "w", encoding="utf-8").write(json.dumps({"label": "OLD",
                                                           "genome": {}}))
    got = adopt_genome.candidates(_fixtures(tmp_path))
    adopt_genome.write_adoption(next(c for c in got if c["validated"]), dest)
    prev = json.loads(open(str(tmp_path / "adopted_genome.prev.json"),
                              encoding="utf-8").read())
    assert prev["label"] == "OLD"


def test_revert_restores_the_previous_adoption(tmp_path):
    dest = str(tmp_path / "adopted_genome.json")
    prev = str(tmp_path / "adopted_genome.prev.json")
    open(prev, "w", encoding="utf-8").write(json.dumps({"label": "OLD",
                                                           "genome": {}}))
    open(dest, "w", encoding="utf-8").write(json.dumps({"label": "NEW",
                                                           "genome": {}}))
    assert adopt_genome.revert(dest) is True
    assert json.loads(open(dest, encoding="utf-8").read())["label"] == "OLD"
    assert not os.path.exists(prev)


def test_revert_with_no_previous_removes_the_file(tmp_path):
    dest = str(tmp_path / "adopted_genome.json")
    open(dest, "w", encoding="utf-8").write(json.dumps({"label": "NEW",
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
    open(str(prog), "w", encoding="utf-8").write("BTC\nETH\n")
    open(str(qual), "w", encoding="utf-8").write("[]")
    removed = adopt_genome.reset_chunk_progress(str(tmp_path))
    assert not os.path.exists(str(prog))
    assert not os.path.exists(str(qual))
    assert len(removed) == 2


def test_resetting_when_there_is_no_progress_is_harmless(tmp_path):
    assert adopt_genome.reset_chunk_progress(str(tmp_path)) == []


# --- the gate that decides what may be adopted at all ---

def _failed_ab(sig):
    """An A/B row that missed its own alpha and floor."""
    return {"holdout": "AMZN,JPM", "objective": "mean", "floor": 0.5,
            "alpha": 0.05,
            "results": {"B": {"sig": sig, "value_raw": 0.40, "p_raw": 0.3349,
                              "n_raw": 14, "value_neural": -0.45,
                              "p_neural": 0.729, "label": "B"}}}


def _fixtures_failed(tmp_path):
    import auto_research as ar
    open(str(tmp_path / "_qd_archive.json"), "w", encoding="utf-8").write(
        json.dumps(ARCHIVE))
    sig = ar.genome_sig(ar.Genome(**ARCHIVE["3_4_5"]["genome"]))
    open(str(tmp_path / "_ab_genomes_20260101-0000.json"), "w",
            encoding="utf-8").write(json.dumps(_failed_ab(sig)))
    return str(tmp_path)


def test_an_ab_candidate_that_missed_its_own_alpha_is_not_validated(tmp_path):
    # Coming from an A/B file is provenance, not a pass. Offering a failed
    # candidate as a decision is how a p=0.33 result reaches production.
    got = adopt_genome.candidates(_fixtures_failed(tmp_path))
    failed = next(c for c in got if c["label"] == "B")
    assert failed["kind"] == "measured"
    assert failed["validated"] is False
    assert "FAILED" in adopt_genome.describe(failed)


def test_a_row_with_no_p_value_does_not_crash_the_listing(tmp_path):
    import auto_research as ar
    open(str(tmp_path / "_qd_archive.json"), "w", encoding="utf-8").write(
        json.dumps(ARCHIVE))
    sig = ar.genome_sig(ar.Genome(**ARCHIVE["3_4_5"]["genome"]))
    open(str(tmp_path / "_ab_genomes_20260101-0000.json"), "w",
            encoding="utf-8").write(json.dumps(
                {"holdout": "AMZN", "results": {"X": {"sig": sig}}}))
    got = adopt_genome.candidates(str(tmp_path))
    line = adopt_genome.describe(next(c for c in got if c["label"] == "X"))
    assert "n/a" in line


def test_main_refuses_a_search_elite_without_the_flag(tmp_path, monkeypatch,
                                                     capsys):
    # Without this the whole --unvalidated gate is untested: deleting it would
    # make every search elite adoptable by default and no test would notice.
    monkeypatch.setattr(adopt_genome, "BASE", _fixtures_failed(tmp_path))
    monkeypatch.setattr("sys.argv", ["adopt_genome.py"])
    monkeypatch.setattr("builtins.input", lambda *a: "")
    adopt_genome.main()
    out = capsys.readouterr().out
    # The only A/B row failed, and search elites need the flag, so nothing is
    # offered at all.
    assert "No validated candidates" in out
    assert "0_0_1" not in out


def test_main_offers_search_elites_only_with_the_flag(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(adopt_genome, "BASE", _fixtures_failed(tmp_path))
    monkeypatch.setattr("sys.argv", ["adopt_genome.py", "--unvalidated"])
    monkeypatch.setattr("builtins.input", lambda *a: "")
    adopt_genome.main()
    out = capsys.readouterr().out
    assert "search fitness" in out
    assert "Cancelled" in out


def test_the_written_record_carries_the_caveat(tmp_path):
    got = adopt_genome.candidates(_fixtures(tmp_path))
    measured = next(c for c in got if c["validated"])
    dest = str(tmp_path / "adopted_genome.json")
    adopt_genome.write_adoption(measured, dest)
    rec = json.loads(open(dest, encoding="utf-8").read())
    caveat = rec["evidence"]["caveat"]
    assert "assumption" in caveat and "2 held-out" in caveat


def test_every_candidate_carries_its_signature(tmp_path):
    # Without this, a tool downstream cannot tell a candidate from the genome it
    # would be measured against, and pays hours to discover they are the same.
    got = adopt_genome.candidates(_fixtures(tmp_path))
    assert got, "fixture produced no candidates"
    assert all(c.get("sig") for c in got)


def test_a_net_demoting_result_is_not_a_validated_candidate(tmp_path):
    """A stored result is read long after it was written. The 2026-08-18 adoption
    was reverted by hand while its file still said PASSED, and the next loop cycle
    would have re-adopted it: adoptable=['axis:labeling'], next=adopt."""
    import auto_research as ar
    open(str(tmp_path / "_qd_archive.json"), "w", encoding="utf-8").write(
        json.dumps(ARCHIVE))
    sig = ar.genome_sig(ar.Genome(**ARCHIVE["3_4_5"]["genome"]))
    body = _ab_for(sig)
    body["results"]["A"].update({"promoted": 3, "demoted": 10, "p_promotion": 0.989})
    open(str(tmp_path / "_ab_genomes_20260101-0000.json"), "w",
         encoding="utf-8").write(json.dumps(body))
    got = adopt_genome.candidates(str(tmp_path))
    measured = next(c for c in got if c["kind"] == "measured")
    assert measured["validated"] is False


def test_a_result_that_promotes_more_than_it_demotes_still_passes(tmp_path):
    import auto_research as ar
    open(str(tmp_path / "_qd_archive.json"), "w", encoding="utf-8").write(
        json.dumps(ARCHIVE))
    sig = ar.genome_sig(ar.Genome(**ARCHIVE["3_4_5"]["genome"]))
    body = _ab_for(sig)
    body["results"]["A"].update({"promoted": 9, "demoted": 4, "p_promotion": 0.09})
    open(str(tmp_path / "_ab_genomes_20260101-0000.json"), "w",
         encoding="utf-8").write(json.dumps(body))
    measured = next(c for c in adopt_genome.candidates(str(tmp_path))
                    if c["kind"] == "measured")
    assert measured["validated"] is True


def test_a_file_written_before_the_counts_existed_is_judged_as_before(tmp_path):
    """Older results carry no counts; absent must not read as a veto."""
    measured = next(c for c in adopt_genome.candidates(_fixtures(tmp_path))
                    if c["kind"] == "measured")
    assert measured["validated"] is True


# --- per-asset adoption ------------------------------------------------------

def _record(tmp_path):
    path = tmp_path / "adopted.json"
    path.write_text(json.dumps({"label": "A", "genome": {"drops": ["vol_z"],
                                                         "extra": []}}),
                    encoding="utf-8")
    return str(path)


def test_a_per_asset_adoption_leaves_every_other_asset_alone(tmp_path):
    from core import adopted

    path = _record(tmp_path)
    genome = {"drops": [], "extra": [], "net_seeds": 3}
    adopt_genome.adopt_for_asset("rtx", genome, "replication +1.20 p=0.019",
                                 path=path)
    rec = adopted.load(path)
    assert adopted.genome_for("RTX", rec) == genome
    assert adopted.genome_for("AAPL", rec) == rec["genome"]
    assert rec["per_asset"]["RTX"]["evidence"]


def test_a_per_asset_adoption_refuses_without_replication_evidence(tmp_path):
    """The pass that SELECTS an asset overstates it - the three picked on
    2026-09-02 kept 30% of it on fresh seeds - so the file records the
    replication or nothing."""
    path = _record(tmp_path)
    with pytest.raises(SystemExit):
        adopt_genome.adopt_for_asset("RTX", {"net_seeds": 3}, "", path=path)
    with pytest.raises(SystemExit):
        adopt_genome.adopt_for_asset("RTX", {}, "measured", path=path)


def test_a_per_asset_adoption_needs_a_global_one_to_sit_beside(tmp_path):
    missing = str(tmp_path / "nothing.json")
    with pytest.raises(SystemExit):
        adopt_genome.adopt_for_asset("RTX", {"net_seeds": 3}, "measured",
                                     path=missing)


def test_dropping_one_asset_puts_it_back_on_the_global_genome(tmp_path):
    from core import adopted

    path = _record(tmp_path)
    adopt_genome.adopt_for_asset("RTX", {"net_seeds": 3}, "measured", path=path)
    assert adopt_genome.drop_asset_adoption("rtx", path=path) is True
    assert adopted.per_asset(adopted.load(path)) == {}
    # and the file is still a valid adoption afterwards
    assert adopted.load(path)["genome"]["drops"] == ["vol_z"]
    assert adopt_genome.drop_asset_adoption("RTX", path=path) is False


def test_one_asset_moving_does_not_order_a_retrain_of_the_other_207(tmp_path):
    prog = tmp_path / "_chunk_progress.txt"
    prog.write_text("AAPL\nRTX\nMSFT\n", encoding="utf-8")
    assert adopt_genome._forget_chunk_progress("rtx", base=str(tmp_path)) is True
    assert prog.read_text(encoding="utf-8").split() == ["AAPL", "MSFT"]
    assert adopt_genome._forget_chunk_progress("RTX", base=str(tmp_path)) is False


def test_the_adoption_report_names_every_per_asset_exception():
    """An exception nobody can see is the worst kind: the one screen that
    answers "what is adopted" must not show the global genome alone while an
    asset trains and serves on another."""
    rec = {"label": "A", "adopted": "2026-07-27", "evidence": {"value": 1.63},
           "genome": {"drops": ["vol_z"], "extra": []},
           "per_asset": {"RTX": {"adopted": "2026-09-02",
                                 "genome": {"drops": [], "extra": [],
                                            "net_seeds": 3},
                                 "evidence": "replication +1.198 p=0.019"}}}
    text = "\n".join(adopt_genome.report_lines(rec))
    assert "PER-ASSET EXCEPTIONS" in text
    assert "RTX" in text and "net_seeds=3" in text
    assert "replication +1.198" in text, "the evidence has to travel with it"
    # positive control: a plain adoption says nothing about exceptions
    plain = "\n".join(adopt_genome.report_lines(
        {"label": "A", "genome": rec["genome"], "evidence": {}}))
    assert "PER-ASSET" not in plain
