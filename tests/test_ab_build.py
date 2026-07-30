"""Unit tests for the A/B builder's configuration half (no training)."""

import io
import json

import ab_build

ADOPTED = {
    "label": "A",
    "genome": {"drops": ["vol_z"], "extra": [], "label_mode": "rel_median",
               "label_window": 30, "thr_margin": 0.02, "regime_mode": "off"},
}


def test_reference_is_the_adopted_genome_when_one_is_adopted(monkeypatch):
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: ADOPTED)
    ref = ab_build.reference()
    assert ref["label"] == "adopted:A"
    assert ref["sig"]
    assert ref["env"]["GTRADE_LABEL_MODE"] == "rel_median"


def test_reference_carries_the_dsl_spec_file(monkeypatch):
    # env_overrides omits GTRADE_DSL_SPECS on purpose, and training children are
    # forced unadopted, so a reference built from it would name seven features
    # and compute none of them.
    rec = {"label": "A", "genome": dict(ADOPTED["genome"],
           extra=[{"name": "zscore_vol_z_20", "op": "zscore",
                   "inputs": ["vol_z"], "params": {"window": 20}}])}
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: rec)
    ref = ab_build.reference()
    assert "GTRADE_DSL_SPECS" in ref["env"]
    assert ref["env"]["GTRADE_EXTRA_FEATURES"] == "zscore_vol_z_20"


def test_reference_is_the_bare_base_when_nothing_is_adopted(monkeypatch):
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: None)
    ref = ab_build.reference()
    assert ref["label"] == "base"
    assert ref["sig"] is None
    assert ref["env"] == {}


def test_previous_holdouts_are_read_from_every_result_file(tmp_path):
    io.open(str(tmp_path / "_ab_genomes_20260101-0000.json"), "w",
            encoding="utf-8").write(json.dumps({"holdout": "AAA,BBB",
                                                "results": {}}))
    io.open(str(tmp_path / "_ab_genomes_20260202-0000.json"), "w",
            encoding="utf-8").write(json.dumps({"holdout": "CCC",
                                                "results": {}}))
    got = ab_build.previous_holdouts(str(tmp_path))
    assert sorted(sum((h.split(",") for h in got), [])) == ["AAA", "BBB", "CCC"]


def test_a_corrupt_result_file_does_not_stop_the_scan(tmp_path):
    io.open(str(tmp_path / "_ab_genomes_bad.json"), "w",
            encoding="utf-8").write("{not json")
    io.open(str(tmp_path / "_ab_genomes_ok.json"), "w",
            encoding="utf-8").write(json.dumps({"holdout": "AAA",
                                                "results": {}}))
    assert ab_build.previous_holdouts(str(tmp_path)) == ["AAA"]


def test_the_config_records_what_it_will_measure_against(monkeypatch):
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: ADOPTED)
    cand = {"label": "0_4_1", "genome": {"drops": ["rsi"]}, "sig": "sigX"}
    cfg = ab_build.build_config([cand], ["A1", "A2"], ab_build.reference(),
                                floor=0.5, alpha=0.05, seed=3,
                                objective="mean")
    assert cfg["reference"] == "adopted:A"
    assert cfg["reference_sig"]
    assert cfg["holdout"] == "A1,A2"
    assert cfg["floor"] == 0.5 and cfg["alpha"] == 0.05 and cfg["seed"] == 3
    assert [c["label"] for c in cfg["candidates"]] == ["0_4_1"]


def test_the_floor_is_frozen_into_the_config(monkeypatch):
    # Read at config time and written down, so a later change to the constant
    # cannot silently reinterpret a pending run.
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: None)
    cfg = ab_build.build_config([], ["A1"], ab_build.reference(), floor=0.9,
                                alpha=0.01, seed=0, objective="mean")
    assert cfg["floor"] == 0.9 and cfg["alpha"] == 0.01


def test_a_config_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: None)
    cfg = ab_build.build_config([], ["A1", "A2"], ab_build.reference(), 0.5,
                                0.05, 0, "mean")
    p = str(tmp_path / "_ab_config.json")
    ab_build.write_config(cfg, p)
    assert ab_build.read_config(p) == cfg


def test_a_missing_config_reads_as_none(tmp_path):
    assert ab_build.read_config(str(tmp_path / "absent.json")) is None


def test_a_candidate_equal_to_the_reference_is_flagged(monkeypatch):
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: ADOPTED)
    ref = ab_build.reference()
    same = {"label": "same", "genome": ADOPTED["genome"], "sig": ref["sig"]}
    other = {"label": "other", "genome": {"drops": ["rsi"]}, "sig": "different"}
    assert ab_build.is_reference(same, ref) is True
    assert ab_build.is_reference(other, ref) is False


def test_a_real_shaped_candidate_is_recognised_as_the_reference(monkeypatch):
    # The earlier test used a hand-built dict with a sig, which passed while the
    # real candidate dicts had no sig at all.
    import auto_research as ar
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: ADOPTED)
    ref = ab_build.reference()
    same = {"label": "x", "genome": ADOPTED["genome"],
            "sig": ar.genome_sig(ar.Genome(**ADOPTED["genome"]))}
    assert ab_build.is_reference(same, ref) is True


def test_nothing_is_the_reference_when_nothing_is_adopted(monkeypatch):
    monkeypatch.setattr(ab_build, "_adopted_record", lambda: None)
    ref = ab_build.reference()
    assert ab_build.is_reference({"label": "x", "genome": {}, "sig": None},
                                 ref) is False
