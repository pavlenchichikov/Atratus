"""Unit tests for core.adopted (pure; no network, no training)."""

import json

from core import adopted

GENOME_A = {
    "drops": ["corr_sp500", "macro_tnx_chg20", "trend_strength", "vol_z"],
    "extra": [
        {"name": "zscore_vol_z_20", "op": "zscore", "inputs": ["vol_z"],
         "params": {"window": 20}},
        {"name": "ratio_bb_pos_rsi", "op": "ratio", "inputs": ["bb_pos", "rsi"],
         "params": {}},
    ],
    "label_mode": "rel_median",
    "label_window": 30,
    "thr_margin": 0.02,
    "regime_mode": "off",
}


def write(tmp_path, payload):
    p = tmp_path / "adopted_genome.json"
    open(str(p), "w", encoding="utf-8").write(json.dumps(payload))
    return str(p)


def test_a_missing_file_is_no_adoption(tmp_path):
    assert adopted.load(str(tmp_path / "absent.json")) is None


def test_corrupt_payloads_are_no_adoption(tmp_path):
    bad = tmp_path / "bad.json"
    open(str(bad), "w", encoding="utf-8").write("{not json")
    assert adopted.load(str(bad)) is None
    assert adopted.load(write(tmp_path, [1, 2, 3])) is None
    assert adopted.load(write(tmp_path, {"label": "A"})) is None


def test_a_valid_file_loads(tmp_path):
    got = adopted.load(write(tmp_path, {"label": "A", "genome": GENOME_A}))
    assert got["label"] == "A"
    assert got["genome"]["label_mode"] == "rel_median"


def test_env_overrides_composes_genome_a():
    env = adopted.env_overrides(GENOME_A)
    assert env["GTRADE_DROP_FEATURES"] == (
        "corr_sp500,macro_tnx_chg20,trend_strength,vol_z")
    assert env["GTRADE_LABEL_MODE"] == "rel_median"
    assert env["GTRADE_LABEL_WINDOW"] == "30"
    assert env["GTRADE_THR_MARGIN"] == "0.02"
    assert env["GTRADE_REGIME_MODE"] == "off"
    assert env["GTRADE_EXTRA_FEATURES"] == "zscore_vol_z_20,ratio_bb_pos_rsi"


def test_env_overrides_never_sets_the_temp_spec_path():
    # Production reads the adopted specs directly. A temp path would die on the
    # next reboot and silently take the seven features with it.
    assert "GTRADE_DSL_SPECS" not in adopted.env_overrides(GENOME_A)


def test_a_default_valued_gene_is_omitted():
    # An empty override dict must mean "production default", so a gene sitting on
    # its default must not be emitted at all.
    env = adopted.env_overrides({"label_mode": "direction", "cb_lr_mult": 1.0,
                                 "regime_mode": "both", "thr_margin": 0.0})
    assert env == {}


def test_triple_barrier_uses_horizon_not_window():
    env = adopted.env_overrides({"label_mode": "triple_barrier",
                                 "label_window": 12})
    assert env["GTRADE_LABEL_HORIZON"] == "12"
    assert "GTRADE_LABEL_WINDOW" not in env


def test_apply_does_not_overwrite_a_shell_value():
    # A one-off experiment from the shell must beat the adopted default.
    environ = {"GTRADE_THR_MARGIN": "0.09"}
    keys = adopted.apply(GENOME_A, environ)
    assert environ["GTRADE_THR_MARGIN"] == "0.09"
    assert "GTRADE_THR_MARGIN" not in keys
    assert environ["GTRADE_LABEL_MODE"] == "rel_median"
    assert "GTRADE_LABEL_MODE" in keys


def test_specs_returns_the_dsl_list_or_empty():
    assert len(adopted.specs({"genome": GENOME_A})) == 2
    assert adopted.specs({"genome": {}}) == []
    assert adopted.specs(None) == []


def test_the_research_loop_and_production_compose_a_genome_identically():
    # If these two ever disagree, production trains something the A/B never
    # measured and nothing raises. That is the whole reason the rules live in one
    # module rather than being copied.
    from dataclasses import asdict

    import auto_research as ar

    g = ar.Genome(drops=["vol_z", "rsi"],
                  extra=[{"name": "zscore_vol_z_20", "op": "zscore",
                          "inputs": ["vol_z"], "params": {"window": 20}}],
                  label_mode="rel_median", label_window=30,
                  cb_depth_delta=-1, cb_lr_mult=1.5, cb_iter_mult=0.7,
                  lookback_delta=5, net_seeds=3, net_uniqueness=1,
                  net_calibrate=1, thr_margin=0.02, band_delta=0.01,
                  regime_mode="off")
    from_research = dict(ar.genome_to_env(g))
    # Research adds a per-candidate temp path; production reads the adopted file.
    from_research.pop("GTRADE_DSL_SPECS", None)
    assert from_research == adopted.env_overrides(asdict(g))
