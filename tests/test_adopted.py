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


def test_cb_uniqueness_gene_emits_its_env():
    from core.adopted import env_overrides
    assert env_overrides({"cb_uniqueness": 1}).get("GTRADE_CB_UNIQUENESS") == "1"
    assert "GTRADE_CB_UNIQUENESS" not in env_overrides({"cb_uniqueness": 0})


# --- per-asset adoption ------------------------------------------------------

RTX_GENOME = {"drops": [], "extra": [
    {"name": "only_rtx_has_this", "op": "ratio", "inputs": ["bb_pos", "rsi"],
     "params": {}}], "net_seeds": 3}
RECORD = {"label": "A", "genome": GENOME_A,
          "per_asset": {"rtx": {"genome": RTX_GENOME, "evidence": "replicated"}}}


def test_an_asset_keeps_the_genome_measured_on_it():
    # The 2026-09-02 candidate failed the gate at -0.30 over 40 assets while
    # being worth +1.20 on RTX, both replicated. Adopting everywhere or nowhere
    # throws one of those facts away.
    assert adopted.genome_for("RTX", RECORD) == RTX_GENOME
    assert adopted.genome_for("AAPL", RECORD) == GENOME_A


def test_a_hand_written_asset_key_still_matches():
    """A per-asset entry that silently never matched would look like a working
    adoption and serve the global genome forever."""
    assert set(adopted.per_asset(RECORD)) == {"RTX"}
    assert adopted.genome_for("rtx", RECORD) == RTX_GENOME
    assert adopted.per_asset({"per_asset": {"X": "not a dict"}}) == {}
    assert adopted.per_asset({"per_asset": []}) == {}


def test_the_dsl_specs_are_the_union_over_every_genome_in_force():
    """core.scoring refuses an asset whose saved feature list names a column the
    frame does not have, so one asset's extra feature has to be computed for
    everyone or that asset drops off the radar."""
    names = [s["name"] for s in adopted.specs(RECORD)]
    assert "only_rtx_has_this" in names
    assert "zscore_vol_z_20" in names
    assert len(names) == len(set(names)), "a shared spec must not be computed twice"


def test_a_process_runs_under_the_genome_of_the_assets_it_was_handed():
    assert adopted.genome_for_assets("RTX", RECORD) == RTX_GENOME
    assert adopted.genome_for_assets("AAPL,MSFT", RECORD) == GENOME_A
    # Serving names no assets and keeps the global genome, exactly as before
    # per-asset adoption existed.
    assert adopted.genome_for_assets(None, RECORD) == GENOME_A
    assert adopted.genome_for_assets("", RECORD) == GENOME_A


def test_a_mixed_process_is_called_out_rather_than_silently_resolved(capsys):
    """One process, one environment: a set spanning two genomes cannot be served
    by either, and picking one quietly would train assets under a genome nobody
    adopted for them."""
    got = adopted.genome_for_assets("AAPL,RTX", RECORD)
    assert got == GENOME_A
    assert "different genomes" in capsys.readouterr().out


def test_a_record_without_the_map_behaves_exactly_as_before():
    plain = {"label": "A", "genome": GENOME_A}
    assert adopted.per_asset(plain) == {}
    assert adopted.genome_for("RTX", plain) == GENOME_A
    assert adopted.genome_for_assets("RTX", plain) == GENOME_A
    assert [s["name"] for s in adopted.specs(plain)] == \
        [s["name"] for s in GENOME_A["extra"]]


def _record():
    return {
        "genome": {"extra": [{"name": "mNEW", "op": "interaction",
                              "inputs": ["a", "b"], "params": {}}],
                   "label_mode": "direction", "label_window": 30},
        "per_asset": {"CAC40": {"genome": {
            "extra": [{"name": "mOLD", "op": "lag", "inputs": ["x"],
                       "params": {"k": 3}}],
            "label_mode": "rel_median", "label_window": 30}}},
    }


def test_serving_can_swap_the_genome_between_assets():
    """Serving scans the whole map in ONE process, so config resolved the global
    genome at import and a per-asset adoption was a decision the system took and
    could not honour. Measured 2026-09-05: 211 assets kept a champion trained
    under the previous genome, correctly, and every one of them was skipped
    because the features that champion needs had stopped being built."""
    from core import adopted

    rec = _record()
    keys = adopted.serving_keys(rec, already_set=adopted.env_overrides(rec["genome"]))
    env = {}
    adopted.apply_for("SBER", rec, keys, environ=env)
    assert env["GTRADE_EXTRA_FEATURES"] == "mNEW", "an unpinned asset gets the global genome"
    adopted.apply_for("CAC40", rec, keys, environ=env)
    assert env["GTRADE_EXTRA_FEATURES"] == "mOLD", "a pinned asset gets its own"


def test_a_genome_does_not_leak_into_the_next_asset():
    """apply() never overwrites a key that is already set, so without an explicit
    removal the first asset processed would fix the genome for every asset after
    it - the same leak config.py strips the environment to avoid when it starts
    one trainer per genome."""
    from core import adopted

    rec = _record()
    keys = adopted.serving_keys(rec, already_set=adopted.env_overrides(rec["genome"]))
    env = {}
    adopted.apply_for("CAC40", rec, keys, environ=env)
    assert env.get("GTRADE_LABEL_MODE") == "rel_median"
    adopted.apply_for("SBER", rec, keys, environ=env)
    assert env["GTRADE_EXTRA_FEATURES"] == "mNEW", "did not swap back"
    assert "GTRADE_LABEL_MODE" not in env, (
        "the pinned asset's label mode survived into an unpinned one; the global "
        "genome does not set it, so it has to be REMOVED, not left behind")


def test_a_value_set_in_the_shell_is_never_swapped_away(monkeypatch):
    """apply()'s promise is that an existing value wins, so a one-off experiment
    exported in the shell beats the adoption. Serving swaps keys per asset and
    must not quietly break that: anything the adoption did not set is not ours."""
    from core import adopted

    rec = _record()
    # config applied the global genome but did NOT set EXTRA_FEATURES, which is
    # what it looks like when the operator exported that key themselves.
    already = [k for k in adopted.env_overrides(rec["genome"])
               if k != "GTRADE_EXTRA_FEATURES"]
    keys = adopted.serving_keys(rec, already_set=already)
    assert "GTRADE_EXTRA_FEATURES" not in keys

    env = {"GTRADE_EXTRA_FEATURES": "mHAND"}
    adopted.apply_for("CAC40", rec, keys, environ=env)
    assert env["GTRADE_EXTRA_FEATURES"] == "mHAND"


def test_with_no_per_asset_adoption_serving_changes_nothing():
    """The common case has to stay a no-op, or every scan pays for a mechanism
    that has nothing to do."""
    from core import adopted

    rec = {"genome": {"extra": [], "label_mode": "direction", "label_window": 30}}
    keys = adopted.serving_keys(rec, already_set=adopted.env_overrides(rec["genome"]))
    env = {}
    for asset in ("SBER", "CAC40", "BTC"):
        adopted.apply_for(asset, rec, keys, environ=env)
    assert all(v == adopted.env_overrides(rec["genome"]).get(k) for k, v in env.items())
