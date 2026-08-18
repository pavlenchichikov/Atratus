"""The route from an axis winner to an A/B arm.

An axis run gates its winner on the held-out set exactly as the QD search does,
but its result is an ENV and everything downstream speaks Genome. Until that
join existed, an adoptable axis winner was unreachable by construction: it
cleared the gate, went into the journal, and no A/B could ever be built from it.
These tests cover the join itself, its refusal to guess, and the pool it feeds.
"""

import json

import adopt_genome
import auto_research as ar


def _axis(name):
    return ar.build_axes([name], ["ret_1", "ret_5", "vol_z", "rsi"])[0]


def test_a_labeling_winner_becomes_the_genome_with_that_label():
    g = ar.genome_from_axis(_axis("labeling"),
                            {"mode": "rel_median", "window": 20})
    assert (g.label_mode, g.label_window) == ("rel_median", 20)


def test_a_triple_barrier_window_is_the_horizon_on_both_sides():
    # The axis writes GTRADE_LABEL_HORIZON while rel_median writes _WINDOW, so a
    # genome that stored the number in the wrong field would train a different
    # label than the gate measured.
    g = ar.genome_from_axis(_axis("labeling"),
                            {"mode": "triple_barrier", "window": 10})
    assert ar.genome_to_env(g)["GTRADE_LABEL_HORIZON"] == "10"


def test_the_nets_axis_keys_are_translated_to_their_gene_names():
    g = ar.genome_from_axis(_axis("nets"), {"seeds": 3, "calibrate": 1})
    assert (g.net_seeds, g.net_calibrate) == (3, 1)


def test_hyper_and_threshold_winners_map_gene_for_gene():
    assert ar.genome_from_axis(_axis("hyper"), {"cb_lr_mult": 0.5}).cb_lr_mult == 0.5
    g = ar.genome_from_axis(_axis("thresholds"),
                            {"thr_margin": 0.02, "band_delta": 0.01})
    assert (g.thr_margin, g.band_delta) == (0.02, 0.01)


def test_a_pruning_winner_becomes_the_drops():
    g = ar.genome_from_axis(_axis("pruning"), [{"drop": "vol_z"}, {"drop": "rsi"}])
    assert g.drops == ["vol_z", "rsi"]


def test_a_feature_winner_keeps_its_specs_and_their_names():
    spec = {"op": "lag", "inputs": ["ret_5"], "params": {"k": 3}, "name": "lag_ret_5_3"}
    g = ar.genome_from_axis(_axis("features"), [spec])
    assert g.extra == [spec]
    assert ar.genome_to_env(g)["GTRADE_EXTRA_FEATURES"] == "lag_ret_5_3"


def test_a_mapping_that_does_not_reproduce_the_measured_env_is_refused():
    """The positive control: the check must be able to FAIL.

    Without this the passing cases prove only that the function returns
    something. A genome that composes to a different env than the gate measured
    would send the A/B off to train one thing and file it as evidence for
    another, which is worse than offering no candidate at all.
    """
    class Drifted:
        name = "labeling"
        to_env = staticmethod(
            lambda cand: {"GTRADE_LABEL_MODE": "rel_median",
                          "GTRADE_LABEL_WINDOW": "999"})

    assert ar.genome_from_axis(Drifted, {"mode": "rel_median", "window": 20}) is None


def test_a_winner_with_no_gene_of_that_name_maps_to_nothing():
    class Unknown:
        name = "nets"
        to_env = staticmethod(lambda cand: {"GTRADE_NET_SEEDS": "3"})

    assert ar.genome_from_axis(Unknown, {"no_such_gene": 1}) is None


# --- the pool the A/B picker reads ------------------------------------------

def _journal(tmp_path, basis="net_auc", adoptable=True):
    genome = {"label_mode": "rel_median", "label_window": 20}
    rec = {"ts": "2026-08-18T04:14:26", "mode": "axes", "basis": basis,
           "axes": ["labeling"], "budget": 30,
           "winners": [{"axis": "labeling", "p": 0.0001, "value": 0.0918,
                        "tag": "mean dScore 0.09", "adoptable": adoptable,
                        "neural_lift": -0.55, "genome": genome,
                        "replicated": True, "clears": 7}]}
    (tmp_path / "_ar_findings.json").write_text(json.dumps([rec]), encoding="utf-8")
    (tmp_path / "_auto_loop.json").write_text(
        json.dumps({"campaign": {"GTRADE_AR_SCORE_BASIS": "net_auc",
                                 "GTRADE_AR_OBJECTIVE": "mean"}}), encoding="utf-8")
    return str(tmp_path)


def test_an_adoptable_axis_winner_reaches_the_candidate_pool(tmp_path):
    got = adopt_genome.candidates(_journal(tmp_path))
    assert [c["label"] for c in got] == ["axis:labeling"]
    assert got[0]["genome"]["label_window"] == 20
    # Its number came from a gate, not an A/B against the live reference, so it
    # is still a search-stage candidate.
    assert got[0]["kind"] == "search"
    assert got[0]["validated"] is False


def test_a_winner_the_gate_turned_down_is_not_offered(tmp_path):
    assert adopt_genome.candidates(_journal(tmp_path, adoptable=False)) == []


def test_a_winner_from_another_basis_is_not_offered(tmp_path):
    """Score-basis values run 1.5 to 8.9 and AUC-basis ones about 0.01, so one
    survivor of the other campaign outranks every candidate of this one."""
    assert adopt_genome.candidates(_journal(tmp_path, basis="raw")) == []


def test_the_archive_wins_when_a_genome_is_in_both(tmp_path):
    base = _journal(tmp_path)
    genome = {"label_mode": "rel_median", "label_window": 20}
    (tmp_path / "_qd_archive.json").write_text(
        json.dumps({"2_4_5": {"fitness": 0.065, "genome": genome}}),
        encoding="utf-8")
    got = adopt_genome.candidates(base)
    assert [c["label"] for c in got] == ["2_4_5"]
