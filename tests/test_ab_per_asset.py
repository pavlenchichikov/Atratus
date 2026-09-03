"""The recovery that reads an A/B's per-seed arms back out of the cache.

Every number the per-asset workflow rests on comes through here, and the arms are
identified by TIME rather than by cache key (a key hashes the data fingerprint,
and market.db moves). Two clock traps cost an hour on 2026-09-02 and both are
pinned below, because the failure they produce is not an error - it is a
plausible number computed from the wrong pair of arms.
"""
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import ab_per_asset

REF = '[regate full {"drops": ["corr_sp500"] 6 chunks on 2 processes: A,B'
CAND = '[regate full {"drops": []] 6 chunks on 2 processes: A,B'


def _log(tmp_path, lines):
    path = tmp_path / "gtrade.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _utc(local_str):
    """The cache stamp for a local log time, using the machine's own offset."""
    local = datetime.strptime(local_str, "%Y-%m-%d %H:%M:%S")
    return (local - timedelta(seconds=ab_per_asset._utc_offset_seconds())).isoformat()


def test_a_phase_is_attributed_to_the_arm_its_signature_names(tmp_path):
    phases = ab_per_asset.phases(_log(tmp_path, [
        "2026-09-01 08:32:50 | INFO | auto_research | " + REF,
        "2026-09-01 14:21:20 | INFO | auto_research | " + CAND,
    ]))
    assert [p[2] for p in phases] == ["ref", "cand"]
    assert [p[1] for p in phases] == ["full", "full"]


def test_the_log_clock_is_converted_to_the_cache_clock(tmp_path):
    """cache_put stamps datetime.utcnow() while the log is local time. Three
    hours apart on this box, and every entry then files under the wrong phase."""
    phases = ab_per_asset.phases(_log(tmp_path, [
        "2026-09-01 08:32:50 | INFO | auto_research | " + REF]))
    assert phases[0][0] == _utc("2026-09-01 08:32:50")


def test_the_day_cutoff_does_not_swallow_the_first_day(tmp_path):
    """The log writes '2026-09-01 08:32:50' with a space, which sorts BELOW 'T',
    so comparing it raw against an ISO cutoff dropped every phase of day one and
    left three seeds looking like the whole run."""
    lines = ["2026-09-01 08:32:50 | INFO | auto_research | " + REF,
             "2026-09-02 01:00:00 | INFO | auto_research | " + CAND]
    since = _utc("2026-09-01 08:00:00")
    assert len(ab_per_asset.phases(_log(tmp_path, lines), since=since)) == 2


def test_an_upper_bound_keeps_a_later_run_out_of_this_one(tmp_path):
    """A confirmation run writes the same '[regate full ...]' lines. Without the
    bound its seeds would be read as more seeds of the original A/B."""
    lines = ["2026-09-01 08:32:50 | INFO | auto_research | " + REF,
             "2026-09-02 07:36:49 | INFO | auto_research | " + REF]
    got = ab_per_asset.phases(_log(tmp_path, lines),
                              until=_utc("2026-09-02 00:00:00"))
    assert len(got) == 1


def test_only_entries_inside_a_full_phase_become_an_arm(tmp_path):
    lines = ["2026-09-01 08:00:00 | INFO | auto_research | " + REF,
             '2026-09-01 09:00:00 | INFO | auto_research | [regate cb {"drops": ["x"] 6 chunks on 2 processes: A',
             "2026-09-01 10:00:00 | INFO | auto_research | " + CAND]
    phases = ab_per_asset.phases(_log(tmp_path, lines))
    cache = {
        "k1": {"ts": _utc("2026-09-01 08:30:00"),
               "rows": [{"Asset": "A", "Score": 1.0}, {"Asset": "B", "Score": 2.0}]},
        "kcb": {"ts": _utc("2026-09-01 09:30:00"),
                "rows": [{"Asset": "A", "Score": 99.0}]},
        "k2": {"ts": _utc("2026-09-01 10:30:00"),
               "rows": [{"Asset": "A", "Score": 3.0}, {"Asset": "B", "Score": 4.0}]},
    }
    rolls = ab_per_asset.rolls(cache, {"A", "B"}, phases)
    assert set(rolls) == {("ref", 0), ("cand", 0)}
    assert rolls[("ref", 0)] == {"A": 1.0, "B": 2.0}
    assert rolls[("cand", 0)] == {"A": 3.0, "B": 4.0}, "the CB screen is not an arm"


def test_only_assets_every_roll_scored_are_paired():
    """An asset averaged over fewer seeds than its neighbours carries a different
    amount of noise with nothing on the row to say so."""
    rolls = {("ref", 0): {"A": 1.0, "B": 1.0}, ("ref", 1): {"A": 2.0, "B": 2.0},
             ("cand", 0): {"A": 3.0, "B": 3.0}, ("cand", 1): {"A": 4.0}}
    assets, deltas = ab_per_asset.paired(rolls)
    assert assets == ["A"]
    assert deltas.tolist() == [[2.0, 2.0]]


def test_benjamini_hochberg_keeps_the_run_of_small_p_values():
    p = np.array([0.001, 0.02, 0.30, 0.60])
    keep = ab_per_asset.bh(p, 0.10)
    assert keep.tolist() == [True, True, False, False]
    # positive control: nothing survives when nothing is small
    assert not ab_per_asset.bh(np.array([0.4, 0.5, 0.9]), 0.10).any()


def test_a_broken_recovery_refuses_instead_of_reporting(tmp_path, monkeypatch):
    """The whole tool rests on an inference about which entries are which arm, so
    it has to reproduce the value the run recorded or say nothing at all.

    The times are relative to now on purpose: original() bounds the search by the
    mtimes of the config it reads and the result it checks against, so a fixture
    dated last week would be filtered out before any arm was even identified.
    """
    now = datetime.now().replace(microsecond=0)
    t_ref = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    t_cand = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    monkeypatch.setattr(ab_per_asset, "CONFIG", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(ab_per_asset, "CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setattr(ab_per_asset, "LOG", _log(tmp_path, [
        "%s | INFO | auto_research | %s" % (t_ref, REF),
        "%s | INFO | auto_research | %s" % (t_cand, CAND)]))
    monkeypatch.setattr(ab_per_asset, "BASE", str(tmp_path))
    (tmp_path / "cfg.json").write_text(json.dumps({"holdout": "A,B"}),
                                       encoding="utf-8")
    (tmp_path / "cache.json").write_text(json.dumps({
        "k1": {"ts": _utc(t_ref), "rows": [{"Asset": "A", "Score": 1.0},
                                           {"Asset": "B", "Score": 1.0}]},
        "k2": {"ts": _utc(t_cand), "rows": [{"Asset": "A", "Score": 2.0},
                                            {"Asset": "B", "Score": 2.0}]},
    }), encoding="utf-8")
    (tmp_path / "_ab_genomes_x.json").write_text(json.dumps({
        "results": {"cand": {"value_raw": 999.0}}}), encoding="utf-8")
    # the config has to look older than the phases it bounds
    old = (now - timedelta(hours=3)).timestamp()
    os.utime(tmp_path / "cfg.json", (old, old))
    try:
        ab_per_asset.original()
        raise AssertionError("a mismatched recovery must not be reported")
    except SystemExit as exc:
        assert "MISMATCH" in str(exc)
    # positive control: the recorded value the arms DO produce is accepted
    (tmp_path / "_ab_genomes_x.json").write_text(json.dumps({
        "results": {"cand": {"value_raw": 1.0}}}), encoding="utf-8")
    assets, _deltas, _label, _file, _rec, got = ab_per_asset.original()
    assert assets == ["A", "B"] and got == 1.0


def test_step_two_confirms_what_step_one_picked():
    """A hardcoded asset list would silently re-confirm the PREVIOUS run's
    winners against the current run's arms, which replicates nothing."""
    import ab_confirm

    assets = ["A", "B", "C"]
    # A is large and consistent, B is large and noisy, C is small.
    deltas = np.array([[4.0, 4.2, 3.8, 4.1],
                       [4.0, -3.0, 6.0, -1.0],
                       [0.1, 0.0, -0.1, 0.2]])
    picked, why = ab_confirm.picks_from_scan(assets, deltas)
    assert picked == ["A"] and "correction" in why
    # and when nothing clears, the extremes are offered with that said out loud
    picked, why = ab_confirm.picks_from_scan(assets[1:], deltas[1:], limit=1)
    assert picked == ["B"] and "not for significance" in why


def test_an_asset_on_its_own_genome_is_labelled_as_such(monkeypatch):
    """A delta against the global adoption and a delta against the asset's OWN
    adoption are different claims, and the row read identically for both.
    -3.8 on an adopted asset means this pass would undo measured, replicated
    work; on an unadopted one it just means the candidate does not help."""
    monkeypatch.setattr("core.adopted.load", lambda path=None: {
        "genome": {"drops": []},
        "per_asset": {"rtx": {"genome": {"drops": ["x"]},
                              "adopted": "2026-09-02"}}})
    own, where, dates = ab_per_asset.adoption_state(["RTX", "SBER"])
    assert set(own) == {"RTX"}, "the file's lower-case key still matches"
    assert where == {"RTX": "own", "SBER": "global"}
    assert dates["RTX"] == "2026-09-02"


def test_with_nothing_adopted_no_asset_claims_a_genome(monkeypatch):
    monkeypatch.setattr("core.adopted.load", lambda path=None: {})
    _own, where, _dates = ab_per_asset.adoption_state(["RTX"])
    assert where == {"RTX": "-"}, "not 'global' when there is no global genome"


def test_an_adoption_made_after_the_baseline_ran_is_flagged():
    """The reference arm is only "the current adopted state" if it ran under
    it. The eval cache is keyed by data fingerprint and seed, not by genome,
    so a pre-adoption baseline row survives an adoption and is reused."""
    dates = {"RTX": "2026-09-02", "AUDCAD": "2026-09-03", "_global": "2026-07-27"}
    assert ab_per_asset.stale_baseline("2026-09-02", dates) == ["AUDCAD"]
    # everything already in force when the arm ran is fine
    assert ab_per_asset.stale_baseline("2026-09-05", dates) == []
    # and with no run date there is nothing to compare against, so no claim
    assert ab_per_asset.stale_baseline(None, dates) == []


def test_the_refusal_says_which_floor_this_holdout_could_resolve(monkeypatch):
    """Telling the operator how many assets a floor needs answers only half the
    question. The other half - given the assets I have, what can I ask - is
    what sent them to the console to guess at a variable name."""
    import ab_build

    monkeypatch.setattr(ab_build, "last_spread", lambda base=None: 2.4272)
    assert ab_build.resolvable_floor(40) == pytest.approx(0.954, abs=0.002)
    assert ab_build.resolvable_floor(146) == pytest.approx(0.5, abs=0.01)

    rows = ab_build.power_table(40)
    assert "floor +0.50  needs  146 assets" in rows[0]
    assert "is not enough" in rows[0]
    assert "this holdout of 40 answers it" in rows[2], "floor 1.00 clears"


def test_no_banked_spread_means_no_claim_about_power(monkeypatch):
    """Before any A/B has run there is nothing to project from, and inventing a
    spread would refuse runs on an imagined number."""
    import ab_build

    monkeypatch.setattr(ab_build, "last_spread", lambda base=None: None)
    assert ab_build.resolvable_floor(40) is None
    assert ab_build.power_table(40) == []


def test_the_chosen_floor_is_what_gets_frozen_into_the_config():
    """The verdict is read months after the environment that produced it moved
    on, so the floor travels in the config rather than being re-derived."""
    import types

    import ab_build

    args = types.SimpleNamespace(floor=1.25, objective="mean")
    assert ab_build._floor_for(args) == 1.25
    args.floor = None
    assert ab_build._floor_for(args) > 0, "falls back to the basis default"
