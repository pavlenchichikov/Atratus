"""The rules the unattended cycle runs on, tested without touching market.db."""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import ab_build
import adopt_genome
import auto_loop
import auto_research as ar


def _clean_env():
    return dict(auto_loop.CAMPAIGN)


def _cand(sig, label, **kw):
    out = {"sig": sig, "label": label, "genome": {}, "validated": False,
           "value": None, "p": None, "kind": "search"}
    out.update(kw)
    return out


def _gate(adoptable=True, clears=0, ts="2026-08-16T10:00:00"):
    return {"adoptable": adoptable, "clears": clears, "ts": ts, "tag": "t"}


# --- phase ordering ---------------------------------------------------------

def test_next_action_finishes_a_pending_ab_before_anything_else():
    st = {"ab_pending": True, "adoptable": ["c1"], "untested": ["c2"]}
    assert auto_loop.next_action(st) == "ab_run"


def test_next_action_adopts_before_building_another_ab():
    st = {"ab_pending": False, "adoptable": ["c1"], "untested": ["c2"]}
    assert auto_loop.next_action(st) == "adopt"


def test_next_action_falls_back_to_search_when_there_is_nothing_to_gate():
    st = {"ab_pending": False, "adoptable": [], "untested": []}
    assert auto_loop.next_action(st) == "search"


def test_next_action_builds_an_ab_when_a_candidate_cleared_the_gate():
    st = {"ab_pending": False, "adoptable": [], "untested": ["c2"]}
    assert auto_loop.next_action(st) == "ab_build"


# --- campaign guards --------------------------------------------------------

def test_the_shipped_campaign_is_self_consistent():
    # Positive control for the two tests below: they only mean something if a
    # correct campaign really does come back empty.
    assert auto_loop.campaign_problems(_clean_env()) == []


def test_the_screen_is_rejected_on_a_net_basis():
    env = _clean_env()
    env["GTRADE_AR_SCREEN"] = "1"
    problems = auto_loop.campaign_problems(env)
    assert any("SCREEN" in p for p in problems)


def test_cheap_illumination_is_rejected_on_a_net_basis():
    env = _clean_env()
    env["GTRADE_AR_ILLUM"] = "cb"
    problems = auto_loop.campaign_problems(env)
    assert any("illuminated by CatBoost alone" in p for p in problems)


def test_full_illumination_is_rejected_on_the_raw_basis():
    env = _clean_env()
    env["GTRADE_AR_SCORE_BASIS"] = "raw"
    env["GTRADE_AR_SCREEN"] = "1"
    problems = auto_loop.campaign_problems(env)
    assert any("does not reproduce" in p for p in problems)


def test_a_moved_objective_stops_the_campaign():
    frozen = {"GTRADE_AR_SCORE_BASIS": "net_auc", "GTRADE_AR_OBJECTIVE": "mean"}
    env = dict(frozen, GTRADE_AR_OBJECTIVE="cvar")
    assert auto_loop.freeze_problems(frozen, env)
    # and an unchanged campaign is not flagged
    assert auto_loop.freeze_problems(frozen, dict(frozen)) == []


def test_a_first_campaign_does_not_discard_a_matching_archive(tmp_path, monkeypatch):
    """A first freeze has nothing to change FROM. Wiping the archive there
    destroys prior search work, which is what happened on 2026-08-17."""
    arch = tmp_path / "_qd_archive.json"
    arch.write_text(json.dumps({"3_4_5": {"fitness": 0.065, "genome": {}}}),
                    encoding="utf-8")
    monkeypatch.setattr(auto_loop, "ARCHIVE_PATH", str(arch))
    monkeypatch.setattr(auto_loop, "STATE_PATH", str(tmp_path / "s.json"))
    auto_loop.start_campaign({"campaign": None, "history": []},
                             dict(auto_loop.CAMPAIGN), "first campaign")
    assert arch.exists(), "the archive was set aside for no reason"


def test_a_basis_change_does_set_the_archive_aside(tmp_path, monkeypatch):
    arch = tmp_path / "_qd_archive.json"
    arch.write_text(json.dumps({"3_4_5": {"fitness": 0.065, "genome": {}}}),
                    encoding="utf-8")
    monkeypatch.setattr(auto_loop, "ARCHIVE_PATH", str(arch))
    state = {"campaign": {"GTRADE_AR_SCORE_BASIS": "net_auc",
                          "GTRADE_AR_OBJECTIVE": "mean"}, "history": []}
    env = dict(auto_loop.CAMPAIGN, GTRADE_AR_SCORE_BASIS="raw",
               GTRADE_AR_SCREEN="1", GTRADE_AR_ILLUM="cb")
    auto_loop.start_campaign(state, env, "switching")
    assert not arch.exists()
    assert (tmp_path / "_qd_archive.json.bak").exists()


def test_score_scale_elites_are_recognised_under_an_auc_basis():
    score = {"a": {"fitness": 5.30}, "b": {"fitness": 1.63}}
    auc = {"a": {"fitness": 0.065}, "b": {"fitness": -0.004}}
    # Score elites would outrank every AUC elite forever
    assert auto_loop.archive_scale_mismatch(score, "net_auc")
    # and AUC elites would hide every real Score winner
    assert auto_loop.archive_scale_mismatch(auc, "raw")
    # matching scales are left alone, which is the case that must not fire
    assert not auto_loop.archive_scale_mismatch(auc, "net_auc")
    assert not auto_loop.archive_scale_mismatch(score, "raw")
    assert not auto_loop.archive_scale_mismatch({}, "net_auc")


def test_the_freeze_refusal_names_its_own_way_out():
    """Without this the campaign is a dead end: it lives in _auto_loop.json, so
    every later start would refuse and say nothing about how to proceed."""
    frozen = {"GTRADE_AR_SCORE_BASIS": "net_auc", "GTRADE_AR_OBJECTIVE": "mean"}
    text = " ".join(auto_loop.freeze_problems(
        frozen, dict(frozen, GTRADE_AR_SCORE_BASIS="ens_auc")))
    assert "--new-campaign" in text


def test_every_basis_the_menu_offers_is_one_auto_research_accepts():
    from core import ar_memory
    menu = {"net_auc", "ens_auc", "net_gain", "raw", "neural"}
    assert menu <= set(ar_memory.SCORE_BASES)


def test_the_basis_pairings_the_menu_derives_are_all_runnable():
    """The launcher derives screen and illumination from the basis instead of
    asking. Every pairing it can produce must survive campaign_problems, or the
    menu would be able to build a run that refuses to start."""
    derived = {"net_auc": ("0", "full"), "ens_auc": ("0", "full"),
               "net_gain": ("0", "full"), "raw": ("1", "cb"),
               "neural": ("1", "cb")}
    for basis, (screen, illum) in derived.items():
        env = dict(auto_loop.CAMPAIGN, GTRADE_AR_SCORE_BASIS=basis,
                   GTRADE_AR_SCREEN=screen, GTRADE_AR_ILLUM=illum)
        assert auto_loop.campaign_problems(env) == [], basis
    # positive control: the pairing the menu refuses to build IS rejected
    bad = dict(auto_loop.CAMPAIGN, GTRADE_AR_SCORE_BASIS="raw",
               GTRADE_AR_SCREEN="0", GTRADE_AR_ILLUM="full")
    assert auto_loop.campaign_problems(bad)


# --- what an unattended A/B is allowed to pick ------------------------------

def test_auto_picks_ignores_an_elite_that_never_cleared_the_gate():
    ref = {"label": "base", "sig": None}
    pool = [_cand("s1", "e1"), _cand("s2", "e2")]
    gates = {"s1": _gate(adoptable=False)}
    # s1 was gated and failed, s2 never reached a gate: neither is testable.
    assert ab_build.auto_picks(pool, ref, gates, set()) == []
    # positive control: the same pool with a passing gate does yield s1
    assert [c["sig"] for c in
            ab_build.auto_picks(pool, ref, {"s1": _gate()}, set())] == ["s1"]


def test_auto_picks_skips_what_was_already_measured_against_this_reference():
    ref = {"label": "base", "sig": None}
    pool = [_cand("s1", "e1")]
    assert ab_build.auto_picks(pool, ref, {"s1": _gate()}, {"s1"}) == []


def test_auto_picks_never_measures_the_reference_against_itself():
    ref = {"label": "adopted:A", "sig": "s1"}
    pool = [_cand("s1", "e1")]
    assert ab_build.auto_picks(pool, ref, {"s1": _gate()}, set()) == []


def test_auto_picks_ranks_replicated_candidates_first_and_caps_the_run():
    ref = {"label": "base", "sig": None}
    pool = [_cand("s%d" % i, "e%d" % i) for i in range(1, 6)]
    gates = {"s1": _gate(clears=0), "s2": _gate(clears=3), "s3": _gate(clears=1),
             "s4": _gate(clears=2), "s5": _gate(clears=0)}
    picks = [c["sig"] for c in ab_build.auto_picks(pool, ref, gates, set())]
    assert picks == ["s2", "s4", "s3"]
    assert len(picks) == ab_build.MAX_CANDIDATES


def test_auto_picks_deduplicates_a_genome_seen_in_several_ab_files():
    ref = {"label": "base", "sig": None}
    pool = [_cand("s1", "e1", kind="measured"), _cand("s1", "e1", kind="measured")]
    assert len(ab_build.auto_picks(pool, ref, {"s1": _gate()}, set())) == 1


def test_tested_against_ignores_runs_taken_against_another_reference(tmp_path):
    for name, ref_sig, sig in (("_ab_genomes_1.json", "REF_A", "s1"),
                               ("_ab_genomes_2.json", "REF_B", "s2")):
        (tmp_path / name).write_text(json.dumps({
            "reference_sig": ref_sig,
            "results": {"e": {"sig": sig}}}), encoding="utf-8")
    assert ab_build.tested_against("REF_A", str(tmp_path)) == {"s1"}
    assert ab_build.tested_against("REF_B", str(tmp_path)) == {"s2"}


# --- what an unattended adopt is allowed to take ----------------------------

def test_best_validated_takes_the_largest_measured_value():
    cands = [_cand("s1", "e1", validated=True, value=0.01, p=0.04),
             _cand("s2", "e2", validated=True, value=0.03, p=0.04)]
    assert adopt_genome.best_validated(cands)["sig"] == "s2"


def test_best_validated_breaks_a_tie_on_the_smaller_p():
    cands = [_cand("s1", "e1", validated=True, value=0.02, p=0.04),
             _cand("s2", "e2", validated=True, value=0.02, p=0.001)]
    assert adopt_genome.best_validated(cands)["sig"] == "s2"


def test_best_validated_never_takes_an_unvalidated_elite():
    # A search fitness is the larger number and must still lose to nothing.
    cands = [_cand("s1", "e1", validated=False, value=5.30, p=None)]
    assert adopt_genome.best_validated(cands) is None


# --- what the launcher menu is allowed to decide ----------------------------

def test_the_menu_can_override_a_campaign_default():
    """The proposer and the wiki are asked about by run_gtrade.bat AND live in
    CAMPAIGN. An update() here silently discarded the answer."""
    env = auto_loop.build_env({"GTRADE_AR_PROPOSER": "llm", "GTRADE_AR_WIKI": "0"})
    assert env["GTRADE_AR_PROPOSER"] == "llm"
    assert env["GTRADE_AR_WIKI"] == "0"


def test_every_campaign_key_is_present_even_when_unset():
    # Presence is the protection: a key MISSING from the child's environment is
    # refilled from .env by load_dotenv, which is how settings used to leak in.
    env = auto_loop.build_env({})
    for key in auto_loop.CAMPAIGN:
        assert key in env, key
    assert env["GTRADE_AR_SCORE_BASIS"] == "net_auc"


def test_an_explicit_budget_beats_both():
    env = auto_loop.build_env({"AR_BUDGET": "99"}, budget=7)
    assert env["AR_BUDGET"] == "7"


def test_a_menu_answer_cannot_quietly_move_a_frozen_constant():
    # It can be set, but the freeze check is what stops the run using it.
    env = auto_loop.build_env({"GTRADE_AR_OBJECTIVE": "cvar"})
    assert env["GTRADE_AR_OBJECTIVE"] == "cvar"
    assert auto_loop.freeze_problems({"GTRADE_AR_SCORE_BASIS": "net_auc",
                                      "GTRADE_AR_OBJECTIVE": "mean"}, env)


# --- the stage the cycle publishes ------------------------------------------

def test_publish_records_the_stage_and_read_state_returns_it(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_loop, "STATE_PATH", str(tmp_path / "_auto_loop.json"))
    state = {"campaign": {"GTRADE_AR_SCORE_BASIS": "net_auc"}, "history": []}
    auto_loop.publish(state, "ab_run", "reference adopted:A", cycle=3)
    out = auto_loop.read_state()
    assert out["current"]["phase"] == "ab_run"
    assert out["current"]["cycle"] == 3
    assert out["current"]["detail"] == "reference adopted:A"
    assert out["campaign"] == {"GTRADE_AR_SCORE_BASIS": "net_auc"}


def test_read_state_on_a_missing_file_reads_as_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_loop, "STATE_PATH", str(tmp_path / "absent.json"))
    assert auto_loop.read_state()["current"] is None


def test_a_killed_run_does_not_keep_reporting_itself_as_running(tmp_path,
                                                                monkeypatch):
    """The exit path that writes "stopped" is a finally block, and a closed
    console window never runs one. Observed on 2026-08-17."""
    monkeypatch.setattr(auto_loop, "STATE_PATH", str(tmp_path / "_auto_loop.json"))
    state = {"campaign": None, "history": []}
    auto_loop.publish(state, "search", "reference base", cycle=1)
    monkeypatch.setattr(auto_loop.runlock, "_alive", lambda pid: False)
    cur = auto_loop.read_state()["current"]
    assert cur["phase"] == "stopped"
    assert "resume" in cur["detail"]
    # positive control: a live owner still reads as running
    monkeypatch.setattr(auto_loop.runlock, "_alive", lambda pid: True)
    assert auto_loop.read_state()["current"]["phase"] == "search"
    # and an unknown owner (no psutil) is not called dead on a guess
    monkeypatch.setattr(auto_loop.runlock, "_alive", lambda pid: None)
    assert auto_loop.read_state()["current"]["phase"] == "search"


def test_net_training_stays_on_one_slot():
    """Assets carry different sequence lengths, and training two at once handed
    models the wrong one: 27 genomes and six hours of GPU produced nothing on
    2026-08-17. Raising this again needs the concurrency bug found first."""
    assert auto_loop.CAMPAIGN["GTRADE_NEURAL_SLOTS"] == "1"


def test_the_safe_retry_actually_differs_from_the_campaign():
    # Otherwise the retry is a second identical attempt dressed up as a recovery.
    assert any(auto_loop.CAMPAIGN.get(k) != v
               for k, v in auto_loop.SAFE_LOAD.items())
    # and only the phases that actually train are worth retrying
    assert set(auto_loop.TRAINING_PHASES) == {"search", "ab_run"}


def test_the_training_chunk_does_not_starve_a_worker():
    # 6 workers are derived on this GPU; a chunk below that idles a slot for the
    # whole run, which is the one thing this size is chosen to avoid.
    assert int(auto_loop.CAMPAIGN["GTRADE_AR_TRAIN_CHUNK"]) >= 6


# --- stopping and resuming --------------------------------------------------

def test_a_stop_request_is_visible_and_clearable(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_loop, "STOP_PATH", str(tmp_path / "_auto_loop.stop"))
    assert not auto_loop.stop_requested()
    auto_loop.request_stop("because")
    assert auto_loop.stop_requested()
    auto_loop.clear_stop()
    assert not auto_loop.stop_requested()
    # clearing an absent request is not an error: a fresh start always clears
    auto_loop.clear_stop()


def test_the_adoption_report_carries_the_genome_and_its_caveat():
    rec = {"label": "2_4_1", "adopted": "2026-08-17",
           "evidence": {"kind": "measured", "value": 0.0212, "p": 0.0031,
                        "n": 14, "holdout": "A,B,C", "floor": 0.005,
                        "caveat": "measured on 3 held-out assets it had not seen"},
           "genome": {"drops": ["vol_z"], "extra": [], "label_mode": "direction",
                      "thr_margin": 0.02}}
    text = "\n".join(adopt_genome.report_lines(rec))
    # the effect must not round away: the floor here is 0.005
    assert "+0.0212" in text and "0.0031" in text
    assert "vol_z" in text and "thr_margin" in text
    assert "caveat" in text
    # and the raw genome is reproduced, so a retrain can be set up from the report
    assert '"thr_margin": 0.02' in text


def test_the_adoption_report_says_so_when_nothing_is_adopted():
    assert "Nothing adopted" in adopt_genome.report_lines(None)[0]


def test_chunk_subsets_splits_in_order_and_keeps_every_asset():
    assert ar.chunk_subsets("A,B,C,D,E", 2) == ["A,B", "C,D", "E"]
    # 0 and "bigger than the subset" both mean one training process
    assert ar.chunk_subsets("A,B,C", 0) == ["A,B,C"]
    assert ar.chunk_subsets("A,B,C", 9) == ["A,B,C"]
    assert ar.chunk_subsets("", 3) == []


def test_unchunked_training_still_trains_the_subset_in_one_call(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_TRAIN_CHUNK", raising=False)
    calls = []
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        calls.append(sub) or [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    ar._cached_train("A,B,C,D", {}, lambda sub: "k:" + sub, "t")
    assert calls == ["A,B,C,D"]


def test_switching_chunking_on_still_reads_an_arm_cached_before_it(monkeypatch):
    """Turning on the safety net must not discard the work it exists to protect."""
    key_of = (lambda sub: "k:" + sub)
    calls = []
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        calls.append(sub) or [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    monkeypatch.delenv("GTRADE_AR_TRAIN_CHUNK", raising=False)
    ar._cached_train("A,B,C,D", {}, key_of, "t")     # cached whole, pre-chunking
    assert calls == ["A,B,C,D"]

    calls.clear()
    monkeypatch.setenv("GTRADE_AR_TRAIN_CHUNK", "2")
    rows = ar._cached_train("A,B,C,D", {}, key_of, "t")
    assert calls == []
    assert [r["Asset"] for r in rows] == ["A", "B", "C", "D"]


def test_chunked_training_resumes_where_an_interruption_left_it(monkeypatch):
    """The point of chunking: an interrupted arm costs one chunk, not the arm."""
    monkeypatch.setenv("GTRADE_AR_TRAIN_CHUNK", "2")
    key_of = (lambda sub: "k:" + sub)
    calls = []

    def dies_on_the_third_chunk(sub, env):
        calls.append(sub)
        if sub == "E,F":
            raise RuntimeError("killed mid-chunk")
        return [{"Asset": a, "Score": 1.0} for a in sub.split(",")]

    monkeypatch.setattr(ar, "_train_once", dies_on_the_third_chunk)
    try:
        ar._cached_train("A,B,C,D,E,F", {}, key_of, "t")
        raise AssertionError("the fake trainer was supposed to die")
    except RuntimeError:
        pass
    assert calls == ["A,B", "C,D", "E,F"]

    calls.clear()
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        calls.append(sub) or [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    rows = ar._cached_train("A,B,C,D,E,F", {}, key_of, "t")
    # only the chunk that was in flight is retrained, and the rows are whole
    assert calls == ["E,F"]
    assert [r["Asset"] for r in rows] == ["A", "B", "C", "D", "E", "F"]


# --- parallel chunk training (process-level, never thread-level) -------------

def test_split_load_divides_what_is_sized_against_the_whole_box(monkeypatch):
    monkeypatch.setenv("GTRADE_TF_POOL_PCT", "0.50")
    monkeypatch.setenv("GTRADE_CB_THREADS", "12")
    one = ar.split_load({}, 1)
    assert one == {}, "a single job must change nothing"
    two = ar.split_load({}, 2)
    assert two["GTRADE_TF_POOL_PCT"] == "0.25"
    assert two["GTRADE_CB_THREADS"] == "6"


def test_split_load_never_allows_a_second_neural_slot(monkeypatch):
    """Parallelism comes from processes now. A second slot inside one process is
    what handed models the wrong sequence length and emptied 27 genomes."""
    monkeypatch.delenv("GTRADE_TF_POOL_PCT", raising=False)
    out = ar.split_load({"GTRADE_NEURAL_SLOTS": "4"}, 2)
    assert out["GTRADE_NEURAL_SLOTS"] == "1"


def test_only_the_first_parallel_chunk_writes_the_real_progress_files():
    # Two writers would make the per-asset ETA on the research page a lie.
    first = ar.split_load({}, 2, progress_dir=None)
    rest = ar.split_load({}, 2, progress_dir="/scratch/x")
    assert "AR_PROGRESS_DIR" not in first
    assert rest["AR_PROGRESS_DIR"] == "/scratch/x"


def test_chunks_train_in_parallel_and_the_rows_stay_in_order(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_TRAIN_CHUNK", "2")
    monkeypatch.setenv("GTRADE_AR_TRAIN_JOBS", "2")
    # train_jobs asks the card before agreeing to two processes, which is the
    # point of it. A test about ORDERING must not also depend on how much VRAM
    # this particular box happens to have spare, so the card is stubbed roomy.
    monkeypatch.setattr(ar, "free_vram_mb", lambda: 4096)
    seen = []
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        seen.append((sub, env.get("GTRADE_TF_POOL_PCT"))) or
        [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    rows = ar._cached_train("A,B,C,D", {}, lambda s: "k:" + s, "t")
    assert [r["Asset"] for r in rows] == ["A", "B", "C", "D"]
    assert sorted(s for s, _ in seen) == ["A,B", "C,D"]
    # and each process was handed a divided share, not the whole card
    assert all(pool is not None for _, pool in seen)


def test_one_job_keeps_the_sequential_path_byte_identical(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_TRAIN_CHUNK", "2")
    monkeypatch.setenv("GTRADE_AR_TRAIN_JOBS", "1")
    seen = []
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        seen.append((sub, dict(env))) or
        [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    ar._cached_train("A,B,C,D", {}, lambda s: "k:" + s, "t")
    assert [s for s, _ in seen] == ["A,B", "C,D"]
    assert all(e == {} for _, e in seen), "no load splitting when jobs=1"


def test_the_safe_retry_also_drops_back_to_one_training_process():
    """Two processes are the campaign default again, and the retry still only
    narrows.

    They peaked at 3956 MiB of a 4096 MiB card and stalled for 15000 s on the
    140 MiB that left. Two things made that possible and both are fixed: the
    pool was sized from memory.TOTAL rather than memory.free, and its 1024 MiB
    floor made the percentage unshrinkable, so lowering it changed nothing. At
    a 0.34 pool each process now takes about 696 MiB and the projected peak
    leaves roughly 800 MiB spare, and auto_research.gpu_fit_jobs refuses the
    second process outright when the card cannot actually take it."""
    campaign = int(auto_loop.CAMPAIGN["GTRADE_AR_TRAIN_JOBS"])
    safe = int(auto_loop.SAFE_LOAD["GTRADE_AR_TRAIN_JOBS"])
    assert campaign == 2
    assert safe <= campaign


def test_the_chunk_cap_still_splits_a_small_unit_across_the_jobs():
    """The bug this closes: chunk 7 never split the 4-asset search unit, so the
    two-process path was dead exactly where the loop spends its time."""
    tier, holdout = "A,B,C,D", ",".join("ABCDEFGHIJKLMN")
    assert ar.effective_chunk_size(tier, size=7, jobs=2) == 2
    assert len(ar.chunk_subsets(tier, 2)) == 2
    # the cap still wins where it is the smaller of the two
    assert ar.effective_chunk_size(holdout, size=7, jobs=2) == 7
    assert len(ar.chunk_subsets(holdout, 7)) == 2


def test_one_job_leaves_the_chunk_cap_exactly_as_configured():
    assert ar.effective_chunk_size("A,B,C,D", size=7, jobs=1) == 7
    # and chunking off stays off, jobs or not
    assert ar.effective_chunk_size("A,B,C,D", size=0, jobs=4) == 0


def test_the_search_path_itself_splits_not_only_the_cached_one(monkeypatch):
    monkeypatch.setattr(ar, "free_vram_mb", lambda: 4096)
    """The miss this covers: parallelism lived in _cached_train, but run_axis and
    run_qd hand `train_env` to the trainer directly, so every candidate - the
    hours of the run - went past the split while only the base was parallel."""
    monkeypatch.setenv("GTRADE_AR_TRAIN_CHUNK", "7")
    monkeypatch.setenv("GTRADE_AR_TRAIN_JOBS", "2")
    seen = []
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        seen.append((sub, env.get("GTRADE_TF_POOL_PCT"))) or
        [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    # the tier unit a search step trains: 4 assets, chunk cap 7
    rows = ar.train_env("SP500,BTC,EURUSD,GOLD", {})
    assert sorted(s for s, _ in seen) == ["EURUSD,GOLD", "SP500,BTC"], seen
    assert [r["Asset"] for r in rows] == ["SP500", "BTC", "EURUSD", "GOLD"]
    assert all(pool is not None for _, pool in seen), "load was not divided"


def test_an_already_chunked_caller_is_not_split_again(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_TRAIN_CHUNK", "7")
    monkeypatch.setenv("GTRADE_AR_TRAIN_JOBS", "2")
    seen = []
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        seen.append(sub) or [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    ar.train_env("SP500,BTC,EURUSD,GOLD", {}, split=False)
    assert seen == ["SP500,BTC,EURUSD,GOLD"]


def test_one_job_leaves_the_search_path_on_a_single_process(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_TRAIN_CHUNK", "7")
    monkeypatch.setenv("GTRADE_AR_TRAIN_JOBS", "1")
    seen = []
    monkeypatch.setattr(ar, "_train_once", lambda sub, env: (
        seen.append((sub, dict(env))) or [{"Asset": a, "Score": 1.0} for a in sub.split(",")]))
    ar.train_env("SP500,BTC,EURUSD,GOLD", {})
    assert seen == [("SP500,BTC,EURUSD,GOLD", {})], "byte-identical at one job"


def test_a_hand_started_phase_borrows_the_campaigns_load_profile():
    """The parallel training path lives entirely in these env keys, so a phase
    started by hand ran one sequential process where the loop ran two."""
    env = {}
    took = auto_loop.apply_load_profile(env)
    assert env["GTRADE_AR_TRAIN_JOBS"] == auto_loop.CAMPAIGN["GTRADE_AR_TRAIN_JOBS"]
    assert env["GTRADE_AR_TRAIN_CHUNK"] == auto_loop.CAMPAIGN["GTRADE_AR_TRAIN_CHUNK"]
    assert set(took) == set(auto_loop.LOAD_KEYS)


def test_an_explicit_load_setting_outranks_the_profile():
    env = {"GTRADE_AR_TRAIN_JOBS": "1"}
    took = auto_loop.apply_load_profile(env)
    assert env["GTRADE_AR_TRAIN_JOBS"] == "1"
    assert "GTRADE_AR_TRAIN_JOBS" not in took


def test_a_hand_started_ab_judges_on_the_campaigns_frozen_basis(tmp_path, monkeypatch):
    """The floor, the re-keying and the verdict all follow the basis, so a run
    that defaults it measures a different quantity than the campaign froze."""
    monkeypatch.setattr(auto_loop, "STATE_PATH", str(tmp_path / "_auto_loop.json"))
    (tmp_path / "_auto_loop.json").write_text(json.dumps(
        {"campaign": {"GTRADE_AR_SCORE_BASIS": "net_auc",
                      "GTRADE_AR_OBJECTIVE": "mean"}}), encoding="utf-8")
    env = {}
    took = auto_loop.apply_campaign_basis(env)
    assert env["GTRADE_AR_SCORE_BASIS"] == "net_auc"
    assert took == {"GTRADE_AR_SCORE_BASIS": "net_auc",
                    "GTRADE_AR_OBJECTIVE": "mean"}


def test_no_campaign_leaves_the_basis_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_loop, "STATE_PATH", str(tmp_path / "nothing.json"))
    env = {}
    assert auto_loop.apply_campaign_basis(env) == {}
    assert env == {}


# --- what a finished cycle records ------------------------------------------

def test_the_history_records_what_the_director_chose(tmp_path, monkeypatch):
    """Without the settings there is nothing to attach a reward to, and an old
    entry that predates them must still read rather than crash."""
    monkeypatch.setattr(auto_loop, "STATE_PATH", str(tmp_path / "s.json"))
    state = {"campaign": None, "history": [{"ts": "old", "action": "search",
                                            "rc": 0, "cycle": 1}]}
    auto_loop.record_cycle(state, action="search", rc=0, cycle=2, seconds=93.5,
                           settings={"GTRADE_AR_AXES": "hyper"}, chosen_by="rl")
    top = state["history"][0]
    assert top["action"] == "search" and top["cycle"] == 2
    assert top["settings"] == {"GTRADE_AR_AXES": "hyper"}
    assert top["chosen_by"] == "rl" and top["seconds"] == 93.5
    # the legacy entry survives untouched
    assert state["history"][1] == {"ts": "old", "action": "search",
                                   "rc": 0, "cycle": 1}
    # and a cycle with no director records that honestly, not as a guess
    auto_loop.record_cycle(state, action="ab_run", rc=0, cycle=3, seconds=1.0,
                           settings=None, chosen_by=None)
    assert state["history"][0]["chosen_by"] is None
    assert len(state["history"]) == 3


def test_the_llm_stays_the_default_director(monkeypatch):
    """Positive control on the routing: an unset mode must not reach the RL
    path at all."""
    monkeypatch.delenv("GTRADE_AR_DIRECTOR_MODE", raising=False)
    monkeypatch.setenv("GTRADE_AR_DIRECTOR", "1")
    seen = {}

    def _propose(*_a, **_k):
        seen["llm"] = True          # returns None: the director declined

    monkeypatch.setattr("core.ar_director.propose", _propose)
    _env, _state, chose = auto_loop.apply_director(_clean_env(),
                                                   {"history": []})
    assert seen == {"llm": True}
    assert chose["by"] is None          # propose returned None, nothing applied


# --- what an A/B could have seen ---------------------------------------------

def test_an_underpowered_run_is_named_as_one_not_reported_as_a_null():
    """A FAILED verdict is two results wearing one word. The campaign's own
    holdout carries a per-asset spread near 3.74 in Score units, so 14 assets
    can only resolve about +2.2 against a floor of +0.5, and every A/B that
    said FAILED was saying "not measurable" without the word."""
    st = {"n": 14, "sd": 3.74, "p": 0.6, "value": 0.1}
    pw = ab_build.power(st, floor=0.5)
    assert pw["powered"] is False
    assert 2.0 < pw["mde"] < 2.6
    assert pw["n_needed"] > 300
    assert "UNDERPOWERED" in ab_build.power_tag(st, 0.5)


def test_a_run_that_could_resolve_the_floor_says_so():
    """The other end of the control: with a tight spread the same code has to
    report the run as powered, or the label means nothing."""
    st = {"n": 200, "sd": 1.0, "p": 0.2, "value": 0.1}
    pw = ab_build.power(st, floor=0.5)
    assert pw["powered"] is True and pw["mde"] < 0.5
    assert "UNDERPOWERED" not in ab_build.power_tag(st, 0.5)


def test_power_says_nothing_rather_than_guessing_without_a_spread():
    assert ab_build.power({"n": 0, "sd": None}, 0.5)["mde"] is None
    assert ab_build.power({"n": 1, "sd": 0.0}, 0.5)["mde"] is None
    assert ab_build.power_tag({"n": 0, "sd": None}, 0.5) == ""


def test_the_projection_refuses_to_guess_without_a_measured_spread(tmp_path):
    """No recorded spread means no projection, not an invented one: the runs
    before 2026-08-21 did not keep the number."""
    assert ab_build.last_spread(str(tmp_path)) is None
    assert ab_build.projected_power(14, 0.5, str(tmp_path)) == ""


def test_the_projection_names_a_holdout_that_cannot_answer(tmp_path, monkeypatch):
    # The spread was recorded by a single-training run, so the projection may
    # only be read back by one. last_spread refuses to mix the two.
    monkeypatch.setenv("GTRADE_AB_SEEDS", "1")
    (tmp_path / "_ab_genomes_20260821-1200.json").write_text(json.dumps(
        {"results": {"c": {"sd_raw": 3.74}}}), encoding="utf-8")
    assert ab_build.last_spread(str(tmp_path)) == 3.74
    small = ab_build.projected_power(14, 0.5, str(tmp_path))
    assert "cannot answer" in small and "would be needed" in small
    # and the other end: a floor the same holdout CAN resolve reads as clear
    big = ab_build.projected_power(14, 5.0, str(tmp_path))
    assert "clears the floor" in big


def test_an_unset_illumination_is_not_a_problem_on_any_basis():
    """campaign_problems must read the same default the run will use.

    It defaulted GTRADE_AR_ILLUM to "cb" while auto_research now derives it from
    the basis. Left as it was, a net-basis campaign that simply never sets the
    variable would be refused for a CatBoost illumination it is not going to do.
    """
    for basis in ("net_auc", "net_gain", "ens_auc", "raw", "neural"):
        env = {"GTRADE_AR_SCORE_BASIS": basis, "GTRADE_AR_SCREEN": "0"}
        assert auto_loop.campaign_problems(env) == [], basis
    # and the contradiction it exists for is still caught
    assert auto_loop.campaign_problems(
        {"GTRADE_AR_SCORE_BASIS": "net_auc", "GTRADE_AR_SCREEN": "0",
         "GTRADE_AR_ILLUM": "cb"})
