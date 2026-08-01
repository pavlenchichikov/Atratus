"""The agent publishes phase and step, and folds measured durations into history."""

import datetime

import auto_research
from core import ar_progress


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "ar_progress.json"))
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "ar_progress_unit.json"))


def test_publish_writes_phase_step_and_seeds_history(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("search", step={"i": 3, "n": 15, "kind": "screen"})
    rec = ar_progress.read_agent()
    assert rec["phase"] == "search"
    assert rec["step"] == {"i": 3, "n": 15, "kind": "screen"}
    # The measured unit-kind history from the 2026-07-23 run is seeded, so the very
    # first run after this change already has a basis for an estimate.
    assert rec["history"]["holdout_14"]
    assert rec["history"]["assets"] == {}


def test_publish_does_not_mutate_the_progress_seed_constant(monkeypatch, tmp_path):
    """dict(PROGRESS_SEED) is a SHALLOW copy: history["assets"] would be the exact
    same nested dict object as PROGRESS_SEED["assets"], so folding into a fresh
    progress file would permanently poison the seed for the rest of the process
  ."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("gate", step={"i": 1, "n": 1, "unit_kind": "holdout_14"})
    ar_progress.write_unit({"done": [["USDJPY", 7200]]})
    auto_research._progress_fold_unit("holdout_14", 9000)
    assert auto_research.PROGRESS_SEED["assets"] == {}


def test_publish_keeps_history_across_calls(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("search", step={"i": 1, "n": 15})
    auto_research._progress_fold_unit("holdout_14", 12345)
    auto_research._progress_publish("gate", step={"i": 1, "n": 3})
    rec = ar_progress.read_agent()
    assert 12345 in rec["history"]["holdout_14"]
    assert rec["phase"] == "gate"


def test_fold_unit_absorbs_per_asset_service_times(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("gate", step={"i": 1, "n": 3})
    ar_progress.write_unit({"done": [["USDJPY", 7200], ["NASDAQ", 600]]})
    auto_research._progress_fold_unit("holdout_14", 9000)
    hist = ar_progress.read_agent()["history"]
    assert hist["assets"]["holdout_14"]["USDJPY"] == [7200]
    assert hist["assets"]["holdout_14"]["NASDAQ"] == [600]
    assert 9000 in hist["holdout_14"]


def test_history_lists_are_capped(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish("gate", step={"i": 1, "n": 3})
    for i in range(30):
        auto_research._progress_fold_unit("tier_4", 1000 + i)
    kept = ar_progress.read_agent()["history"]["tier_4"]
    assert len(kept) <= auto_research.PROGRESS_KEEP
    assert 1029 in kept        # the newest measurement survives


def test_publish_never_raises_when_storage_fails(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(ar_progress, "write_agent", lambda *_a, **_k: (_ for _ in ()).throw(OSError("x")))
    auto_research._progress_publish("search", step={"i": 1, "n": 15})   # must return


def test_legacy_flat_assets_history_is_discarded_not_migrated(monkeypatch, tmp_path):
    """An earlier build left a FLAT history["assets"] on disk
    (asset name straight to a list of numbers). Those samples are exactly the
    poisoned, cross-population contamination the per-unit-kind keying exists to
    prevent, so they must be dropped rather than carried forward under some
    guessed unit_kind."""
    _isolate(monkeypatch, tmp_path)
    ar_progress.write_agent({
        "phase": "gate",
        "history": {"holdout_14": [30000], "assets": {"USDJPY": [14], "AIRBUS": [11]}},
    })
    auto_research._progress_publish("gate", step={"i": 1, "n": 1})
    hist = ar_progress.read_agent()["history"]
    assert hist["assets"] == {}


def _run_qd_smoke(monkeypatch, tier_env):
    """Drive a real run_qd with a fake trainer that emulates the SEAM a real
    trainer subprocess exercises: unit_begin/unit_asset_start/unit_asset_done/
    unit_end, with CatBoost-only (GTRADE_SCREEN_ONLY) evaluations running much
    faster than full-ensemble ones - the same shape every real train_hybrid.py
    subprocess produces. A fake clock (ar_progress._now patched) advances by a
    fixed per-asset amount between start and done so the simulated durations are
    deterministic and the test does not actually sleep for hours."""
    monkeypatch.setattr(auto_research, "_qd_load", dict)
    monkeypatch.setattr(auto_research, "_qd_save", lambda a: None)
    monkeypatch.setattr(auto_research, "BUDGET", 2, raising=False)
    monkeypatch.setenv("GTRADE_AR_QD_INIT", "2")
    monkeypatch.setenv("GTRADE_AR_QD_FINAL", "1")
    monkeypatch.setenv("GTRADE_AR_TIER", tier_env)
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    import random as _r
    _r.seed(0)

    clock = {"t": datetime.datetime(2026, 1, 1)}
    monkeypatch.setattr(ar_progress, "_now", lambda: clock["t"])

    FULL_STEP = datetime.timedelta(seconds=3700)   # full ensemble: hour-scale
    SCREEN_STEP = datetime.timedelta(seconds=8)    # CatBoost-only: second-scale

    def fake_train(subset, env):
        n_drop = len([d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d])
        s = 1.0 + 0.7 * n_drop
        assets = subset.split(",")
        step = SCREEN_STEP if env.get("GTRADE_SCREEN_ONLY") else FULL_STEP
        ar_progress.unit_begin(assets, 4)
        for a in assets:
            ar_progress.unit_asset_start(a)
            clock["t"] += step
            ar_progress.unit_asset_done(a)
        ar_progress.unit_end()
        return [{"Asset": a, "Score": s} for a in assets]

    calls = []
    real_write_agent = ar_progress.write_agent

    def spy_write_agent(payload):
        calls.append(dict(payload))
        real_write_agent(payload)

    monkeypatch.setattr(ar_progress, "write_agent", spy_write_agent)
    auto_research.run_qd(train_fn=fake_train)
    return calls


def test_seam_gate_unit_never_reads_the_trailing_cb_trains_asset_times(monkeypatch, tmp_path):
    """The seam between auto_research.py and core/ar_progress.py: _heldout_eval
    runs a full-ensemble train then a CatBoost-only (screen) train over the SAME
    held-out assets, and the CB train - always second - is what a naive
    "read the unit file after both finish" fold would absorb into
    history["assets"]["holdout_14"]. A fake_train
    that never calls unit_begin/unit_asset_* cannot exercise this at all, which is
    the seam where the two files meet - so this test drives a REAL run_qd
    through a fake_train that does call them, with screen-scale durations
    orders of magnitude smaller than full-scale ones, and asserts a holdout
    estimate can never read a screen-scale sample."""
    _isolate(monkeypatch, tmp_path)
    _run_qd_smoke(monkeypatch, "0")   # tier off: isolate to the holdout_14 path
    hist = ar_progress.read_agent()["history"]
    holdout_assets = hist.get("assets", {}).get("holdout_14", {})
    assert holdout_assets, "expected per-asset holdout_14 samples to have been recorded"
    for asset, times in holdout_assets.items():
        assert all(t >= 1000 for t in times), (
            "holdout_14 per-asset history contaminated with a screen-scale (CB-only) "
            f"time: {asset}={times} (expected only full-ensemble, hour-scale samples)")


def test_tier_check_folds_a_tier_4_measurement_when_tier_is_on(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    _run_qd_smoke(monkeypatch, "1")
    hist = ar_progress.read_agent()["history"]
    # a real tier check ran, so a fresh measurement landed past the seeded four
    assert len(hist["tier_4"]) > len(auto_research.PROGRESS_SEED["tier_4"])


def test_tier_off_never_folds_tier_4_or_pends_it(monkeypatch, tmp_path):
    """GTRADE_AR_TIER=0 is a supported configuration: no tier check ever runs,
    so neither a tier_4 measurement nor a tier_4 pending unit should appear in
    any published record - a near-zero tier_4 fold would drag its median
    toward zero and make run_remaining under-estimate the time left."""
    _isolate(monkeypatch, tmp_path)
    calls = _run_qd_smoke(monkeypatch, "0")
    hist = ar_progress.read_agent()["history"]
    # no new measurement: the list is exactly the untouched seed
    assert hist["tier_4"] == list(auto_research.PROGRESS_SEED["tier_4"])
    # no call ever advertised a tier_4 unit as pending
    assert all("tier_4" not in (c.get("pending_units") or []) for c in calls)
    # sanity: the gate phase actually ran (the assertions above are not vacuous)
    assert any(c.get("phase") == "gate" for c in calls)


def test_search_step_publishes_unit_kind_and_gets_an_estimate(monkeypatch, tmp_path):
    """Search and
    warmup steps published no unit_kind, so unit_remaining() had nothing to
    look up and the entire search phase read "no history yet" even though
    screen_10 was seeded, leaving no ETA for the phase that runs first."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "search", step={"i": 1, "n": 15, "kind": "screen", "unit_kind": "screen_10"})
    ar_progress.write_unit({"order": ["SP500"], "workers": 1, "done": []})
    snap = ar_progress.snapshot()
    assert snap["eta"]["basis"] == "unit-kind median, no per-asset history yet"
    assert snap["eta"]["unit_left_s"] is not None


def test_fold_screen_unit_keeps_its_own_asset_bucket_separate_from_holdout(monkeypatch, tmp_path):
    """A screen unit trains CatBoost only, so its assets finish in seconds; a
    holdout/tier (gate) unit trains the full ensemble and takes hours for the
    same asset. Now that history["assets"] is keyed by unit_kind, a screen
    unit's per-asset times land in their OWN bucket (history["assets"]["screen_10"])
    where a holdout estimate - which only ever reads
    history["assets"]["holdout_14"] - can never see them; that structural
    separation is what replaced the old fold_assets=False flag."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "search", step={"i": 1, "n": 15, "kind": "screen", "unit_kind": "screen_10"})
    before = len(ar_progress.read_agent()["history"]["screen_10"])
    ar_progress.write_unit({"done": [["SP500", 12], ["NVDA", 9]]})
    auto_research._progress_fold_unit("screen_10", 45)
    hist = ar_progress.read_agent()["history"]
    assert len(hist["screen_10"]) == before + 1
    assert 45 in hist["screen_10"]
    assert hist["assets"]["screen_10"] == {"SP500": [12], "NVDA": [9]}
    assert "holdout_14" not in hist["assets"]


def test_fold_gate_unit_absorbs_per_asset_times_under_its_own_kind(monkeypatch, tmp_path):
    """A gate unit (tier_4/holdout_14) trains the full ensemble, so its assets'
    own service times are exactly what the per-asset ETA needs - unlike a
    screen unit's CatBoost-only seconds - and they land under history["assets"]
    ["holdout_14"], not a flat top-level bucket."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "gate", step={"i": 1, "n": 1, "kind": "elite_holdout", "unit_kind": "holdout_14"})
    ar_progress.write_unit({"done": [["USDJPY", 7200]]})
    auto_research._progress_fold_unit("holdout_14", 9000)
    hist = ar_progress.read_agent()["history"]
    assert hist["assets"]["holdout_14"]["USDJPY"] == [7200]


def test_fold_skips_entirely_on_a_cache_hit_since_marker_unchanged(monkeypatch, tmp_path):
    """A cache hit means no unit_begin ran, so the unit file's 'started' stamp is
    unchanged: the fold must be skipped ENTIRELY - neither the wall time nor any
    per-asset time - because folding it would record a near-zero wall time and
    re-absorb the PREVIOUS unit's per-asset samples under a fresh label
  ."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "gate", step={"i": 1, "n": 1, "kind": "elite_holdout", "unit_kind": "holdout_14"})
    ar_progress.write_unit({"started": "2026-01-01T00:00:00", "done": [["USDJPY", 7200]]})
    auto_research._progress_fold_unit("holdout_14", 7200)   # since not given: real fold
    hist = ar_progress.read_agent()["history"]
    assert hist["assets"]["holdout_14"]["USDJPY"] == [7200]
    before_wall = list(hist["holdout_14"])

    # Simulate a cache hit around the NEXT evaluation: no new unit_begin runs, so
    # 'started' stays the same, but the caller still measured ~0.05s of wall time
    # around the cache lookup.
    mark = ar_progress.read_unit().get("started")
    auto_research._progress_fold_unit("holdout_14", 0.05, since=mark)
    hist_after = ar_progress.read_agent()["history"]
    assert hist_after["holdout_14"] == before_wall            # no new wall-time sample
    assert hist_after["assets"]["holdout_14"]["USDJPY"] == [7200]   # not duplicated, no 0 added


def test_fold_floors_a_sub_second_wall_time_instead_of_recording_zero(monkeypatch, tmp_path):
    """Belt-and-suspenders on top of the since= cache-hit guard: even when a fold
    does proceed, int(0.05) truncates to 0 and PROGRESS_KEEP=12 means twelve such
    folds would drag a median to zero - so the wall time is floored to at least
    one second, never zero."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "gate", step={"i": 1, "n": 1, "kind": "elite_holdout", "unit_kind": "holdout_14"})
    auto_research._progress_fold_unit("holdout_14", 0.05)
    hist = ar_progress.read_agent()["history"]
    assert hist["holdout_14"][-1] >= 1
