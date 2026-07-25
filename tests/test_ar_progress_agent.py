"""The agent publishes phase and step, and folds measured durations into history."""

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
    assert hist["assets"]["USDJPY"] == [7200]
    assert hist["assets"]["NASDAQ"] == [600]
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


def _run_qd_smoke(monkeypatch, tier_env):
    """Drive a real run_qd with a cheap fake trainer, the same pattern
    test_run_qd_illuminates_and_gates in test_auto_research.py uses."""
    monkeypatch.setattr(auto_research, "_qd_load", lambda: {})
    monkeypatch.setattr(auto_research, "_qd_save", lambda a: None)
    monkeypatch.setattr(auto_research, "BUDGET", 2, raising=False)
    monkeypatch.setenv("GTRADE_AR_QD_INIT", "2")
    monkeypatch.setenv("GTRADE_AR_QD_FINAL", "1")
    monkeypatch.setenv("GTRADE_AR_TIER", tier_env)
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    import random as _r
    _r.seed(0)

    def fake_train(subset, env):
        n_drop = len([d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d])
        s = 1.0 + 0.7 * n_drop
        return [{"Asset": a, "Score": s} for a in subset.split(",")]

    calls = []
    real_write_agent = ar_progress.write_agent

    def spy_write_agent(payload):
        calls.append(dict(payload))
        real_write_agent(payload)

    monkeypatch.setattr(ar_progress, "write_agent", spy_write_agent)
    auto_research.run_qd(train_fn=fake_train)
    return calls


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
    """Found by actually running the agent (2026-07-25 smoke test): search and
    warmup steps published no unit_kind, so unit_remaining() had nothing to
    look up and the entire search phase read "no history yet" even though
    screen_10 was seeded - the owner had no ETA for the phase that runs first."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "search", step={"i": 1, "n": 15, "kind": "screen", "unit_kind": "screen_10"})
    ar_progress.write_unit({"order": ["SP500"], "workers": 1, "done": []})
    snap = ar_progress.snapshot()
    assert snap["eta"]["basis"] == "unit-kind median, no per-asset history yet"
    assert snap["eta"]["unit_left_s"] is not None


def test_fold_screen_unit_grows_screen_history_but_not_assets(monkeypatch, tmp_path):
    """A screen unit trains CatBoost only, so its assets finish in seconds; a
    holdout/tier (gate) unit trains the full ensemble and takes hours for the
    same asset. Folding a screen unit's per-asset times into history["assets"]
    would mix those two populations and poison the per-asset median the whole
    ETA rests on, so fold_assets=False must keep them out - even though
    unit.json genuinely has "done" entries a careless unconditional fold would
    happily absorb.
    """
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "search", step={"i": 1, "n": 15, "kind": "screen", "unit_kind": "screen_10"})
    before = len(ar_progress.read_agent()["history"]["screen_10"])
    ar_progress.write_unit({"done": [["SP500", 12], ["NVDA", 9]]})
    auto_research._progress_fold_unit("screen_10", 45, fold_assets=False)
    hist = ar_progress.read_agent()["history"]
    assert len(hist["screen_10"]) == before + 1
    assert 45 in hist["screen_10"]
    assert hist["assets"] == {}


def test_fold_gate_unit_still_absorbs_per_asset_times_by_default(monkeypatch, tmp_path):
    """fold_assets defaults to True: a gate unit (tier_4/holdout_14) trains the
    full ensemble, so its assets' own service times are exactly what the
    per-asset ETA needs - unlike a screen unit's CatBoost-only seconds."""
    _isolate(monkeypatch, tmp_path)
    auto_research._progress_publish(
        "gate", step={"i": 1, "n": 1, "kind": "elite_holdout", "unit_kind": "holdout_14"})
    ar_progress.write_unit({"done": [["USDJPY", 7200]]})
    auto_research._progress_fold_unit("holdout_14", 9000)
    hist = ar_progress.read_agent()["history"]
    assert hist["assets"]["USDJPY"] == [7200]
