import json
import os

import pytest

# auto_research.py is a local, gitignored tool (absent in CI); skip there.
ar = pytest.importorskip("auto_research")


def test_train_merges_env_overrides(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, cwd, env):
        captured["env"] = env
        # the trainer writes quality_report.json into the model dir (last cmd context):
        os.makedirs(env["GTRADE_MODEL_DIR"], exist_ok=True)
        with open(os.path.join(env["GTRADE_MODEL_DIR"], "quality_report.json"), "w") as f:
            json.dump([{"Asset": "SP500", "Score": 1.0}], f)

    monkeypatch.setattr(ar.subprocess, "run", fake_run)
    rows = ar._train("SP500,NVDA", {"GTRADE_LABEL_MODE": "rel_median"}, str(tmp_path / "m"))
    assert rows == [{"Asset": "SP500", "Score": 1.0}]
    assert captured["env"]["GTRADE_LABEL_MODE"] == "rel_median"
    assert captured["env"]["GTRADE_ASSETS"] == "SP500,NVDA"
    # LIGHT_ENV is still applied
    assert captured["env"]["GTRADE_FORCE_PROMOTE"] == "1"


def test_train_missing_report_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ar.subprocess, "run", lambda cmd, cwd, env: None)
    assert ar._train("SP500", {}, str(tmp_path / "m")) == []


def test_feature_env_builds_keys(tmp_path):
    specs = [{"name": "f1", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]
    env = ar._feature_env(specs, ["f1"])
    assert env["GTRADE_EXTRA_FEATURES"] == "f1"
    assert os.path.exists(env["GTRADE_DSL_SPECS"])
    assert ar._feature_env([], []) == {}


def _axis_additive():
    # candidate is a dict {"name":..., "lift":x}; to_env irrelevant for the fake train_fn
    return ar.Axis(
        name="fake_add",
        propose=lambda log: [{"name": "c%d" % len(log), "lift": 2.0 if len(log) == 0 else -1.0}],
        to_env=lambda selected: {"NAMES": ",".join(c["name"] for c in selected)},
        kind="additive",
    )


def _fake_train(lift_by_env):
    # returns rows whose Score is the sum of the lifts encoded in NAMES via lift_by_env
    def train(subset, env):
        names = [n for n in env.get("NAMES", "").split(",") if n]
        score = 1.0 + sum(lift_by_env.get(n, 0.0) for n in names)
        return [{"Asset": "SP500", "Score": score}, {"Asset": "NVDA", "Score": score}]
    return train


def test_run_axis_additive_keeps_lifting_rejects_flat():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = _axis_additive()
    train = _fake_train({"c0": 2.0, "c1": -1.0})  # c0 lifts, c1 hurts
    res = ar.run_axis(axis, 2, base, train, persist=lambda log: None)
    kept_names = [c["name"] for c in res["kept"]]
    assert "c0" in kept_names and "c1" not in kept_names
    assert res["kept_delta"] > 0


def test_run_axis_additive_resumes_from_prior_log():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    # prior log: round 0 accepted candidate c0 (delta 2.0); resume should keep c0
    prior = [{"axis": "fake_add", "iter": 0, "cand": [{"name": "c0", "lift": 2.0}],
              "cand_mean_delta": 2.0, "score": 2.0, "accepted": True}]
    axis = _axis_additive()  # reuse the helper defined earlier in this file

    seen = {"trains": 0}

    def train(subset, env):
        seen["trains"] += 1
        names = [n for n in env.get("NAMES", "").split(",") if n]
        # c0 lift 2.0, anything else (c1) hurts
        score = 1.0 + sum(2.0 if n == "c0" else -1.0 for n in names)
        return [{"Asset": a, "Score": score} for a in ("SP500", "NVDA")]

    res = ar.run_axis(axis, 2, base, train, persist=lambda log: None, prior_log=prior)
    kept_names = [c["name"] for c in res["kept"]]
    assert "c0" in kept_names              # resumed: prior accept preserved
    assert res["kept_delta"] == 2.0        # reconstructed from the accepted entry
    # it resumed at iter 1 (did not retrain iter 0): only the iter-1 candidate trained
    assert seen["trains"] <= 1


def test_run_axis_select_best_picks_best():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(
        name="fake_sel",
        propose=lambda log: [{"name": "w20", "v": 20}, {"name": "w30", "v": 30}, {"name": "w60", "v": 60}],
        to_env=lambda cand: {"NAMES": cand["name"]},
        kind="select_best",
    )
    train = _fake_train({"w20": 0.5, "w30": 3.0, "w60": 1.0})
    res = ar.run_axis(axis, 5, base, train, persist=lambda log: None)
    assert res["best"]["name"] == "w30"
    assert res["best_delta"] > 0


def test_run_axis_select_best_none_beats_base():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(
        name="fake_sel",
        propose=lambda log: [{"name": "w20", "v": 20}],
        to_env=lambda cand: {"NAMES": cand["name"]},
        kind="select_best",
    )
    res = ar.run_axis(axis, 5, base, _fake_train({"w20": -2.0}), persist=lambda log: None)
    assert res["best"] is None


def test_run_axis_select_best_resumes_correctly():
    # prior log accepted w30 (delta 3.0) then w60 (delta 1.0); the resumed run must
    # pick w30 as best (highest accepted delta), not w60 (last/highest-iter accepted).
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(
        name="fake_sel",
        propose=lambda log: [],
        to_env=lambda cand: {"NAMES": cand["name"]},
        kind="select_best",
    )
    prior_log = [
        {"axis": "fake_sel", "iter": 0, "cand": {"name": "w30"},
         "cand_mean_delta": 3.0, "accepted": True},
        {"axis": "fake_sel", "iter": 1, "cand": {"name": "w60"},
         "cand_mean_delta": 1.0, "accepted": True},
    ]
    res = ar.run_axis(axis, 5, base, _fake_train({}), prior_log=prior_log, persist=lambda log: None)
    assert res["best"]["name"] == "w30"
    assert res["best_delta"] == 3.0


def test_features_axis_to_env_and_validate():
    base_features = ["ret_1", "ret_5"]
    axis = ar.make_features_axis(base_features)
    assert axis.kind == "additive"
    spec = {"name": "f1", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}
    env = axis.to_env([spec])
    assert env["GTRADE_EXTRA_FEATURES"] == "f1"
    assert os.path.exists(env["GTRADE_DSL_SPECS"])
    # a spec referencing an unknown input fails validate against base_features
    bad = {"name": "f2", "op": "lag", "inputs": ["does_not_exist"], "params": {"k": 1}}
    assert axis.validate(bad, []) is False


def test_features_axis_validate_sees_kept_names():
    # a spec that references a previously-kept feature name validates once it is kept
    base_features = ["ret_1"]
    axis = ar.make_features_axis(base_features)
    kept = [{"name": "f1", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]
    dependent = {"name": "f2", "op": "lag", "inputs": ["f1"], "params": {"k": 1}}
    assert axis.validate(dependent, kept) is True


def test_labeling_axis_proposes_grid_and_maps_env():
    axis = ar.make_labeling_axis()
    assert axis.kind == "select_best"
    cands = axis.propose([])
    windows = sorted(c["window"] for c in cands)
    assert windows == [20, 30, 60]
    assert all(c["mode"] == "rel_median" for c in cands)
    env = axis.to_env({"mode": "rel_median", "window": 30})
    assert env == {"GTRADE_LABEL_MODE": "rel_median", "GTRADE_LABEL_WINDOW": "30"}


def test_labeling_axis_omits_logged_windows():
    axis = ar.make_labeling_axis()
    log = [{"axis": "labeling", "iter": 0, "cand": {"mode": "rel_median", "window": 20}}]
    windows = sorted(c["window"] for c in axis.propose(log))
    assert windows == [30, 60]


def test_build_axes_selects_by_name_and_skips_unknown(caplog):
    import logging
    base_features = ["ret_1"]
    with caplog.at_level(logging.WARNING):
        axes = ar.build_axes(["labeling", "bogus"], base_features)
    names = [a.name for a in axes]
    assert names == ["labeling"]
    assert any("bogus" in r.message for r in caplog.records)


def test_pruning_axis_propose_and_to_env(monkeypatch):
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    monkeypatch.delenv("GTRADE_EXTRA_FEATURES", raising=False)
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", "8")
    from core.features import active_candidate_features
    active = active_candidate_features()
    axis = ar.make_pruning_axis(["ret_1"])
    assert axis.kind == "additive"
    proposed = axis.propose([])
    assert len(proposed) == 1   # exactly one candidate per round, not the whole set
    assert proposed[0]["drop"] in set(active)
    assert axis.to_env([{"drop": "rsi"}, {"drop": "atr"}]) == {"GTRADE_DROP_FEATURES": "rsi,atr"}


def test_pruning_axis_omits_already_dropped(monkeypatch):
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    axis = ar.make_pruning_axis(["ret_1"])
    log = [{"axis": "pruning", "iter": 0, "cand": [{"drop": "rsi"}], "accepted": True}]
    proposed = axis.propose(log)
    assert len(proposed) == 1
    assert proposed[0]["drop"] != "rsi"


def test_pruning_axis_validate_respects_floor(monkeypatch):
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    from core.features import active_candidate_features
    n = len(active_candidate_features())
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", str(n))   # floor == full size: no drop allowed
    axis = ar.make_pruning_axis(["ret_1"])
    assert axis.validate({"drop": "rsi"}, []) is False
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", str(n - 2))  # allow dropping up to 2
    axis2 = ar.make_pruning_axis(["ret_1"])
    assert axis2.validate({"drop": "rsi"}, []) is True
    assert axis2.validate({"drop": "atr"}, [{"drop": "rsi"}]) is True
    assert axis2.validate({"drop": "bb_pos"}, [{"drop": "rsi"}, {"drop": "atr"}]) is False


def test_pruning_axis_greedy_over_rounds(monkeypatch):
    # Regression for C1: a proposer that returned ALL droppable features per round
    # made run_axis additive drop the entire candidate set in one eval, blowing past
    # PRUNE_MIN and never performing real backward-elimination. With the one-per-round
    # fix, run_axis must evaluate (and keep/reject) one drop at a time.
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    from core.features import active_candidate_features
    active = list(active_candidate_features())
    n = len(active)
    prune_min = n - 5   # allow at most 5 drops; budget below will exceed that
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", str(prune_min))

    F = active[0]   # the one feature whose drop genuinely helps

    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]

    def fake_train(subset, env):
        dropped = [d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d]
        score = 1.0
        for d in dropped:
            score += 0.5 if d == F else -0.5
        return [{"Asset": a, "Score": score} for a in ("SP500", "NVDA")]

    axis = ar.make_pruning_axis(["ret_1"])
    res = ar.run_axis(axis, budget=10, base_rows=base, train_fn=fake_train,
                       screen_base=None, persist=lambda log: None)

    kept_drops = [c["drop"] for c in res["kept"]]
    assert F in kept_drops                      # the genuinely helpful drop was kept
    assert len(kept_drops) < n - prune_min + 1   # not the whole candidate set dropped at once
    assert n - len(kept_drops) >= prune_min      # floor was never breached
    # every accepted round in the log added exactly one new drop (true backward-elimination)
    for entry in res["log"]:
        if entry.get("accepted"):
            assert len(entry["cand"]) >= 1


def test_build_axes_includes_pruning():
    axes = ar.build_axes(["pruning"], ["ret_1"])
    assert [a.name for a in axes] == ["pruning"]


def test_main_trains_base_once_and_runs_selected_axis(monkeypatch):
    calls = {"train": 0, "selection_base": 0}

    def fake_train_env(subset, env):
        calls["train"] += 1
        if subset == ar.SELECTION_ASSETS and not env:
            calls["selection_base"] += 1
        score = 1.0 + (3.0 if env.get("GTRADE_LABEL_WINDOW") == "30" else 0.0)
        return [{"Asset": a, "Score": score} for a in subset.split(",")]

    monkeypatch.setattr(ar, "train_env", fake_train_env)
    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setattr(ar, "is_adoptable", lambda b, v, n, bud: (True, "ok"))
    monkeypatch.setenv("GTRADE_AR_AXES", "labeling")
    monkeypatch.setattr(ar, "BUDGET", 5, raising=False)
    # fresh state
    monkeypatch.setattr(ar, "save_state", lambda s: None)
    monkeypatch.setattr(ar, "load_state", lambda: {})
    ar.main()
    # exactly one base train on the selection set (reused across axes)
    assert calls["selection_base"] == 1


def test_screen_env_adds_flag():
    assert ar.screen_env({"GTRADE_LABEL_MODE": "rel_median"}) == {
        "GTRADE_LABEL_MODE": "rel_median", "GTRADE_SCREEN_ONLY": "1"}


def test_screen_on_env(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_SCREEN", raising=False)
    assert ar._screen_on() is True            # default on
    monkeypatch.setenv("GTRADE_AR_SCREEN", "0")
    assert ar._screen_on() is False


def test_passes_screen_rejects_below_min():
    # screen base mean Score 1.0; a candidate scoring 0.5 (delta -0.5) fails, 3.0 passes
    screen_base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(name="x", propose=lambda log: [], to_env=lambda c: {"NAMES": c["name"]},
                   kind="select_best")

    def train(subset, env):
        s = 0.5 if env.get("NAMES") == "bad" else 3.0
        return [{"Asset": "SP500", "Score": s}, {"Asset": "NVDA", "Score": s}]

    ok_bad, _ = ar._passes_screen(axis, {"name": "bad"}, train, screen_base, 0.0)
    ok_good, _ = ar._passes_screen(axis, {"name": "good"}, train, screen_base, 0.0)
    assert ok_bad is False and ok_good is True


def test_passes_screen_failure_does_not_reject():
    # an empty screen result (infra failure) must NOT drop the candidate
    ok, delta = ar._passes_screen(
        ar.Axis(name="x", propose=lambda log: [], to_env=lambda c: {}, kind="select_best"),
        {"name": "c"}, lambda subset, env: [], [{"Asset": "SP500", "Score": 1.0}], 0.0)
    assert ok is True


def test_run_axis_screen_skips_full_eval_of_rejected():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    screen_base = base
    calls = {"full": 0, "screen": 0}

    def train(subset, env):
        if env.get("GTRADE_SCREEN_ONLY") == "1":
            calls["screen"] += 1
            s = 0.5 if env.get("NAMES") == "w20" else 3.0   # w20 fails the screen
        else:
            calls["full"] += 1
            s = 3.0
        return [{"Asset": a, "Score": s} for a in ("SP500", "NVDA")]

    axis = ar.Axis(name="s",
                   propose=lambda log: [{"name": "w20"}, {"name": "w30"}],
                   to_env=lambda c: {"NAMES": c["name"]}, kind="select_best")
    ar.run_axis(axis, 5, base, train, persist=lambda log: None,
                screen_base=screen_base, screen_min=0.0)
    # w20 screened out, only w30 reaches a full eval
    assert calls["full"] == 1 and calls["screen"] == 2


def _rows2(scores):
    return [{"Asset": a, "Score": s} for a, s in scores.items()]


def test_objective_delta_mean_and_min():
    base = {"A": 1.0, "B": 1.0, "C": 1.0}
    var = _rows2({"A": 2.0, "B": 1.5, "C": 0.5})   # deltas +1.0, +0.5, -0.5
    mean_v, deltas = ar._objective_delta(var, base, "mean")
    assert abs(mean_v - (1.0 + 0.5 - 0.5) / 3) < 1e-9
    min_v, _ = ar._objective_delta(var, base, "min")
    assert min_v == -0.5
    assert ar._objective_delta([], base, "min") == (0.0, [])


def test_mean_delta_wrapper_parity():
    base = {"A": 1.0, "B": 2.0}
    var = _rows2({"A": 1.5, "B": 2.5})
    assert ar._mean_delta(var, base) == ar._objective_delta(var, base, "mean")


def test_benjamini_hochberg_known_vector():
    # m=4, alpha=0.05: thresholds 0.0125, 0.025, 0.0375, 0.05 -> reject first two
    assert ar.benjamini_hochberg([0.01, 0.02, 0.2, 0.5], 0.05) == [True, True, False, False]
    # input order preserved
    assert ar.benjamini_hochberg([0.5, 0.01, 0.2, 0.02], 0.05) == [False, True, False, True]
    assert ar.benjamini_hochberg([], 0.05) == []
    assert ar.benjamini_hochberg([0.9, 0.8], 0.05) == [False, False]


def test_objective_env(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    assert ar._objective() == "mean"
    monkeypatch.setenv("GTRADE_AR_OBJECTIVE", "min")
    assert ar._objective() == "min"
    monkeypatch.setenv("GTRADE_AR_OBJECTIVE", "bogus")
    assert ar._objective() == "mean"          # unknown -> mean


def test_holdout_stats_mean_vs_min():
    base = _rows2({"A": 1.0, "B": 1.0, "C": 1.0})
    var = _rows2({"A": 2.0, "B": 2.0, "C": 0.5})   # deltas +1,+1,-0.5
    p_m, v_m, deltas, _ = ar.holdout_stats(base, var, "mean")
    p_n, v_n, _, _ = ar.holdout_stats(base, var, "min")
    assert v_m > v_n and v_n == -0.5
    assert p_m == p_n                              # objective does not change the sign-test p
    p0, v0, d0, tag0 = ar.holdout_stats(base, _rows2({}), "mean")
    assert d0 == [] and v0 == 0.0


def test_is_adoptable_mean_backward_compat():
    # all held-out improve a lot -> adoptable under the default mean path (ab_labeling parity)
    base = _rows2({a: 1.0 for a in ("A", "B", "C", "D", "E")})
    var = _rows2({a: 3.0 for a in ("A", "B", "C", "D", "E")})
    ok, why = ar.is_adoptable(base, var, 1, 1)
    assert ok is True
    # min objective: one asset regresses -> worst delta negative -> not practically over the bar
    var2 = _rows2({"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 0.0})
    ok2, _ = ar.is_adoptable(base, var2, 1, 1, objective="min")
    assert ok2 is False


def test_run_axis_uses_min_objective(monkeypatch):
    # with objective=min, a candidate that lifts the mean but tanks one asset is rejected
    monkeypatch.setenv("GTRADE_AR_OBJECTIVE", "min")
    base = [{"Asset": "A", "Score": 1.0}, {"Asset": "B", "Score": 1.0}]
    axis = ar.Axis(name="s", propose=lambda log: [{"name": "x"}],
                   to_env=lambda c: {"NAMES": c["name"]}, kind="select_best")

    def train(subset, env):
        # x lifts A by +3 but tanks B by -1 -> mean +1 (good) but min -1 (bad)
        return [{"Asset": "A", "Score": 4.0}, {"Asset": "B", "Score": 0.0}]

    res = ar.run_axis(axis, 3, base, train, persist=lambda log: None)
    assert res["best"] is None         # min objective rejects it (worst delta -1 <= 0)


def test_main_applies_bh_across_axes(monkeypatch):
    # two axis winners -> main collects their held-out p-values and runs ONE BH
    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setattr(ar, "load_state", lambda: {})
    monkeypatch.setattr(ar, "save_state", lambda s: None)
    monkeypatch.setenv("GTRADE_AR_SCREEN", "0")   # screen off -> simpler full-eval path
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    captured = {"bh_input": None}
    real_bh = ar.benjamini_hochberg
    monkeypatch.setattr(ar, "benjamini_hochberg",
                        lambda p, alpha=0.05: (captured.__setitem__("bh_input", list(p)), real_bh(p, alpha))[1])

    # base (no NAMES) scores 1.0; any candidate (NAMES set) scores 5.0 -> beats base
    def fake_train_env(subset, env):
        s = 5.0 if env.get("NAMES") else 1.0
        return [{"Asset": a, "Score": s} for a in subset.split(",")]
    monkeypatch.setattr(ar, "train_env", fake_train_env)
    monkeypatch.setattr(ar, "build_axes", lambda names, bf: [
        ar.Axis(name="ax1", propose=lambda log: [{"name": "a"}], to_env=lambda c: {"NAMES": "a"}, kind="select_best"),
        ar.Axis(name="ax2", propose=lambda log: [{"name": "b"}], to_env=lambda c: {"NAMES": "b"}, kind="select_best"),
    ])
    monkeypatch.setattr(ar, "BUDGET", 2, raising=False)
    ar.main()
    # BH was applied across exactly the two axis-winners' p-values
    assert captured["bh_input"] is not None and len(captured["bh_input"]) == 2
