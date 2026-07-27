"""Unit tests for the shared feature chain (pure; a fake engine, no database)."""

import io
import os

import pandas as pd

from core import features


def frame(n=60):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": 1.0, "high": 1.2, "low": 0.9,
                         "close": 1.0, "volume": 100.0}, index=idx)


def test_build_features_calls_every_step_in_order(monkeypatch):
    calls = []

    def step(name, ret_tuple=False):
        def fn(df, *a, **kw):
            calls.append(name)
            return (df, []) if ret_tuple else df
        return fn

    monkeypatch.setattr(features, "engineer_features", step("engineer"))
    monkeypatch.setattr(features, "add_weekly_features", step("weekly"))
    monkeypatch.setattr(features, "add_crossasset_features", step("crossasset"))
    monkeypatch.setattr(features, "add_macro_features", step("macro"))
    monkeypatch.setattr(features, "add_cross_lag_features", step("cross_lag"))
    monkeypatch.setattr(features, "add_chronos_features", step("chronos"))
    monkeypatch.setattr(features, "add_dsl_features", step("dsl", ret_tuple=True))
    monkeypatch.setattr(features, "load_dsl_specs", lambda: [])

    df, skipped = features.build_features(frame(), "btc", object())
    assert calls == ["engineer", "weekly", "crossasset", "macro", "cross_lag",
                     "chronos", "dsl"]
    assert skipped == []


def test_build_features_returns_the_skipped_spec_names(monkeypatch):
    # A silently skipped spec is a missing column, which is the same train/serve
    # skew by another route. The caller must be able to see it.
    monkeypatch.setattr(features, "engineer_features", lambda df: df)
    for name in ("add_weekly_features", "add_crossasset_features",
                 "add_chronos_features", "add_macro_features",
                 "add_cross_lag_features"):
        monkeypatch.setattr(features, name, lambda df, *a, **kw: df)
    monkeypatch.setattr(features, "load_dsl_specs", lambda: [{"name": "x"}])
    monkeypatch.setattr(features, "add_dsl_features",
                        lambda df, engine, specs: (df, ["x"]))
    _df, skipped = features.build_features(frame(), "btc", object())
    assert skipped == ["x"]


def test_no_adoption_leaves_the_frame_untouched_by_the_last_two_steps(monkeypatch):
    # With chronos off and no specs, the two steps the serve path is gaining must
    # be strict no-ops, or switching a shipped caller would change its numbers.
    monkeypatch.delenv("GTRADE_CHRONOS", raising=False)
    monkeypatch.delenv("GTRADE_DSL_SPECS", raising=False)
    df = frame()
    before = features.add_chronos_features(df.copy(), "btc", None)
    assert list(before.columns) == list(df.columns)
    from core.feature_dsl import add_dsl_features, load_dsl_specs
    after, skipped = add_dsl_features(df.copy(), None, load_dsl_specs())
    assert list(after.columns) == list(df.columns)
    assert skipped == []


# Files allowed to build a chain by hand, each for a stated reason.
#
# auto_research measures candidates by spawning train_hybrid as a subprocess; its
# own chain feeds the correlation screen, which calls materialize directly.
#
# pretrain_foundation deliberately uses its own fixed FOUNDATION_FEATURES schema
# rather than the active candidate list, so the full chain would be wrong for it.
CHAIN_EXEMPT = ("features.py", "feature_dsl.py", "auto_research.py",
                "pretrain_foundation.py")


def test_the_chain_is_defined_in_exactly_one_place():
    # The drift guard. This bug arrived because the chain is copy-pasted: the DSL
    # step was added to training and forgotten in every other caller.
    #
    # engineer_features is the signal, not the last two steps: a caller that had
    # only the first five never called add_dsl_features either, so guarding on
    # that would let a hand-built five-step chain pass unnoticed. Anyone building
    # the chain by hand starts here.
    root = os.path.dirname(os.path.dirname(os.path.abspath(features.__file__)))
    offenders = []
    for dirpath, _dirs, names in os.walk(root):
        if any(part in dirpath for part in (".git", "__pycache__", "tests",
                                            ".superpowers", "docs")):
            continue
        for fn in names:
            if not fn.endswith(".py") or fn in CHAIN_EXEMPT:
                continue
            src = io.open(os.path.join(dirpath, fn), encoding="utf-8",
                          errors="ignore").read()
            for step in ("engineer_features(", "add_dsl_features(",
                         "add_chronos_features("):
                if step in src:
                    offenders.append("%s calls %s" % (fn, step))
    assert offenders == [], (
        "these callers bypass build_features and will drift: %s" % offenders)
