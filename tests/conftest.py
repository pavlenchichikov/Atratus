"""Shared fixtures. ar_memory state files are per-test isolated so no test
pollutes (or depends on) the real project-root research memory."""

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_environ():
    """Undo any os.environ the code under test writes for the whole process.

    monkeypatch.setenv already covers what a TEST sets. This covers what the
    code it calls sets, which nothing else does. A CLI entry point applying its
    process defaults inside a function left GTRADE_AR_TRAIN_CHUNK and
    GTRADE_AR_TRAIN_JOBS behind for the rest of the session, and three tests in
    a LATER file - alphabetically later, which is why it only ever failed in CI -
    stopped using their injected trainer and trained for real against a database
    that is not there.

    Silent restore rather than a failed test: a leak is the caller's bug to fix
    where it happens, and this only has to stop it reaching the next test.
    """
    saved = dict(os.environ)
    yield
    if os.environ != saved:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def _isolate_ar_memory(tmp_path, monkeypatch):
    from core import ar_memory
    monkeypatch.setattr(ar_memory, "TRIED_PATH", str(tmp_path / "_ar_tried.json"))
    monkeypatch.setattr(ar_memory, "CACHE_PATH", str(tmp_path / "_ar_eval_cache.json"))
    monkeypatch.setattr(ar_memory, "FINDINGS_PATH", str(tmp_path / "_ar_findings.json"))
    monkeypatch.setattr(ar_memory, "DB_PATH", str(tmp_path / "market.db"))
    monkeypatch.setattr(ar_memory, "REPLICATION_PATH", str(tmp_path / "_ar_replication.json"))
    import core.ar_wiki as _ar_wiki
    monkeypatch.setattr(_ar_wiki, "WIKI_DIR", str(tmp_path / "_ar_wiki"))
    # AR_PROGRESS_DIR, not monkeypatch.setattr, is what isolates ar_progress here:
    # the trainer runs as a subprocess (see progress_paths() in core/ar_progress.py),
    # so only an inherited environment variable reaches it. Individual test files
    # that also do monkeypatch.setattr(ar_progress, "AGENT_FILE"/"UNIT_FILE", ...)
    # keep working - progress_paths() takes the directory from this env var and
    # the filename from whatever AGENT_FILE/UNIT_FILE currently is, so the two
    # mechanisms agree as long as both point at this same tmp_path, which they do.
    monkeypatch.setenv("AR_PROGRESS_DIR", str(tmp_path))
    # Tests must never talk to a real LLM, no matter what the local .env enables:
    # GTRADE_AR_WIKI=1 + GTRADE_AR_LLM=ollama would otherwise make any test that
    # walks a run_qd/regate path fire compile_wiki() and load a real local model
    # (observed 2026-07-16: a suite run started Ollama). Tests that exercise these
    # paths inject fakes and set the vars explicitly.
    for var in ("GTRADE_AR_WIKI", "GTRADE_AR_PROPOSER", "GTRADE_AR_REFLECT",
                "GTRADE_AR_LLM", "GTRADE_AR_LLM_MODEL", "GTRADE_AR_RL",
                "GTRADE_TIMING_POLICY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _isolate_fitted_levels_policy(tmp_path, monkeypatch):
    """Point core.levels at a policy file that does not exist.

    levels_policy.json is gitignored, so CI never has one and the suite there
    measures the shipped K_ENTRY / K_STOP. A developer who has run [TL] does
    have one, and core.levels.load_policy() reads it on every levels() call and
    on every train_levels.baseline_params() call - so the same tests that pass
    in CI fail on the machine that actually fitted the policy, with numbers that
    look like an arithmetic bug and are not one.

    The path, not the function: tests call load_policy(explicit_path) to check
    the loader itself, and several monkeypatch load_policy for their own
    purposes. Both keep working when only the default location moves.
    """
    from core import levels
    monkeypatch.setattr(levels, "POLICY_PATH", str(tmp_path / "no_levels_policy.json"))


@pytest.fixture(autouse=True)
def _isolate_local_adoption(tmp_path, monkeypatch):
    """Undo what importing config did to this process on a developer's machine.

    config.py loads .env and then applies the adopted genome to os.environ, both
    at IMPORT time - which happens during collection, before any test or any
    fixture exists. _restore_environ above cannot help: by the time it takes its
    first snapshot the values are already there, so it faithfully preserves them.

    Neither .env nor adopted_genome.json is in git, so CI runs with none of this
    and the suite is green there. Locally the same tests read the developer's
    adopted feature drops, label mode and alert flag, and fail with differences
    that look like product bugs. Deleting the .env keys by name is what this file
    already did for four of them one at a time; taking them from the file makes
    the next flag somebody adds a non-event.
    """
    from core import adopted

    try:
        from dotenv import dotenv_values
        env_keys = list(dotenv_values(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")))
    except ImportError:
        env_keys = []
    for key in env_keys:
        if key.startswith("GTRADE_"):        # local feature flags only; leave
            monkeypatch.delenv(key, raising=False)   # proxies and credentials alone

    record = adopted.load()
    for key in adopted.env_overrides((record or {}).get("genome") or {}):
        monkeypatch.delenv(key, raising=False)
    # and the record itself: core.feature_dsl.load_dsl_specs() falls back to it
    # directly, so clearing the env alone still leaves seven adopted DSL specs in
    # force for anything that asks what features are active.
    monkeypatch.setattr(adopted, "PATH", str(tmp_path / "no_adopted_genome.json"))
