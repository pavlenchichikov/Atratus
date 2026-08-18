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
