"""Unit tests for core.runlock (pure; no real processes are signalled)."""

import json
import os
import sys
import time

import pytest

from core import runlock


def lock_path(tmp_path):
    return str(tmp_path / "_loop.lock")


def test_acquire_on_a_free_door_succeeds(tmp_path):
    ok, reason = runlock.acquire(lock_path(tmp_path), "cycle")
    assert ok is True and reason is None
    pid, holder = runlock.owner(lock_path(tmp_path))
    assert pid == os.getpid() and holder == "cycle"


def test_a_live_owner_is_respected(tmp_path, monkeypatch):
    p = lock_path(tmp_path)
    runlock.acquire(p, "cycle")
    monkeypatch.setattr(runlock, "_alive", lambda pid: True)
    ok, reason = runlock.acquire(p, "retrain")
    assert ok is False
    assert "held by cycle" in reason


def test_a_dead_owner_is_taken_over(tmp_path, monkeypatch):
    # The bug this module exists for: a killed cycle used to hold the door shut
    # forever, and the only symptom was "lock present; skipping".
    p = lock_path(tmp_path)
    runlock.acquire(p, "cycle")
    monkeypatch.setattr(runlock, "_alive", lambda pid: False)
    ok, reason = runlock.acquire(p, "retrain")
    assert ok is True and reason is None
    _pid, holder = runlock.owner(p)
    assert holder == "retrain"


def test_the_old_zero_byte_format_is_stale_when_old(tmp_path, monkeypatch):
    # The lock that actually blocked this project was a zero-byte file with no
    # owner recorded at all.
    p = lock_path(tmp_path)
    open(p, "w").close()
    monkeypatch.setattr(runlock, "_alive", lambda pid: None)
    old = time.time() - (runlock._STALE_AFTER_HOURS + 1) * 3600
    os.utime(p, (old, old))
    assert runlock.is_stale(p) is True
    ok, _reason = runlock.acquire(p, "retrain")
    assert ok is True


def test_a_recent_ownerless_lock_is_still_respected(tmp_path, monkeypatch):
    # Without a way to probe liveness, a lock made minutes ago probably has a
    # live owner. Taking it over would let two retrains run at once.
    p = lock_path(tmp_path)
    open(p, "w").close()
    monkeypatch.setattr(runlock, "_alive", lambda pid: None)
    assert runlock.is_stale(p) is False
    ok, reason = runlock.acquire(p, "retrain")
    assert ok is False and reason


def test_release_only_drops_our_own_lock(tmp_path):
    p = lock_path(tmp_path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid() + 1, "holder": "someone else"}, fh)
    # A takeover means another run legitimately owns it now; deleting their lock
    # on the way out would hand the door to a third run.
    assert runlock.release(p) is False
    assert os.path.exists(p)


def test_release_drops_our_own_lock(tmp_path):
    p = lock_path(tmp_path)
    runlock.acquire(p, "cycle")
    assert runlock.release(p) is True
    assert not os.path.exists(p)


def test_a_missing_lock_is_not_stale(tmp_path):
    assert runlock.is_stale(str(tmp_path / "absent.lock")) is False


def test_liveness_is_unknown_without_psutil_and_signals_nothing(monkeypatch):
    """Without psutil the answer is "unknown", never a kill.

    This is the branch the Windows warning is about: os.kill(pid, 0) routes to
    TerminateProcess there, so runlock must not reach for it itself.

    The earlier version of this test patched os.kill globally and asserted it was
    never called at all. That held only while psutil was absent: psutil's POSIX
    pid_exists probes with os.kill(pid, 0), which is the correct call on Linux,
    so declaring psutil a dependency turned this green test red in CI without
    anything in runlock changing. What runlock owes is the ANSWER, not the
    mechanism a third-party library uses to reach it.
    """
    monkeypatch.setitem(sys.modules, "psutil", None)   # makes `import psutil` fail
    called = []
    monkeypatch.setattr(os, "kill", lambda *a: called.append(a))
    assert runlock._alive(os.getpid()) is None
    assert called == []


def test_liveness_answers_from_psutil_when_it_is_installed():
    pytest.importorskip("psutil")
    assert runlock._alive(os.getpid()) is True
