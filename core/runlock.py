"""A run lock that a crash cannot leave holding the door shut.

The previous lock was a zero-byte file: whoever created it owned the door until
someone deleted it by hand. A killed cycle therefore blocked every later retrain
silently, and the only symptom was "lock present; skipping" forever. One such
file sat in the tree from 2026-06-30 and blocked loop_retrain.py for a month.

A lock now records the pid that took it, so a lock whose owner is gone is
recognised as stale and taken over. A lock whose owner is alive is still
respected: that is the whole point of having one.
"""

import json
import os

_STALE_AFTER_HOURS = 48


def _alive(pid):
    """True when a process with this pid exists.

    psutil when available, because os.kill(pid, 0) is not a liveness probe on
    Windows - it routes to TerminateProcess and would kill the very process it
    was asked about. Without psutil the caller falls back to the age rule, which
    is coarse but never kills anything.
    """
    if not pid:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.pid_exists(int(pid))
    except (TypeError, ValueError):
        return None


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def owner(path):
    """(pid, holder) recorded in the lock, or (None, None) when unreadable.

    An empty dict covers the old zero-byte format, which carried no owner at all.
    """
    data = _read(path)
    return data.get("pid"), data.get("holder")


def is_stale(path, now=None):
    """True when the lock exists but its owner is gone.

    Falls back to age when liveness cannot be determined, so a machine without
    psutil still recovers instead of blocking forever.
    """
    if not os.path.exists(path):
        return False
    pid, _holder = owner(path)
    alive = _alive(pid)
    if alive is True:
        return False
    if alive is False:
        return True
    # Unknown owner (old format) or no way to probe: fall back to age.
    import time
    age_h = ((now or time.time()) - os.path.getmtime(path)) / 3600.0
    return age_h > _STALE_AFTER_HOURS


def acquire(path, holder="run"):
    """Take the lock. Returns (True, None) on success.

    Returns (False, reason) only when a LIVE owner holds it. A stale lock is
    taken over and the takeover is reported to the caller so it reaches the log
    rather than happening invisibly.
    """
    if os.path.exists(path):
        if not is_stale(path):
            pid, held_by = owner(path)
            return False, ("held by {} (pid {})".format(held_by or "an earlier run",
                                                    pid or "unknown"))
        pid, held_by = owner(path)
        try:
            os.remove(path)
        except OSError as exc:
            return False, f"stale lock could not be removed: {exc}"
        print("[lock] took over a stale lock from {} (pid {})".format(held_by or "an unknown holder", pid or "unrecorded"))
    import datetime
    payload = {"pid": os.getpid(), "holder": holder,
               "since": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError as exc:
        return False, f"could not write the lock: {exc}"
    return True, None


def release(path):
    """Drop the lock, but only if this process still owns it.

    A takeover means someone else is legitimately holding it now; deleting their
    lock on the way out would hand the door to a third run.
    """
    pid, _holder = owner(path)
    if pid is not None and int(pid) != os.getpid():
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True
