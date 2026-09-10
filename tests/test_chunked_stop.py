"""Ctrl+C must stop the chunked trainer NOW, not at the end of the chunk.

It used to take up to an hour: this module installed no SIGINT handler, so the
interrupt reached the ThreadPoolExecutor, whose context manager waits for every
running future, while each child train_hybrid answered the same Ctrl+C by
finishing the assets it had already started.
"""
import subprocess
import sys
import threading
import time

import train_chunked as tc


def _reset():
    tc._stop.clear()
    with tc._live_lock:
        tc._live.clear()


def test_the_handler_terminates_a_running_child():
    _reset()
    rc = {}

    def run():
        rc["v"] = tc._spawn([sys.executable, "-c", "import time; time.sleep(120)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t = threading.Thread(target=run)
    t.start()
    deadline = time.time() + 10
    while not tc._live and time.time() < deadline:
        time.sleep(0.05)
    assert tc._live, "child never registered"

    t0 = time.time()
    tc._stop_now(None, None)
    t.join(timeout=15)
    assert not t.is_alive(), "the child outlived the handler"
    assert time.time() - t0 < 15
    assert rc["v"] != 0, "a terminated child must not look like a success"
    _reset()


def test_a_chunk_that_has_not_started_returns_at_once():
    """The queued futures are the other half of the delay: without this check
    each one would spawn a fresh trainer after the stop was requested."""
    _reset()
    tc._stop.set()
    t0 = time.time()
    ci, chunk, rc = tc._run_chunk(7, 10, ["AAPL"], False, 2)
    assert (ci, chunk) == (7, ["AAPL"]) and rc == -2
    assert time.time() - t0 < 1
    _reset()


def test_the_child_is_unregistered_after_a_normal_exit():
    """A leaked entry would have the handler terminate a pid it no longer owns."""
    _reset()
    rc = tc._spawn([sys.executable, "-c", "pass"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert rc == 0
    assert tc._live == []
