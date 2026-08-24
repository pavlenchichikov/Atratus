"""The one thing gtrade.log must never do: stop recording.

Every process writes to the same file, so on Windows a rollover is a rename of
a file someone else has open. The stock handler fails that rename and goes
silent for the rest of the run.
"""

import logging
import os

from core.logger import SharedRotatingFileHandler


def _handler(tmp_path, max_bytes=200):
    fh = SharedRotatingFileHandler(str(tmp_path / "t.log"), maxBytes=max_bytes,
                                   backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    return fh


def _emit(fh, n, prefix="x"):
    for i in range(n):
        fh.emit(logging.LogRecord("t", logging.INFO, __file__, 1,
                                  "%s%03d%s" % (prefix, i, "-" * 40), (), None))


def test_a_rollover_nobody_blocks_still_rotates(tmp_path):
    fh = _handler(tmp_path)
    _emit(fh, 40)
    fh.close()
    assert (tmp_path / "t.log.1").exists()
    assert os.path.getsize(tmp_path / "t.log") < 200


def test_a_blocked_rollover_keeps_writing_instead_of_going_silent(tmp_path, monkeypatch):
    """The 2026-08-24 failure: the file froze at exactly maxBytes."""
    fh = _handler(tmp_path)
    _emit(fh, 10)                       # rotates freely, proves the control
    monkeypatch.setattr(os, "rename", _refuse)
    _emit(fh, 40, prefix="blocked")
    fh.close()
    body = (tmp_path / "t.log").read_text(encoding="utf-8")
    assert "blocked039" in body         # the last record survived the block
    assert os.path.getsize(tmp_path / "t.log") > 200


def test_a_blocked_rollover_is_not_retried_on_every_record(tmp_path, monkeypatch):
    """A rename that cannot work must not be attempted once per line."""
    tries = []

    def counted(src, dst):
        tries.append(src)
        _refuse(src, dst)

    fh = _handler(tmp_path)
    monkeypatch.setattr(os, "rename", counted)
    _emit(fh, 60)                       # ~3000 bytes at maxBytes=200
    fh.close()
    assert 0 < len(tries) <= 20


def _refuse(src, dst):
    raise PermissionError(32, "held open by another process")
