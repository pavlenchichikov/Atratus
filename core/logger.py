"""Centralized logging configuration for Atratus.

Usage in any module:
    from core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")

Logs go to both console (INFO) and file gtrade.log (DEBUG).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_LOG_DIR, "gtrade.log")
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 3

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


class SharedRotatingFileHandler(RotatingFileHandler):
    """A rotating handler that survives a rollover another process is blocking.

    Every Atratus process logs to the SAME gtrade.log, and Windows refuses to
    rename a file anyone else still holds open. The stock handler closes the
    stream, fails the rename, and logging swallows the error - after which the
    file never receives another record. Observed 2026-08-24: gtrade.log stuck
    at exactly 5242880 bytes since 07:02 with the campaign still cycling, and a
    gtrade.log.3 with no .1 or .2 from the half-finished rename cascade.

    Losing records is worse than an oversized file, so a blocked rollover keeps
    writing and re-arms one maxBytes later, which also stops a doomed rename
    from being attempted once per record. Whichever process next finds the file
    unheld rotates it for everyone.
    """

    _retry_at = 0

    def doRollover(self):
        try:
            super().doRollover()
            self._retry_at = 0
        except OSError:
            if self.stream is None:
                self.stream = self._open()
            self.stream.seek(0, 2)
            self._retry_at = self.stream.tell() + self.maxBytes

    def shouldRollover(self, record):
        if not super().shouldRollover(record):
            return 0
        return 1 if self.stream.tell() >= self._retry_at else 0


def _setup_root():
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler - INFO level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # File handler - DEBUG level, rotating
    try:
        fh = SharedRotatingFileHandler(
            _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)
    except OSError:
        # Can't write to log file (read-only FS, permissions, etc.)
        pass

    # Suppress noisy third-party loggers
    for name in ("urllib3", "matplotlib", "PIL", "h5py", "absl"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Initializes root logging on first call."""
    _setup_root()
    return logging.getLogger(name)
