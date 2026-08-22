"""A warning that fires on success is a warning nobody reads.

The loader tries several paths. The first, load_keras_native, fails for every
legacy champion and a later one then succeeds - so warning inside it printed two
lines per asset across 200+ assets on every predict run. The one time a member
really is lost, that line would sit unread among four hundred identical ones.
"""
import logging
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import model_io


def test_a_loaded_champion_says_nothing_at_warning_level(caplog, tmp_path):
    good = tmp_path / "m.keras"
    good.write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="model_io"):
        model_io._warn_lost(str(good), object())      # something loaded it
    assert caplog.records == []


def test_a_champion_no_loader_could_read_warns_once(caplog, tmp_path):
    lost = tmp_path / "m.keras"
    lost.write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="model_io"):
        model_io._warn_lost(str(lost), None)
    assert len(caplog.records) == 1
    assert "no loader could read it" in caplog.records[0].message


def test_a_file_that_was_never_there_is_not_a_lost_champion(caplog, tmp_path):
    """Absent and unreadable are different states. An asset that never had a
    transformer is not a member that went missing."""
    with caplog.at_level(logging.WARNING, logger="model_io"):
        model_io._warn_lost(str(tmp_path / "nope.keras"), None)
    assert caplog.records == []


def test_the_first_attempt_only_speaks_at_debug(caplog, tmp_path):
    junk = tmp_path / "junk.keras"
    junk.write_bytes(b"definitely not a model")
    with caplog.at_level(logging.WARNING, logger="model_io"):
        assert model_io.load_keras_native(str(junk)) is None
    assert caplog.records == [], "the first of several attempts must stay quiet"
