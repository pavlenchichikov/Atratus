import pytest


def test_screen_only_predicate(monkeypatch):
    import train_hybrid as th
    monkeypatch.delenv("GTRADE_SCREEN_ONLY", raising=False)
    assert th._screen_only() is False
    monkeypatch.setenv("GTRADE_SCREEN_ONLY", "1")
    assert th._screen_only() is True
    monkeypatch.setenv("GTRADE_SCREEN_ONLY", "0")
    assert th._screen_only() is False


@pytest.mark.slow
def test_screen_only_trains_fast_cb_only(tmp_path, monkeypatch):
    """A single-asset CB-only run finishes and writes a quality_report with a Score.
    Excluded from the fast suite (marker 'slow'); run with -m slow when a DB exists."""
    import json
    import os
    import subprocess
    import sys
    if not os.path.exists("market.db"):
        pytest.skip("no market DB")
    env = dict(os.environ)
    env.update({"GTRADE_SCREEN_ONLY": "1", "GTRADE_ASSETS": "SP500",
                "GTRADE_MODEL_DIR": str(tmp_path), "GTRADE_MAX_FOLDS": "3",
                "GTRADE_FORCE_PROMOTE": "1"})
    subprocess.run([sys.executable, "train_hybrid.py"], env=env, check=False)
    report = tmp_path / "quality_report.json"
    assert report.exists()
    rows = json.load(open(report))
    assert rows and "Score" in rows[0]
