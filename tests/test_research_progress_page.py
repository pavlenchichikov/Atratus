"""The /research page shows where the agent is, or says there is no data."""

import datetime
import json

from fastapi.testclient import TestClient

import webapp
from core import ar_progress


def _client():
    return TestClient(webapp.app)


def test_api_reports_no_data_when_nothing_ran(monkeypatch, tmp_path):
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "a.json"))
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "u.json"))
    body = _client().get("/api/research").json()
    assert body["progress"]["state"] == "no data"


def test_api_and_page_show_phase_step_and_estimate(monkeypatch, tmp_path):
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "a.json"))
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "u.json"))
    ar_progress.write_agent({
        "phase": "gate",
        "step": {"i": 2, "n": 3, "kind": "elite_holdout", "unit_kind": "holdout_14"},
        "history": {"holdout_14": [30000], "assets": {"holdout_14": {"SLOW": [3600]}}},
        "pending_units": [],
    })
    ar_progress.write_unit({"workers": 4, "assets_total": 1, "order": ["SLOW"], "done": []})
    body = _client().get("/api/research").json()
    assert body["progress"]["phase"] == "gate"
    assert body["progress"]["step"]["i"] == 2
    assert body["progress"]["eta"]["unit_left_s"] == 3600.0
    page = _client().get("/research")
    assert page.status_code == 200
    # "gate" alone is not proof: templates/research.html's static lead text already
    # contains the word "gate" ("clears the held-out gate in two independent
    # runs"), so that alone would pass even with the whole panel deleted. Assert on
    # strings only the rendered panel can produce.
    assert "2/3" in page.text
    assert "elite_holdout" in page.text
    assert "Assets in this step" in page.text


def test_page_renders_when_there_is_no_progress_data(monkeypatch, tmp_path):
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "a.json"))
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "u.json"))
    page = _client().get("/research")
    assert page.status_code == 200
    assert "not published" in page.text


def test_page_shows_age_not_a_stopped_claim_when_stale(monkeypatch, tmp_path):
    """A record older than STALE_AFTER_S must read as elapsed time, never as a
    flat claim that the run stopped, because age is what the reader actually
    knows."""
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "a.json"))
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "u.json"))
    ar_progress.write_agent({
        "phase": "gate",
        "step": {"i": 2, "n": 3, "kind": "elite_holdout", "unit_kind": "holdout_14"},
        "history": {}, "pending_units": [],
    })
    ar_progress.write_unit({"workers": 4, "assets_total": 1, "order": ["SLOW"], "done": []})
    stale_stamp = (datetime.datetime.now() - datetime.timedelta(seconds=ar_progress.STALE_AFTER_S + 1))
    for path in (ar_progress.AGENT_FILE, ar_progress.UNIT_FILE):
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        record["updated_at"] = stale_stamp.isoformat(timespec="seconds")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
    page = _client().get("/research")
    assert page.status_code == 200
    assert "No update for" in page.text
    assert "Run stopped" not in page.text
    # The stale branch still shows the phase; it does not make the panel vanish.
    assert "gate" in page.text
    assert "2/3" in page.text


def test_hms_formats_durations():
    assert webapp._hms(None) == "unknown"
    assert webapp._hms(45) == "45s"
    assert webapp._hms(600) == "10m"
    assert webapp._hms(46714) == "12h 58m"
