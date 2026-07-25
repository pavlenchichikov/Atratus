"""The /research page shows where the agent is, or says there is no data."""

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
        "history": {"holdout_14": [30000], "assets": {"SLOW": [3600]}},
        "pending_units": [],
    })
    ar_progress.write_unit({"workers": 4, "assets_total": 1, "order": ["SLOW"], "done": []})
    body = _client().get("/api/research").json()
    assert body["progress"]["phase"] == "gate"
    assert body["progress"]["step"]["i"] == 2
    assert body["progress"]["eta"]["unit_left_s"] == 3600.0
    page = _client().get("/research")
    assert page.status_code == 200
    assert "gate" in page.text


def test_page_renders_when_there_is_no_progress_data(monkeypatch, tmp_path):
    monkeypatch.setattr(ar_progress, "AGENT_FILE", str(tmp_path / "a.json"))
    monkeypatch.setattr(ar_progress, "UNIT_FILE", str(tmp_path / "u.json"))
    page = _client().get("/research")
    assert page.status_code == 200


def test_hms_formats_durations():
    assert webapp._hms(None) == "unknown"
    assert webapp._hms(45) == "45s"
    assert webapp._hms(600) == "10m"
    assert webapp._hms(46714) == "12h 58m"
