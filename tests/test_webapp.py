"""Тесты веб-интерфейса через TestClient (без TensorFlow и реальной БД)."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from core import track_record
import webapp


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE prediction_log (
            date TEXT, asset TEXT, signal TEXT, probability REAL,
            actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL
        )
    """)
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-10','BTC','BUY',0.62,NULL,NULL,0.62,NULL)")
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-09','BTC','SELL',0.58,0.004,0,0.58,NULL)")
    con.commit()
    con.close()
    monkeypatch.setattr(track_record, "DB_PATH", path)
    return TestClient(webapp.app)


def test_radar_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "BTC" in r.text


def test_asset_page(client):
    r = client.get("/asset/BTC")
    assert r.status_code == 200
    assert "2026-06-10" in r.text


def test_asset_page_unknown_404(client):
    assert client.get("/asset/NOPE").status_code == 404


def test_risk_page(client):
    r = client.get("/risk")
    assert r.status_code == 200


def test_api_signals(client):
    data = client.get("/api/signals").json()
    assert data[0]["asset"] == "BTC"
    assert data[0]["signal"] == "BUY"


def test_api_track(client):
    data = client.get("/api/track/BTC").json()
    assert data["asset"] == "BTC"
    assert len(data["track"]) == 2


def test_api_risk(client):
    data = client.get("/api/risk").json()
    assert "config" in data


def test_api_prices(client):
    data = client.get("/api/prices/BTC?days=30").json()
    assert data["asset"] == "BTC"
    assert isinstance(data["series"], list)


def test_api_prices_unknown_404(client):
    assert client.get("/api/prices/NOPE").status_code == 404


def test_models_page(client):
    r = client.get("/models")
    assert r.status_code == 200
