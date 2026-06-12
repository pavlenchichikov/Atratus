"""Веб-интерфейс: радар сигналов, track record, риск.

Читает готовые предсказания из market.db (их пишет predict.py),
модели не загружает — стартует мгновенно.

Запуск:
    uvicorn webapp:app --host 0.0.0.0 --port 8000
"""

import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import FULL_ASSET_MAP, RADAR_GROUPS
from core import track_record
from risk_manager import RISK_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_STATE_PATH = os.path.join(BASE_DIR, "models", "risk_state.json")

app = FastAPI(title="G-Trade")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def _risk_state():
    if os.path.exists(RISK_STATE_PATH):
        try:
            with open(RISK_STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _grouped_signals():
    """Сигналы, разложенные по группам радара (как в консоли)."""
    sigs = {s["asset"]: s for s in track_record.latest_signals()}
    groups = []
    for group, members in RADAR_GROUPS.items():
        rows = [sigs[a] for a in members if a in sigs]
        if rows:
            groups.append({"name": group, "rows": rows})
    return groups, len(sigs)


@app.get("/", response_class=HTMLResponse)
def radar(request: Request):
    groups, total = _grouped_signals()
    return templates.TemplateResponse(request, "radar.html", {
        "groups": groups, "total": total,
    })


@app.get("/asset/{name}", response_class=HTMLResponse)
def asset_page(request: Request, name: str):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    track = track_record.asset_track(name, limit=60)
    acc = track_record.asset_accuracy(name)
    return templates.TemplateResponse(request, "asset.html", {
        "asset": name, "track": track, "acc": acc,
    })


@app.get("/risk", response_class=HTMLResponse)
def risk_page(request: Request):
    state = _risk_state()
    return templates.TemplateResponse(request, "risk.html", {
        "state": state, "config": RISK_CONFIG,
    })


@app.get("/api/signals")
def api_signals():
    return track_record.latest_signals()


@app.get("/api/track/{name}")
def api_track(name: str):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    return {
        "asset": name,
        "track": track_record.asset_track(name, limit=60),
        "accuracy": track_record.asset_accuracy(name),
    }


@app.get("/api/risk")
def api_risk():
    return {"state": _risk_state(), "config": RISK_CONFIG}
