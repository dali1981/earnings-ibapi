from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

import os


app = FastAPI(title="Trading Frontend", version="0.1.0")

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "calendar_base": os.getenv(
                "FRONTEND_CALENDAR_BASE", "http://localhost:9001"
            ),
            "equity_base": os.getenv("FRONTEND_EQUITY_BASE", "http://localhost:9002"),
            "chains_base": os.getenv("FRONTEND_CHAINS_BASE", "http://localhost:9004"),
            "orchestrator_base": os.getenv(
                "FRONTEND_ORCHESTRATOR_BASE", "http://localhost:9005"
            ),
        },
    )
