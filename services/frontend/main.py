from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

import os
from app_settings.settings import get_settings
from fastapi.responses import FileResponse


app = FastAPI(title="Trading Frontend", version="0.1.0")

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = static_dir / "favicon.ico"
    if path.exists():
        # Explicit media type to help some browsers
        return FileResponse(str(path), media_type="image/x-icon")
    return HTMLResponse(status_code=404)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    s = get_settings()
    # Defaults from settings file, with legacy env override if provided
    bases = {
        "calendar": s.api.services.calendar.url,
        "equity": s.api.services.equity.url,
        "chains": s.api.services.chains.url,
        "orchestrator": s.api.services.orchestrator.url,
    }
    # Legacy override support (FRONTEND_*_BASE) for compatibility
    bases["calendar"] = os.getenv("FRONTEND_CALENDAR_BASE", bases["calendar"])  # type: ignore
    bases["equity"] = os.getenv("FRONTEND_EQUITY_BASE", bases["equity"])  # type: ignore
    bases["chains"] = os.getenv("FRONTEND_CHAINS_BASE", bases["chains"])  # type: ignore
    bases["orchestrator"] = os.getenv("FRONTEND_ORCHESTRATOR_BASE", bases["orchestrator"])  # type: ignore

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "calendar_base": bases["calendar"],
            "equity_base": bases["equity"],
            "chains_base": bases["chains"],
            "orchestrator_base": bases["orchestrator"],
        },
    )


# (No same-origin proxy: frontend JS calls services directly)
