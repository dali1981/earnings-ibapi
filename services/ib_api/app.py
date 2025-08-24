from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.ib import router as ib_router


def create_app() -> FastAPI:
    app = FastAPI(title="IBX Service API", version="0.1.0")
    app.include_router(ib_router, prefix="/ib", tags=["ib"])
    # Open up CORS for local development; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()
