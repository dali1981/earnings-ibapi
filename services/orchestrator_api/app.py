from fastapi import FastAPI
import logging
import os
from fastapi.middleware.cors import CORSMiddleware

from .routers.orchestrator import router as orchestrator_router
from .routers.orchestrator import router_public as orchestrator_public_router
from common.utils.logging_setup import setup_logging, disable_noisy_loggers


def create_app() -> FastAPI:
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    app = FastAPI(title="Orchestrator Service", version="0.1.0")
    # Prefixed and public routes for convenience
    app.include_router(orchestrator_router)
    app.include_router(orchestrator_public_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()
