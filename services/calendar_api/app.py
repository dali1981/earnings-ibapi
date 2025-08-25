from fastapi import FastAPI
import logging
import os
from fastapi.middleware.cors import CORSMiddleware

from .routers.calendar import router as calendar_router
from common.utils.logging_setup import setup_logging, disable_noisy_loggers


def create_app() -> FastAPI:
    # Pretty logging
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    app = FastAPI(title="Calendar Service", version="0.2.0")
    app.include_router(calendar_router)
    # Enable CORS for local UI/testing (adjust origins as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()
