from __future__ import annotations

from .routers.ib import router as ib_router
from common.utils.logging_setup import disable_noisy_loggers, setup_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
import os


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


@app.on_event("startup")
async def _configure_logging():
    # Configure pretty logging (colors, brackets, timestamps) and make uvicorn loggers
    # propagate to root so they use our handlers/formatters.
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
