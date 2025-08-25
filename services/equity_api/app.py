from fastapi import FastAPI
import logging
import os

from common.utils.logging_setup import setup_logging, disable_noisy_loggers

from .routers.equity import router as equity_router


def create_app() -> FastAPI:
    # Pretty logging
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    app = FastAPI(title="Equity History Service", version="0.2.0")
    app.include_router(equity_router)
    return app


app = create_app()
