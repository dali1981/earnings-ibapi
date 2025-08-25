from fastapi import FastAPI
import logging
import os

from common.utils.logging_setup import setup_logging, disable_noisy_loggers

from .routers.chains import router as chains_router


def create_app() -> FastAPI:
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    app = FastAPI(title="Option Chain Service", version="0.2.0")
    app.include_router(chains_router)
    return app


app = create_app()
