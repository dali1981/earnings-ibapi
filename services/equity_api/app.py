from .routers.equity import router as equity_router
from common.utils.logging_setup import disable_noisy_loggers, setup_logging
from fastapi import FastAPI

import logging
import os


app = FastAPI(title="Equity History Service", version="0.2.0")
app.include_router(equity_router)


@app.on_event("startup")
async def _configure_logging():
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
