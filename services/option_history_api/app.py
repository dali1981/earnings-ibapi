from .routers.options import router as options_router
from common.observability import init_observability
from common.utils.logging_setup import disable_noisy_loggers, setup_logging
from fastapi import FastAPI

import logging
import os


app = FastAPI(title="Option History Service", version="0.2.0")
app.include_router(options_router)


@app.on_event("startup")
async def _configure_logging():
    # Initialize observability first
    init_observability(service_name="option-history-api", service_version="0.2.0")
    
    # Then configure logging
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
