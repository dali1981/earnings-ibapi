from .routers.chains import router as chains_router
from common.utils.logging_setup import disable_noisy_loggers, setup_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
import os


app = FastAPI(title="Option Chain Service", version="0.2.0")
app.include_router(chains_router)

# Enable CORS to allow direct calls from the React UI or other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _configure_logging():
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
