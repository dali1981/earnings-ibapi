"""Global logging bootstrap for the project.

Python auto-imports `sitecustomize` if present on sys.path. We use this to
apply a consistent, pretty logging configuration (colors, timestamps, brackets)
across all entrypoints: services, Celery workers, jobs, scripts, and tests.
"""
from __future__ import annotations

import logging
import os

try:
    from common.utils.logging_setup import setup_logging, disable_noisy_loggers

    # Respect environment override; default to INFO
    level = os.environ.get("APP_LOG_LEVEL", "INFO")
    # Enable file logging by default; allow disabling via env
    file_logging = os.environ.get("APP_LOG_TO_FILE", "1") not in {"0", "false", "False"}

    # If logging is already configured (handlers exist), we still reapply to
    # ensure our formatters are in place uniformly.
    setup_logging(level=level, enable_file_logging=file_logging)
    disable_noisy_loggers()

    # Make common framework loggers inherit our handlers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "celery", "celery.app", "celery.worker"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
except Exception:
    # Never block interpreter startup if logging configuration fails
    pass

