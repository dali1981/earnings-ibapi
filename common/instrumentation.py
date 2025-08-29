"""
Lightweight instrumentation shim for repositories.

Purpose:
- Allow repository code to import logging/lineage helpers without pulling in
  heavy dependencies when running unit tests.
- Prefer real implementations when available, otherwise fall back to no-ops.

Design:
- This module never configures logging; it only provides references to helpers.
- It tries to import real implementations; on failure, it provides no-ops.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable
import logging


# ----------------------------------------------------------------------------
# Reliability: exceptions, logger, context, perf decorators
# ----------------------------------------------------------------------------
try:
    from common.reliability import (  # type: ignore
        DataValidationException as _DataValidationException,
        StorageException as _StorageException,
        get_trading_logger as _get_trading_logger,
        log_context as _log_context,
        performance_log as _performance_log,
    )

    DataValidationException = _DataValidationException  # noqa: N816
    StorageException = _StorageException  # noqa: N816
    get_trading_logger = _get_trading_logger  # noqa: N816
    log_context = _log_context  # noqa: N816

    def performance_log(operation: str) -> Callable:  # noqa: D401
        return _performance_log(operation)

except Exception:

    class DataValidationException(ValueError):
        def __init__(self, message: str, field: str | None = None, **kwargs: Any):
            super().__init__(message)
            self.field = field
            self.context = kwargs

    class StorageException(Exception):
        def __init__(self, message: str, storage_type: str | None = None, **kwargs: Any):
            super().__init__(message)
            self.storage_type = storage_type
            self.context = kwargs

    def get_trading_logger(name: str):  # noqa: D401
        return logging.getLogger(name)

    @contextmanager
    def log_context(**kwargs: Any):  # noqa: D401
        yield

    def performance_log(operation: str) -> Callable:  # noqa: D401
        def decorator(func: Callable) -> Callable:
            return func

        return decorator


# ----------------------------------------------------------------------------
# Lineage: OperationType and decorator
# ----------------------------------------------------------------------------
try:
    from common.lineage.core import OperationType as _OperationType  # type: ignore
    from common.lineage.decorators import track_repository_operation as _track_repo  # type: ignore

    OperationType = _OperationType  # noqa: N816

    def track_repository_operation(operation_type: Any):  # noqa: D401
        return _track_repo(operation_type)

except Exception:

    class OperationType:  # noqa: D401, N801 (keep name to ease migration)
        READ = "read"
        WRITE = "write"

    def track_repository_operation(operation_type: Any):  # noqa: D401
        def decorator(func: Callable) -> Callable:
            return func

        return decorator
