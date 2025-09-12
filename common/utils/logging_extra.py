from __future__ import annotations

from typing import Any, Mapping

import contextvars
import logging


_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(correlation_id: str | None) -> None:
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def clear_correlation_id() -> None:
    _correlation_id.set(None)


class CorrelationIdAdapter(logging.LoggerAdapter):
    """Lightweight adapter to include correlation id uniformly in logs.

    Usage:
        base = logging.getLogger(__name__)
        logger = CorrelationIdAdapter(base, {"cid": correlation_id})
        logger.info("message")

    This avoids coupling global logging configuration to modules and plays well
    with existing logging context/decorators since it's purely additive.
    """

    def process(self, msg: Any, kwargs: Mapping[str, Any]):
        extra = kwargs.get("extra", {}) or {}
        cid = self.extra.get("cid") or get_correlation_id()
        if cid is not None:
            # augment record
            extra.setdefault("cid", cid)
            kwargs["extra"] = extra
        return msg, kwargs
