"""Deprecated shim. Use `utils.events.consumer` instead."""

from common.utils.events.consumer import ConsumerCheckpoint, JSONLConsumer  # noqa: F401

import warnings


warnings.warn(
    "utils.event_bus_consumer is deprecated; use utils.events.consumer",
    DeprecationWarning,
    stacklevel=2,
)
