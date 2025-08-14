"""Deprecated shim. Use `utils.events.consumer` instead."""

import warnings

from utils.events.consumer import ConsumerCheckpoint, JSONLConsumer  # noqa: F401

warnings.warn(
    "utils.event_bus_consumer is deprecated; use utils.events.consumer",
    DeprecationWarning,
    stacklevel=2,
)
