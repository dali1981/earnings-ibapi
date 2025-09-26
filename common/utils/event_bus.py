"""Deprecated shim. Use `utils.events.bus` instead."""

from common.utils.events.bus import Event, EventBus, FileEventBus  # noqa: F401

import warnings


warnings.warn(
    "utils.event_bus is deprecated; use utils.events.bus",
    DeprecationWarning,
    stacklevel=2,
)
