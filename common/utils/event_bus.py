"""Deprecated shim. Use `utils.events.bus` instead."""

import warnings

from common.utils.events.bus import Event, EventBus, FileEventBus  # noqa: F401

warnings.warn(
    "utils.event_bus is deprecated; use utils.events.bus",
    DeprecationWarning,
    stacklevel=2,
)
