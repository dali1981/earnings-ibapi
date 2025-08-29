"""Deprecated shim. Use `utils.events.checkpoint` instead."""

from common.utils.events.checkpoint import (  # noqa: F401
    load_checkpoint,
    save_checkpoint,
)

import warnings


warnings.warn(
    "utils.checkpoint is deprecated; use utils.events.checkpoint",
    DeprecationWarning,
    stacklevel=2,
)
