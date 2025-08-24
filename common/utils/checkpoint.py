"""Deprecated shim. Use `utils.events.checkpoint` instead."""

import warnings

from common.utils.events.checkpoint import load_checkpoint, save_checkpoint  # noqa: F401

warnings.warn(
    "utils.checkpoint is deprecated; use utils.events.checkpoint",
    DeprecationWarning,
    stacklevel=2,
)
