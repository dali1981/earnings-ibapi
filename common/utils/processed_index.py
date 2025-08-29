"""Deprecated shim. Use `utils.events.processed_index` instead."""

from common.utils.events.processed_index import ProcessedIndex  # noqa: F401

import warnings


warnings.warn(
    "utils.processed_index is deprecated; use utils.events.processed_index",
    DeprecationWarning,
    stacklevel=2,
)
