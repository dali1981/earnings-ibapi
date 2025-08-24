"""Deprecated shim. Use `utils.events.processed_index` instead."""

import warnings

from common.utils.events.processed_index import ProcessedIndex  # noqa: F401

warnings.warn(
    "utils.processed_index is deprecated; use utils.events.processed_index",
    DeprecationWarning,
    stacklevel=2,
)
