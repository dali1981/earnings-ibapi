"""Deprecated repository package.

Use `repositories/` instead. This package will be removed in a future cleanup.
"""

import warnings


warnings.warn(
    "The 'repos' package is deprecated. Use 'repositories' instead.",
    DeprecationWarning,
    stacklevel=2,
)
