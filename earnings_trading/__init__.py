"""
Deprecated: 'earnings_trading' package

This package previously mixed discovery + trading entrypoints. The repository
now separates concerns:
- Use `earnings.discovery` for pure earnings discovery and exports.
- Use `trading.strategy_analyzer` for optional trading analysis later.

This package remains as a compatibility shim only and should not be used for
new code.
"""

import warnings as _warnings

_warnings.warn(
    "The 'earnings_trading' package is deprecated. Use 'earnings.discovery' "
    "for pure discovery and 'trading.strategy_analyzer' for optional trading "
    "analysis.",
    DeprecationWarning,
)

__all__ = []
