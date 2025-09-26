"""
Common package: a unified namespace for shared modules.

This package provides import shortcuts so you can do:

    from common.repositories import EarningsCalendarRepository
    from common.utils import get_logger

without changing existing module locations. It re-exports from the
original packages to avoid disruptive moves.
"""
