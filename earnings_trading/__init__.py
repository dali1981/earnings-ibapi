"""
Earnings Trading Module

Earnings-driven options trading system that discovers market opportunities
and executes data collection for options strategies around earnings events.
"""

from .discovery import (
    EarningsDiscoveryEngine,
    EarningsCandidate,
    OptionsStrategy
)

__all__ = [
    'EarningsDiscoveryEngine',
    'EarningsCandidate', 
    'OptionsStrategy'
]