"""
Trading analysis package for earnings-driven options strategies.

This package handles trading-specific analysis and strategy determination
based on pure earnings discoveries from the earnings discovery system.
"""

from .strategy_analyzer import (OptionsStrategy, OptionsStrategyAnalyzer,
                                TradingOpportunity)

__all__ = ["OptionsStrategyAnalyzer", "TradingOpportunity", "OptionsStrategy"]
