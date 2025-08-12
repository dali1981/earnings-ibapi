"""
Unified repository package for trading data storage.
Provides base repository functionality and specific implementations for different data types.
"""

from .base import BaseRepository
from .equity_bars import EquityBarRepository
from .option_bars import OptionBarRepository
from .option_chains import OptionChainSnapshotRepository
from .contract_descriptions import ContractDescriptionsRepository

__all__ = [
    'BaseRepository',
    'EquityBarRepository', 
    'OptionBarRepository',
    'OptionChainSnapshotRepository',
    'ContractDescriptionsRepository'
]

# Version info
__version__ = "2.0.0"
__author__ = "Trading System"