"""
Unified repository package for trading data storage.
Provides base repository functionality and specific implementations for different data types.
"""

from .base import BaseRepository
from .contract_descriptions import ContractDescriptionsRepository
from .contracts import ContractRepository
from .equity_bars import EquityBarRepository
from .option_bars import OptionBarRepository
from .option_chains import OptionChainSnapshotRepository
from .option_contracts import OptionContractsRepository


__all__ = [
    "BaseRepository",
    "ContractRepository",
    "ContractDescriptionsRepository",
    "EquityBarRepository",
    "OptionBarRepository",
    "OptionChainSnapshotRepository",
    "OptionContractsRepository",
]

# Version info
__version__ = "2.0.0"
__author__ = "Trading System"
