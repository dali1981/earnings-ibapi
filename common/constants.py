"""System-wide constants for the trading platform."""

from enum import Enum
from typing import Final


# IB API Constants
class SecType(str, Enum):
    """Security type constants."""
    STOCK = "STK"
    OPTION = "OPT"  
    FUTURE = "FUT"
    FOREX = "CASH"
    INDEX = "IND"
    COMMODITY = "CMDTY"
    BOND = "BOND"


class OptionRight(str, Enum):
    """Option right constants."""
    CALL = "C"
    PUT = "P"


class Exchange(str, Enum):
    """Exchange constants."""
    SMART = "SMART"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CBOE = "CBOE"


# Data Constants
DEFAULT_BAR_SIZES: Final = ["1 min", "5 mins", "15 mins", "30 mins", "1 hour", "1 day"]
MAX_DAYS_HISTORY: Final = 365
DEFAULT_EXCHANGE: Final = "SMART"
DEFAULT_CURRENCY: Final = "USD"

# Repository Constants  
PARQUET_ENGINE: Final = "pyarrow"
MAX_ROWS_PER_PARTITION: Final = 100000
DEFAULT_PARTITION_COLS: Final = ["symbol", "date"]

# API Constants
DEFAULT_TIMEOUT: Final = 30.0
MAX_RETRIES: Final = 3
RATE_LIMIT_REQUESTS_PER_SECOND: Final = 2

# IB API Specific
IB_DEFAULT_HOST: Final = "127.0.0.1"
IB_PAPER_TRADING_PORT: Final = 7497
IB_LIVE_TRADING_PORT: Final = 7496
IB_DEFAULT_CLIENT_ID: Final = 1

# Event Bus Constants
EVENT_BUS_MAX_RETRIES: Final = 3
EVENT_BUS_RETRY_DELAY: Final = 1.0
DEFAULT_EVENT_TTL_SECONDS: Final = 3600  # 1 hour

# Cache Constants
DEFAULT_CACHE_TTL_SECONDS: Final = 300  # 5 minutes
MAX_CACHE_SIZE_MB: Final = 100

# Validation Constants
MAX_SYMBOL_LENGTH: Final = 10
MIN_SYMBOL_LENGTH: Final = 1
MAX_STRIKE_PRICE: Final = 10000.0
MIN_STRIKE_PRICE: Final = 0.01

# File System Constants
DATA_ROOT_DEFAULT: Final = "data"
LOGS_ROOT_DEFAULT: Final = "data/logs"
CACHE_ROOT_DEFAULT: Final = "data/cache"