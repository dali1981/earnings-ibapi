# config.py
import logging.config
from pathlib import Path

# --- IB Connection ---
IB_HOST = "127.0.0.1"
IB_PORT = 4002  # Paper trading port

# Dedicated client IDs for different jobs/examples
IB_CLIENT_IDS = {
    "default": 77,
    "streamer": 17,
    "contract_details_backfill": 8,
    "ib_source": 101,
    "contracts_backfill": 1,
}

# Fallback client id
IB_CLIENT_ID = IB_CLIENT_IDS["default"]

# --- Data Paths ---
DATA_ROOT = Path("data")
EQUITY_BARS_PATH = DATA_ROOT / "equity_bars"
OPTION_BARS_PATH = DATA_ROOT / "option_bars"
OPTION_CHAINS_PATH = DATA_ROOT / "option_chains"
CONTRACTS_PATH = DATA_ROOT / "contracts"
EARNINGS_PATH = DATA_ROOT / "earnings"
EXPORTS_PATH = DATA_ROOT / "exports"
CACHE_PATH = DATA_ROOT / "cache"
LOGS_PATH = DATA_ROOT / "logs"

# --- Logging Configuration ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": LOGS_PATH / "app.log",
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 5,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

# --- Backfill Settings ---
BACKFILL_START_DATE = "2023-01-01"
BACKFILL_END_DATE = "2023-12-31"
BACKFILL_BAR_SIZE = "1 day"

# --- Earnings Data Collection Settings ---
EARNINGS_CONFIG = {
    # Collection timing
    "default_days_ahead": 30,          # How many days ahead to collect earnings
    "max_days_ahead": 60,              # Maximum lookahead window
    "collection_hour": 6,              # Daily collection time (6 AM)
    
    # Data sources (in priority order)
    "data_sources": ["nasdaq", "fmp", "finnhub"],
    "primary_source": "nasdaq",        # Free, reliable source
    
    # Data retention
    "retention_days": 365,             # Keep 1 year of data
    "cache_hours": 6,                  # Cache fresh data for 6 hours
    
    # API rate limits
    "api_delays": {
        "nasdaq": 0.3,                 # Seconds between NASDAQ requests
        "fmp": 0.25,                   # FMP free tier: 250/day
        "finnhub": 1.0                 # Finnhub free tier: 60/min
    }
}

# --- Pure Earnings Discovery Settings ---
DISCOVERY_CONFIG = {
    # Basic market filters (NO trading-specific filters)
    "min_market_cap": 1_000_000_000,   # $1B minimum market cap
    "max_market_cap": 2_000_000_000_000, # $2T maximum (exclude mega caps)
    "min_avg_volume": 1_000_000,       # 1M shares average volume
    "min_price": 10.0,                 # $10 minimum stock price
    "max_price": 1000.0,               # $1000 maximum stock price
    
    # Discovery timing windows (how far ahead to look for earnings)
    "discovery_min_days": 1,           # At least 1 day ahead
    "discovery_max_days": 60,          # Within 2 months (discovery scope)
    
    # Data collection windows (when to collect price/IV data around earnings date)
    "data_collection": {
        "pre_earnings_days": 3,        # Collect 3 days before earnings
        "post_earnings_days": 2,       # Collect 2 days after earnings  
        "lookback_days": 30,           # Historical baseline data (30 days back)
        "iv_history_days": 90          # IV historical analysis (90 days back)
    },
    
    # Trading strategy analysis settings (moved here for compatibility)
    "strategy_optimal_start": 7,       # Strategy works best 7+ days before earnings
    "strategy_optimal_end": 14,        # Strategy works best up to 14 days before
    
    # Options requirements (for trading analysis)
    "require_weekly_options": False,    # Don't require weekly options
    "require_liquid_options": False,    # Don't require liquid options
    "min_iv_rank": 20,                 # Minimum IV rank (when available)
    
    # Strategy scoring weights (for trading analysis)
    "timing_weights": {
        "optimal": 100.0,              # 7-14 days = 100 points
        "good": 80.0,                  # 3-21 days = 80 points  
        "fair": 60.0,                  # 1-3 days = 60 points
        "poor": 40.0                   # Outside ranges = 40 points
    },
    
    # Strategy multipliers (for trading analysis)
    "strategy_multipliers": {
        "calendar_spread": {
            "14_plus_days": 1.0,
            "7_to_14_days": 0.8,
            "under_7_days": 0.5
        },
        "straddle": {
            "under_5_days": 1.0,
            "5_to_10_days": 0.9,
            "over_10_days": 0.7
        },
        "strangle": {
            "under_7_days": 1.0,
            "7_to_14_days": 0.8,
            "over_14_days": 0.6
        }
    },
    
    # Quality thresholds (for trading analysis)
    "min_opportunity_score": 50.0,     # Minimum score to be considered
    "excellent_threshold": 90.0,       # Excellent opportunity threshold
    "good_threshold": 75.0             # Good opportunity threshold
}

# --- Create necessary directories ---
DATA_ROOT.mkdir(exist_ok=True)
EQUITY_BARS_PATH.mkdir(exist_ok=True)
OPTION_BARS_PATH.mkdir(exist_ok=True)
OPTION_CHAINS_PATH.mkdir(exist_ok=True)
CONTRACTS_PATH.mkdir(exist_ok=True)
EARNINGS_PATH.mkdir(exist_ok=True)
EXPORTS_PATH.mkdir(exist_ok=True)
CACHE_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)

# --- Apply logging configuration ---
logging.config.dictConfig(LOGGING_CONFIG)
