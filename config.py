# config.py
import logging.config
from pathlib import Path

# --- IB Connection ---
IB_HOST = "127.0.0.1"
IB_PORT = 4002  # Paper trading port
IB_CLIENT_ID = 77

# --- Data Paths ---
DATA_ROOT = Path("data")
EQUITY_BARS_PATH = DATA_ROOT / "equity_bars"
OPTION_BARS_PATH = DATA_ROOT / "option_bars"
OPTION_CHAINS_PATH = DATA_ROOT / "option_chains"
CONTRACTS_PATH = DATA_ROOT / "contracts"
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

# --- Create necessary directories ---
DATA_ROOT.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)

# --- Apply logging configuration ---
logging.config.dictConfig(LOGGING_CONFIG)