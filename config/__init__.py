"""Configuration management for the trading platform."""

from .settings import get_config, settings
from .base import AppConfig, DatabaseConfig, IBAPIConfig, APIConfig

# Legacy compatibility layer - provides old config.py exports using app_settings
def _get_legacy_config():
    """Get legacy configuration values using the active app_settings system."""
    try:
        # Use the working app_settings system
        from app_settings.settings import get_settings
        settings = get_settings()
        
        return {
            'EARNINGS_CONFIG': {
                "default_days_ahead": 3,
                "default_days_back": 360,
                "max_days_ahead": 60,
                "collection_hour": 6,
                "data_sources": ["nasdaq"],
                "primary_source": "nasdaq",
                "retention_days": 365,
                "cache_hours": 6,
                "api_delays": {
                    "nasdaq": 0.3,
                    "fmp": 0.25,
                    "finnhub": 1.0,
                },
            },
            'EARNINGS_PATH': settings.paths.earnings,
            'DATA_ROOT': settings.paths.data_root,
            'EQUITY_BARS_PATH': settings.paths.equity_bars,
            'OPTION_BARS_PATH': settings.paths.option_bars,
            'OPTION_CHAINS_PATH': settings.paths.option_chains,
            'CONTRACTS_PATH': settings.paths.contracts,
            'CACHE_PATH': settings.paths.cache,
            'LOGS_PATH': settings.paths.logs,
            'IB_HOST': settings.ib.host,
            'IB_PORT': settings.ib.port,
            'IB_CLIENT_IDS': settings.ib.client_ids,
            'IB_CLIENT_ID': settings.ib.client_ids.get("default", 77),
        }
    except Exception:
        # Fallback to hardcoded values if app_settings fails
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]
        data_root = project_root / "data"
        return {
            'EARNINGS_CONFIG': {
                "default_days_ahead": 3,
                "default_days_back": 360,
                "max_days_ahead": 60,
                "collection_hour": 6,
                "data_sources": ["nasdaq"],
                "primary_source": "nasdaq",
                "retention_days": 365,
                "cache_hours": 6,
                "api_delays": {"nasdaq": 0.3, "fmp": 0.25, "finnhub": 1.0},
            },
            'EARNINGS_PATH': data_root / "earnings",
            'DATA_ROOT': data_root,
            'EQUITY_BARS_PATH': data_root / "equity_bars",
            'OPTION_BARS_PATH': data_root / "option_bars", 
            'OPTION_CHAINS_PATH': data_root / "option_chains",
            'CONTRACTS_PATH': data_root / "contracts",
            'CACHE_PATH': data_root / "cache",
            'LOGS_PATH': data_root / "logs",
            'IB_HOST': "127.0.0.1",
            'IB_PORT': 4002,
            'IB_CLIENT_IDS': {"default": 77, "streamer": 17},
            'IB_CLIENT_ID': 77,
        }

# Export legacy configuration variables
_legacy = _get_legacy_config()
EARNINGS_CONFIG = _legacy['EARNINGS_CONFIG']
EARNINGS_PATH = _legacy['EARNINGS_PATH']
DATA_ROOT = _legacy['DATA_ROOT']
EQUITY_BARS_PATH = _legacy['EQUITY_BARS_PATH']
OPTION_BARS_PATH = _legacy['OPTION_BARS_PATH']
OPTION_CHAINS_PATH = _legacy['OPTION_CHAINS_PATH']
CONTRACTS_PATH = _legacy['CONTRACTS_PATH']
CACHE_PATH = _legacy['CACHE_PATH']
LOGS_PATH = _legacy['LOGS_PATH']
IB_HOST = _legacy['IB_HOST']
IB_PORT = _legacy['IB_PORT']
IB_CLIENT_IDS = _legacy['IB_CLIENT_IDS']
IB_CLIENT_ID = _legacy['IB_CLIENT_ID']

__all__ = [
    "get_config",
    "settings", 
    "AppConfig",
    "DatabaseConfig",
    "IBAPIConfig",
    "APIConfig",
    # Legacy exports
    "EARNINGS_CONFIG",
    "EARNINGS_PATH",
    "DATA_ROOT",
    "EQUITY_BARS_PATH", 
    "OPTION_BARS_PATH",
    "OPTION_CHAINS_PATH",
    "CONTRACTS_PATH",
    "CACHE_PATH",
    "LOGS_PATH",
    "IB_HOST",
    "IB_PORT", 
    "IB_CLIENT_IDS",
    "IB_CLIENT_ID",
]