"""Development environment configuration."""

from pathlib import Path
from ..base import AppConfig, APIConfig, LoggingConfig, IBAPIConfig


class DevelopmentConfig(AppConfig):
    """Development-specific configuration."""
    
    environment: str = "development"
    debug: bool = True
    
    # Development API settings
    api: APIConfig = APIConfig(
        reload=True,
        log_level="DEBUG",
        workers=1,  # Single worker for development
        cors_origins=[
            "http://localhost:3000",  # Next.js frontend
            "http://localhost:9000",  # Legacy React frontend
            "http://localhost:8080",  # Alternative dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:9000",
            "http://127.0.0.1:8080",
        ]
    )
    
    # Development IB API settings (paper trading)
    ibapi: IBAPIConfig = IBAPIConfig(
        port=7497,  # Paper trading port
        timeout=60.0,  # Longer timeout for debugging
        max_requests_per_second=5,  # More relaxed rate limiting
        enable_delayed_data=True
    )
    
    # Development logging
    logging: LoggingConfig = LoggingConfig(
        level="DEBUG",
        console_enabled=True,
        file_enabled=True,
        file_path=Path("data/logs/dev.log"),
        json_format=False,  # Human-readable for development
        include_correlation_id=True
    )
    
    # Development features
    features: dict = {
        "metrics_enabled": True,
        "caching_enabled": True,  # Useful for development speed
        "validation_enabled": True,  # Catch issues early
        "partitioning_enabled": False,  # Simpler for small dev datasets
        "debug_mode": True,
        "hot_reload": True,
        "dev_tools": True
    }