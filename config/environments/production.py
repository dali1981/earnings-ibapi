"""Production environment configuration."""

from pathlib import Path
from ..base import AppConfig, APIConfig, LoggingConfig, IBAPIConfig, SecurityConfig, CeleryConfig


class ProductionConfig(AppConfig):
    """Production-specific configuration."""
    
    environment: str = "production"
    debug: bool = False
    
    # Production API settings
    api: APIConfig = APIConfig(
        workers=4,  # Multiple workers for production
        reload=False,  # Never reload in production
        log_level="INFO",
        cors_origins=[
            "https://trading.yourdomain.com",  # Production frontend
            "https://api.yourdomain.com",  # API domain
        ],
        rate_limit_enabled=True,
        rate_limit_requests=1000,  # 1000 requests per minute
        max_request_size=8388608,  # 8MB limit
        response_timeout=30.0
    )
    
    # Production IB API settings (live trading)
    ibapi: IBAPIConfig = IBAPIConfig(
        port=7496,  # Live trading port
        timeout=30.0,
        max_requests_per_second=2,  # Conservative rate limiting
        max_connection_attempts=5,
        connection_retry_delay=10.0,
        enable_delayed_data=False,  # Use real-time data
        max_simultaneous_requests=20  # Conservative for stability
    )
    
    # Production logging
    logging: LoggingConfig = LoggingConfig(
        level="INFO",
        console_enabled=False,  # No console output in production
        file_enabled=True,
        file_path=Path("/var/log/trading-platform/app.log"),
        file_max_bytes=52428800,  # 50MB log files
        file_backup_count=10,
        json_format=True,  # Structured logging for production
        include_correlation_id=True
    )
    
    # Production security
    security: SecurityConfig = SecurityConfig(
        api_key_enabled=True,
        jwt_enabled=True,
        jwt_expiration_hours=1,  # Short-lived tokens
        https_enabled=True,
        validate_request_size=True,
        max_request_size_mb=8
    )
    
    # Production Celery
    celery: CeleryConfig = CeleryConfig(
        broker_url="redis://redis-cluster:6379/0",
        result_backend="redis://redis-cluster:6379/0",
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=500,  # Restart workers periodically
        task_default_rate_limit="50/m"  # Conservative task rate limiting
    )
    
    # Production features
    features: dict = {
        "metrics_enabled": True,
        "caching_enabled": True,
        "validation_enabled": True,
        "partitioning_enabled": True,
        "monitoring_enabled": True,
        "alerting_enabled": True,
        "health_checks": True,
        "graceful_shutdown": True
    }
    
    def validate_production_settings(self) -> list[str]:
        """Validate production-specific settings.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Security validations
        if not self.security.api_key_enabled and not self.security.jwt_enabled:
            errors.append("At least one authentication method must be enabled in production")
        
        if self.security.jwt_enabled and not self.security.jwt_secret_key:
            errors.append("JWT secret key must be set in production")
        
        if not self.security.https_enabled:
            errors.append("HTTPS should be enabled in production")
        
        # API validations
        if self.api.workers < 2:
            errors.append("Production should use multiple API workers")
        
        if self.api.reload:
            errors.append("Auto-reload should be disabled in production")
        
        # Logging validations
        if self.logging.level == "DEBUG":
            errors.append("Debug logging should not be used in production")
        
        if not self.logging.file_enabled:
            errors.append("File logging should be enabled in production")
        
        # IB API validations
        if self.ibapi.port == 7497:
            errors.append("WARNING: Using paper trading port in production")
        
        if self.ibapi.max_requests_per_second > 5:
            errors.append("Rate limiting should be conservative in production")
        
        return errors