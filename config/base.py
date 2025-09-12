"""Base configuration classes using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import List, Optional, Dict, Any
from pathlib import Path


class DatabaseConfig(BaseSettings):
    """Database and data storage configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        extra="forbid"
    )
    
    data_root: Path = Field(default=Path("data"), description="Root directory for data storage")
    parquet_engine: str = Field(default="pyarrow", description="Parquet engine to use")
    max_rows_per_partition: int = Field(default=100000, ge=1000, description="Maximum rows per partition")
    partition_columns: List[str] = Field(default=["symbol", "date"], description="Default partition columns")
    compression: str = Field(default="snappy", description="Compression algorithm")
    
    @validator('data_root', pre=True)
    def validate_data_root(cls, v):
        """Ensure data_root is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @validator('parquet_engine')
    def validate_parquet_engine(cls, v):
        """Validate parquet engine choice."""
        allowed_engines = {"pyarrow", "fastparquet"}
        if v not in allowed_engines:
            raise ValueError(f"parquet_engine must be one of {allowed_engines}")
        return v
    
    @validator('compression')
    def validate_compression(cls, v):
        """Validate compression algorithm."""
        allowed_compression = {"snappy", "gzip", "brotli", "lz4", "zstd", "none"}
        if v not in allowed_compression:
            raise ValueError(f"compression must be one of {allowed_compression}")
        return v


class IBAPIConfig(BaseSettings):
    """Interactive Brokers API configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="IBAPI_",
        env_file=".env"
    )
    
    host: str = Field(default="127.0.0.1", description="IB Gateway/TWS host")
    port: int = Field(default=7497, ge=1, le=65535, description="IB Gateway/TWS port")
    client_id: int = Field(default=1, ge=0, le=32, description="Client ID")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Connection timeout in seconds")
    max_requests_per_second: int = Field(default=2, ge=1, le=50, description="Rate limiting")
    
    # Connection retry settings
    max_connection_attempts: int = Field(default=3, ge=1, description="Maximum connection retry attempts")
    connection_retry_delay: float = Field(default=5.0, ge=1.0, description="Delay between retries (seconds)")
    
    # Market data settings
    enable_delayed_data: bool = Field(default=True, description="Enable delayed market data")
    max_simultaneous_requests: int = Field(default=50, ge=1, le=100, description="Max concurrent requests")
    
    @validator('port')
    def validate_port(cls, v):
        """Validate IB API port."""
        # Common IB ports
        ib_ports = {7496, 7497, 4001, 4002}  # Live, Paper, Gateway Live, Gateway Paper
        if v not in ib_ports:
            # Warning but allow custom ports
            pass
        return v


class APIConfig(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env"
    )
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload (development only)")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # CORS settings
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:9000"], 
        description="Allowed CORS origins"
    )
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per minute")
    
    # Request/Response settings
    max_request_size: int = Field(default=16777216, ge=1024, description="Max request size in bytes (16MB)")
    response_timeout: float = Field(default=60.0, ge=1.0, description="Response timeout in seconds")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate logging level."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed_levels:
            raise ValueError(f"log_level must be one of {allowed_levels}")
        return v.upper()


class CeleryConfig(BaseSettings):
    """Celery task queue configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="CELERY_",
        env_file=".env"
    )
    
    broker_url: str = Field(default="redis://localhost:6379/0", description="Message broker URL")
    result_backend: str = Field(default="redis://localhost:6379/0", description="Result backend URL")
    
    # Task settings
    task_serializer: str = Field(default="json", description="Task serialization format")
    result_serializer: str = Field(default="json", description="Result serialization format")
    accept_content: List[str] = Field(default=["json"], description="Accepted content types")
    timezone: str = Field(default="UTC", description="Timezone for scheduled tasks")
    
    # Worker settings
    worker_prefetch_multiplier: int = Field(default=1, ge=1, description="Worker prefetch multiplier")
    task_acks_late: bool = Field(default=True, description="Acknowledge tasks after completion")
    worker_max_tasks_per_child: int = Field(default=1000, ge=1, description="Max tasks per worker process")
    
    # Rate limiting
    task_default_rate_limit: str = Field(default="100/m", description="Default task rate limit")
    
    @validator('task_serializer', 'result_serializer')
    def validate_serializer(cls, v):
        """Validate serializer format."""
        allowed_formats = {"json", "pickle", "yaml", "msgpack"}
        if v not in allowed_formats:
            raise ValueError(f"serializer must be one of {allowed_formats}")
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env"
    )
    
    level: str = Field(default="INFO", description="Global logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # File logging
    file_enabled: bool = Field(default=True, description="Enable file logging")
    file_path: Path = Field(default=Path("data/logs/app.log"), description="Log file path")
    file_max_bytes: int = Field(default=10485760, ge=1024, description="Max log file size (10MB)")
    file_backup_count: int = Field(default=5, ge=1, description="Number of backup log files")
    
    # Console logging
    console_enabled: bool = Field(default=True, description="Enable console logging")
    console_level: Optional[str] = Field(default=None, description="Console-specific log level")
    
    # Structured logging
    json_format: bool = Field(default=False, description="Use JSON log format")
    include_correlation_id: bool = Field(default=True, description="Include correlation IDs")
    
    @validator('level', 'console_level')
    def validate_log_level(cls, v):
        """Validate logging level."""
        if v is None:
            return v
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed_levels:
            raise ValueError(f"log level must be one of {allowed_levels}")
        return v.upper()
    
    @validator('file_path', pre=True)
    def validate_file_path(cls, v):
        """Ensure log file path is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class SecurityConfig(BaseSettings):
    """Security and authentication configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_file=".env"
    )
    
    # API Keys
    api_key_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_keys: List[str] = Field(default=[], description="Valid API keys")
    
    # JWT Settings
    jwt_enabled: bool = Field(default=False, description="Enable JWT authentication")
    jwt_secret_key: Optional[str] = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, description="JWT token expiration (hours)")
    
    # HTTPS
    https_enabled: bool = Field(default=False, description="Enable HTTPS")
    ssl_cert_file: Optional[Path] = Field(default=None, description="SSL certificate file")
    ssl_key_file: Optional[Path] = Field(default=None, description="SSL private key file")
    
    # Request validation
    validate_request_size: bool = Field(default=True, description="Validate request sizes")
    max_request_size_mb: int = Field(default=16, ge=1, description="Max request size (MB)")
    
    @validator('jwt_algorithm')
    def validate_jwt_algorithm(cls, v):
        """Validate JWT algorithm."""
        allowed_algorithms = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v not in allowed_algorithms:
            raise ValueError(f"jwt_algorithm must be one of {allowed_algorithms}")
        return v


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    version: str = Field(default="1.0.0", description="Application version")
    
    # Application metadata
    app_name: str = Field(default="Trading Platform", description="Application name")
    app_description: str = Field(default="Earnings-based trading platform", description="App description")
    
    # Feature flags
    features: Dict[str, bool] = Field(
        default={
            "metrics_enabled": True,
            "caching_enabled": True,
            "validation_enabled": True,
            "partitioning_enabled": True
        },
        description="Feature flags"
    )
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ibapi: IBAPIConfig = Field(default_factory=IBAPIConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = {"development", "testing", "staging", "production"}
        if v.lower() not in allowed_environments:
            raise ValueError(f"environment must be one of {allowed_environments}")
        return v.lower()
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"
    
    def get_feature(self, feature_name: str, default: bool = False) -> bool:
        """Get feature flag value.
        
        Args:
            feature_name: Name of the feature flag
            default: Default value if feature is not configured
            
        Returns:
            Feature flag value
        """
        return self.features.get(feature_name, default)
    
    def set_feature(self, feature_name: str, enabled: bool) -> None:
        """Set feature flag value.
        
        Args:
            feature_name: Name of the feature flag
            enabled: Whether the feature is enabled
        """
        self.features[feature_name] = enabled
    
    def get_data_path(self, *path_parts: str) -> Path:
        """Get path relative to data root.
        
        Args:
            *path_parts: Path components to join
            
        Returns:
            Complete path
        """
        return self.database.data_root.joinpath(*path_parts)
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.database.data_root,
            self.database.data_root / "logs",
            self.database.data_root / "cache",
            self.database.data_root / "cache" / "events",
            self.database.data_root / "cache" / "state",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)