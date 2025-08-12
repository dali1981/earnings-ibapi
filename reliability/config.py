"""
Configuration management for the reliability system.
Provides centralized configuration for retry policies, circuit breakers, and logging.
"""
import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .retry import RetryStrategy, JitterType
from .exceptions import ErrorSeverity


@dataclass
class GlobalRetryConfig:
    """Global retry configuration defaults."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: JitterType = JitterType.EQUAL
    log_attempts: bool = True


@dataclass
class GlobalCircuitBreakerConfig:
    """Global circuit breaker configuration defaults."""
    
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_duration: float = 60.0
    monitoring_window: float = 300.0
    failure_rate_threshold: float = 0.5
    minimum_requests: int = 10
    recovery_timeout: float = 30.0
    max_half_open_requests: int = 1
    log_state_changes: bool = True
    collect_metrics: bool = True


@dataclass 
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    format_type: str = "json"  # "json" or "text"
    include_context: bool = True
    console_logging: bool = True
    file_logging: bool = False
    log_file: Optional[str] = None
    log_rotation: bool = True
    max_log_size: str = "100MB"
    backup_count: int = 5


@dataclass
class OperationConfig:
    """Configuration for specific operation types."""
    
    # IB API specific settings
    ib_connection_timeout: float = 30.0
    ib_request_timeout: float = 60.0
    ib_rate_limit_delay: float = 0.1
    ib_max_retries: int = 5
    
    # Market data settings
    market_data_timeout: float = 30.0
    market_data_retries: int = 3
    
    # Historical data settings  
    historical_data_timeout: float = 120.0
    historical_data_retries: int = 3
    historical_data_rate_limit: float = 0.2
    
    # Contract details settings
    contract_details_timeout: float = 15.0
    contract_details_retries: int = 2
    
    # Option chain settings
    option_chain_timeout: float = 45.0
    option_chain_retries: int = 2
    option_chain_rate_limit: float = 0.3


@dataclass
class ReliabilityConfig:
    """Main configuration class for the reliability system."""
    
    # Global settings
    retry: GlobalRetryConfig = field(default_factory=GlobalRetryConfig)
    circuit_breaker: GlobalCircuitBreakerConfig = field(default_factory=GlobalCircuitBreakerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    operations: OperationConfig = field(default_factory=OperationConfig)
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_log_threshold: float = 1.0  # Log operations taking longer than 1s
    
    # Feature flags
    enable_circuit_breakers: bool = True
    enable_retries: bool = True
    enable_structured_logging: bool = True
    
    # Environment-specific settings
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReliabilityConfig':
        """Create configuration from dictionary."""
        
        # Convert nested dictionaries back to dataclasses
        if 'retry' in data and isinstance(data['retry'], dict):
            data['retry'] = GlobalRetryConfig(**data['retry'])
        
        if 'circuit_breaker' in data and isinstance(data['circuit_breaker'], dict):
            data['circuit_breaker'] = GlobalCircuitBreakerConfig(**data['circuit_breaker'])
            
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
            
        if 'operations' in data and isinstance(data['operations'], dict):
            data['operations'] = OperationConfig(**data['operations'])
        
        return cls(**data)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ReliabilityConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_operation_timeout(self, operation_type: str) -> float:
        """Get timeout for specific operation type."""
        timeout_map = {
            'ib_connection': self.operations.ib_connection_timeout,
            'ib_request': self.operations.ib_request_timeout,
            'market_data': self.operations.market_data_timeout,
            'historical_data': self.operations.historical_data_timeout,
            'contract_details': self.operations.contract_details_timeout,
            'option_chain': self.operations.option_chain_timeout,
        }
        return timeout_map.get(operation_type, 30.0)
    
    def get_operation_retries(self, operation_type: str) -> int:
        """Get retry count for specific operation type."""
        retry_map = {
            'ib_connection': self.operations.ib_max_retries,
            'market_data': self.operations.market_data_retries,
            'historical_data': self.operations.historical_data_retries,
            'contract_details': self.operations.contract_details_retries,
            'option_chain': self.operations.option_chain_retries,
        }
        return retry_map.get(operation_type, self.retry.max_attempts)
    
    def get_operation_rate_limit(self, operation_type: str) -> float:
        """Get rate limit delay for specific operation type."""
        rate_limit_map = {
            'ib_request': self.operations.ib_rate_limit_delay,
            'historical_data': self.operations.historical_data_rate_limit,
            'option_chain': self.operations.option_chain_rate_limit,
        }
        return rate_limit_map.get(operation_type, 0.0)


# Environment-specific configurations

def get_development_config() -> ReliabilityConfig:
    """Get configuration optimized for development."""
    config = ReliabilityConfig()
    config.environment = "development"
    config.debug_mode = True
    config.logging.level = "DEBUG"
    config.logging.console_logging = True
    config.logging.include_context = True
    config.retry.log_attempts = True
    config.circuit_breaker.log_state_changes = True
    return config


def get_production_config() -> ReliabilityConfig:
    """Get configuration optimized for production."""
    config = ReliabilityConfig()
    config.environment = "production"
    config.debug_mode = False
    config.logging.level = "INFO"
    config.logging.file_logging = True
    config.logging.log_file = "/var/log/trading/reliability.log"
    config.logging.log_rotation = True
    config.enable_performance_monitoring = True
    config.performance_log_threshold = 2.0  # Higher threshold for production
    
    # More aggressive retry settings for production
    config.retry.max_attempts = 5
    config.retry.max_delay = 120.0
    
    # More conservative circuit breaker settings
    config.circuit_breaker.failure_threshold = 3
    config.circuit_breaker.timeout_duration = 120.0
    
    return config


def get_testing_config() -> ReliabilityConfig:
    """Get configuration optimized for testing."""
    config = ReliabilityConfig()
    config.environment = "testing"
    config.debug_mode = True
    config.logging.level = "WARNING"  # Reduce noise in tests
    config.logging.console_logging = False
    
    # Fast retry settings for tests
    config.retry.max_attempts = 2
    config.retry.base_delay = 0.1
    config.retry.max_delay = 1.0
    
    # Fast circuit breaker recovery for tests
    config.circuit_breaker.timeout_duration = 1.0
    config.circuit_breaker.failure_threshold = 2
    
    # Shorter timeouts for tests
    config.operations.ib_connection_timeout = 5.0
    config.operations.market_data_timeout = 5.0
    config.operations.historical_data_timeout = 10.0
    
    return config


# Global configuration instance
_global_config: Optional[ReliabilityConfig] = None


def get_config() -> ReliabilityConfig:
    """Get the global configuration instance."""
    global _global_config
    
    if _global_config is None:
        # Try to load from environment variable
        config_file = os.getenv('TRADING_RELIABILITY_CONFIG')
        if config_file and os.path.exists(config_file):
            _global_config = ReliabilityConfig.load_from_file(config_file)
        else:
            # Use environment-based default
            env = os.getenv('TRADING_ENVIRONMENT', 'development').lower()
            if env == 'production':
                _global_config = get_production_config()
            elif env == 'testing':
                _global_config = get_testing_config()
            else:
                _global_config = get_development_config()
    
    return _global_config


def set_config(config: ReliabilityConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def load_config_from_env() -> ReliabilityConfig:
    """Load configuration from environment variables."""
    config = ReliabilityConfig()
    
    # Logging configuration
    config.logging.level = os.getenv('TRADING_LOG_LEVEL', config.logging.level)
    config.logging.format_type = os.getenv('TRADING_LOG_FORMAT', config.logging.format_type)
    config.logging.log_file = os.getenv('TRADING_LOG_FILE', config.logging.log_file)
    
    # Retry configuration
    if os.getenv('TRADING_MAX_RETRIES'):
        config.retry.max_attempts = int(os.getenv('TRADING_MAX_RETRIES'))
    if os.getenv('TRADING_RETRY_BASE_DELAY'):
        config.retry.base_delay = float(os.getenv('TRADING_RETRY_BASE_DELAY'))
    if os.getenv('TRADING_RETRY_MAX_DELAY'):
        config.retry.max_delay = float(os.getenv('TRADING_RETRY_MAX_DELAY'))
    
    # Circuit breaker configuration
    if os.getenv('TRADING_CB_FAILURE_THRESHOLD'):
        config.circuit_breaker.failure_threshold = int(os.getenv('TRADING_CB_FAILURE_THRESHOLD'))
    if os.getenv('TRADING_CB_TIMEOUT'):
        config.circuit_breaker.timeout_duration = float(os.getenv('TRADING_CB_TIMEOUT'))
    
    # Feature flags
    if os.getenv('TRADING_ENABLE_CIRCUIT_BREAKERS'):
        config.enable_circuit_breakers = os.getenv('TRADING_ENABLE_CIRCUIT_BREAKERS').lower() == 'true'
    if os.getenv('TRADING_ENABLE_RETRIES'):
        config.enable_retries = os.getenv('TRADING_ENABLE_RETRIES').lower() == 'true'
    
    # Environment
    config.environment = os.getenv('TRADING_ENVIRONMENT', config.environment)
    
    return config


# Configuration validation
def validate_config(config: ReliabilityConfig) -> bool:
    """Validate configuration values."""
    
    errors = []
    
    # Validate retry configuration
    if config.retry.max_attempts < 1:
        errors.append("retry.max_attempts must be at least 1")
    if config.retry.base_delay < 0:
        errors.append("retry.base_delay must be non-negative")
    if config.retry.max_delay < config.retry.base_delay:
        errors.append("retry.max_delay must be >= base_delay")
    if config.retry.multiplier <= 1.0:
        errors.append("retry.multiplier must be > 1.0")
    
    # Validate circuit breaker configuration
    if config.circuit_breaker.failure_threshold < 1:
        errors.append("circuit_breaker.failure_threshold must be at least 1")
    if config.circuit_breaker.success_threshold < 1:
        errors.append("circuit_breaker.success_threshold must be at least 1")
    if config.circuit_breaker.timeout_duration < 0:
        errors.append("circuit_breaker.timeout_duration must be non-negative")
    if not 0 <= config.circuit_breaker.failure_rate_threshold <= 1:
        errors.append("circuit_breaker.failure_rate_threshold must be between 0 and 1")
    
    # Validate logging configuration
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.logging.level not in valid_levels:
        errors.append(f"logging.level must be one of {valid_levels}")
    
    if config.logging.format_type not in ["json", "text"]:
        errors.append("logging.format_type must be 'json' or 'text'")
    
    # Validate operation timeouts
    if config.operations.ib_connection_timeout <= 0:
        errors.append("operations.ib_connection_timeout must be positive")
    
    if errors:
        raise ValueError(f"Configuration validation errors: {'; '.join(errors)}")
    
    return True