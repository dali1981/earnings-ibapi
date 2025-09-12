"""Configuration factory and settings access."""

import os
import logging
from typing import Type, Dict, Any, Optional
from .base import AppConfig
from .environments import DevelopmentConfig, TestingConfig, ProductionConfig

logger = logging.getLogger(__name__)

# Configuration mapping
CONFIG_MAP: Dict[str, Type[AppConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig
}


def get_config(environment: Optional[str] = None) -> AppConfig:
    """Get configuration based on environment.
    
    Args:
        environment: Environment name (overrides APP_ENV if provided)
        
    Returns:
        Configuration instance for the environment
        
    Raises:
        ValueError: If environment is unknown
    """
    env = environment or os.getenv("APP_ENV", "development")
    
    if env not in CONFIG_MAP:
        available_envs = list(CONFIG_MAP.keys())
        raise ValueError(
            f"Unknown environment: '{env}'. "
            f"Available environments: {available_envs}"
        )
    
    config_class = CONFIG_MAP[env]
    
    try:
        config = config_class()
        
        # Perform environment-specific validation
        if hasattr(config, 'validate_production_settings') and env == 'production':
            validation_errors = config.validate_production_settings()
            if validation_errors:
                logger.warning(f"Production configuration issues: {validation_errors}")
        
        # Ensure required directories exist
        config.ensure_directories()
        
        logger.info(f"Loaded {env} configuration")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load {env} configuration: {e}")
        raise


def reload_config() -> AppConfig:
    """Reload configuration from environment.
    
    Returns:
        New configuration instance
    """
    global settings
    settings = get_config()
    logger.info("Configuration reloaded")
    return settings


def register_environment(name: str, config_class: Type[AppConfig]) -> None:
    """Register a custom environment configuration.
    
    Args:
        name: Environment name
        config_class: Configuration class
    """
    CONFIG_MAP[name] = config_class
    logger.info(f"Registered custom environment: {name}")


def get_available_environments() -> list[str]:
    """Get list of available environment names.
    
    Returns:
        List of environment names
    """
    return list(CONFIG_MAP.keys())


def validate_environment(environment: str) -> bool:
    """Check if an environment is valid.
    
    Args:
        environment: Environment name to validate
        
    Returns:
        True if environment is valid, False otherwise
    """
    return environment in CONFIG_MAP


def create_config_from_dict(config_dict: Dict[str, Any]) -> AppConfig:
    """Create configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration instance
    """
    # Use development config as base and override with provided values
    config = DevelopmentConfig(**config_dict)
    return config


def export_config(config: Optional[AppConfig] = None, format: str = "dict") -> Any:
    """Export configuration in specified format.
    
    Args:
        config: Configuration to export (uses global settings if None)
        format: Export format ('dict', 'json', 'yaml', 'env')
        
    Returns:
        Configuration in requested format
    """
    if config is None:
        config = settings
    
    if format == "dict":
        return config.model_dump()
    
    elif format == "json":
        return config.model_dump_json(indent=2)
    
    elif format == "yaml":
        import yaml
        return yaml.dump(config.model_dump(), default_flow_style=False)
    
    elif format == "env":
        # Generate .env file content
        return _generate_env_content(config)
    
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _generate_env_content(config: AppConfig) -> str:
    """Generate .env file content from configuration.
    
    Args:
        config: Configuration instance
        
    Returns:
        .env file content as string
    """
    lines = [
        "# Trading Platform Configuration",
        f"APP_ENV={config.environment}",
        f"APP_DEBUG={config.debug}",
        f"APP_VERSION={config.version}",
        "",
        "# Database Configuration",
        f"DB_DATA_ROOT={config.database.data_root}",
        f"DB_PARQUET_ENGINE={config.database.parquet_engine}",
        f"DB_MAX_ROWS_PER_PARTITION={config.database.max_rows_per_partition}",
        "",
        "# IB API Configuration", 
        f"IBAPI_HOST={config.ibapi.host}",
        f"IBAPI_PORT={config.ibapi.port}",
        f"IBAPI_CLIENT_ID={config.ibapi.client_id}",
        f"IBAPI_TIMEOUT={config.ibapi.timeout}",
        f"IBAPI_MAX_REQUESTS_PER_SECOND={config.ibapi.max_requests_per_second}",
        "",
        "# API Configuration",
        f"API_HOST={config.api.host}",
        f"API_PORT={config.api.port}",
        f"API_WORKERS={config.api.workers}",
        f"API_RELOAD={config.api.reload}",
        f"API_LOG_LEVEL={config.api.log_level}",
        "",
        "# Logging Configuration",
        f"LOG_LEVEL={config.logging.level}",
        f"LOG_FILE_ENABLED={config.logging.file_enabled}",
        f"LOG_FILE_PATH={config.logging.file_path}",
        f"LOG_CONSOLE_ENABLED={config.logging.console_enabled}",
        f"LOG_JSON_FORMAT={config.logging.json_format}",
    ]
    
    return "\n".join(lines)


# Global settings instance
settings = get_config()

# Backward compatibility aliases
def get_data_root() -> str:
    """Get data root directory path.
    
    Returns:
        Data root path as string
    """
    return str(settings.database.data_root)


def get_ib_config() -> dict:
    """Get IB API configuration as dictionary.
    
    Returns:
        IB API configuration dictionary
    """
    return {
        "host": settings.ibapi.host,
        "port": settings.ibapi.port,
        "client_id": settings.ibapi.client_id,
        "timeout": settings.ibapi.timeout,
    }


def get_api_config() -> dict:
    """Get API server configuration as dictionary.
    
    Returns:
        API server configuration dictionary
    """
    return {
        "host": settings.api.host,
        "port": settings.api.port,
        "workers": settings.api.workers,
        "log_level": settings.api.log_level,
    }


def is_development() -> bool:
    """Check if running in development environment.
    
    Returns:
        True if development, False otherwise
    """
    return settings.environment == "development"


def is_production() -> bool:
    """Check if running in production environment.
    
    Returns:
        True if production, False otherwise
    """
    return settings.environment == "production"


def is_testing() -> bool:
    """Check if running in testing environment.
    
    Returns:
        True if testing, False otherwise
    """
    return settings.environment == "testing"