"""Testing environment configuration."""

from pathlib import Path
from ..base import AppConfig, APIConfig, LoggingConfig, IBAPIConfig, DatabaseConfig


class TestingConfig(AppConfig):
    """Testing-specific configuration."""
    
    environment: str = "testing"
    debug: bool = True
    
    # Testing database settings
    database: DatabaseConfig = DatabaseConfig(
        data_root=Path("data/test"),  # Separate test data directory
        max_rows_per_partition=1000,  # Smaller partitions for tests
        compression="none"  # Faster for small test datasets
    )
    
    # Testing API settings
    api: APIConfig = APIConfig(
        port=8001,  # Different port for testing
        workers=1,  # Single worker for predictable testing
        reload=False,  # No auto-reload during tests
        log_level="WARNING",  # Reduce test noise
        rate_limit_enabled=False,  # No rate limiting in tests
        response_timeout=10.0  # Shorter timeout for tests
    )
    
    # Testing IB API settings (mock/disabled)
    ibapi: IBAPIConfig = IBAPIConfig(
        port=7498,  # Different port for testing
        timeout=5.0,  # Short timeout for fast test failures
        max_requests_per_second=50,  # Maximum allowed rate for tests
        max_connection_attempts=1,  # Fail fast in tests
        connection_retry_delay=1.0
    )
    
    # Testing logging (minimal)
    logging: LoggingConfig = LoggingConfig(
        level="ERROR",  # Only log errors during tests
        console_enabled=False,  # No console output during tests
        file_enabled=True,
        file_path=Path("data/test/logs/test.log"),
        json_format=True,  # Structured for test analysis
        include_correlation_id=True
    )
    
    # Testing features
    features: dict = {
        "metrics_enabled": False,  # Disable for test performance
        "caching_enabled": False,  # Disable for test predictability
        "validation_enabled": True,  # Important for testing data quality
        "partitioning_enabled": False,  # Simpler for tests
        "test_mode": True,
        "mock_external_apis": True,
        "fast_teardown": True
    }
    
    def setup_test_environment(self) -> None:
        """Set up the testing environment."""
        # Ensure test directories exist
        self.ensure_directories()
        
        # Additional test-specific setup
        test_dirs = [
            self.database.data_root / "fixtures",
            self.database.data_root / "temp",
            self.database.data_root / "mocks",
        ]
        
        for directory in test_dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def cleanup_test_environment(self) -> None:
        """Clean up after tests."""
        import shutil
        
        # Remove test data directory
        if self.database.data_root.exists():
            shutil.rmtree(self.database.data_root, ignore_errors=True)