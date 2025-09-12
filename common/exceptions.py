"""
Common exception classes for the trading platform.
Centralizes all custom exceptions to avoid duplication.
"""


class DataValidationException(ValueError):
    """Raised when data validation fails in repositories."""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message)
        self.field = field
        self.details = kwargs
        
    def __repr__(self) -> str:
        details = f", field={self.field}" if self.field else ""
        return f"DataValidationException('{self.args[0]}'{details})"


class RepositoryException(Exception):
    """Base exception for repository operations."""
    pass


class DataIntegrityException(RepositoryException):
    """Raised when data integrity constraints are violated."""
    pass


class StorageException(RepositoryException):
    """Raised when storage operations fail."""
    pass


class ConfigurationException(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class APIException(Exception):
    """Base exception for API-related errors."""
    pass


class IBAPIException(APIException):
    """Interactive Brokers API specific exceptions."""
    
    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.error_code = error_code