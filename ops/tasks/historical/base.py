"""
Base utilities and mixins for historical data tasks.

This module provides common functionality used across historical data tasks,
including rate limiting, error handling, and data validation utilities.

TODO: Extract shared functionality from historical_data.py into this module.
"""

from typing import Any, Dict, List
from common.exceptions import DataValidationException


class HistoricalDataMixin:
    """Common functionality for historical data tasks."""
    
    def validate_historical_request(self, contract: Dict[str, Any], **kwargs) -> None:
        """Validate historical data request parameters."""
        required_fields = ['symbol', 'secType', 'exchange', 'currency']
        missing_fields = [field for field in required_fields if not contract.get(field)]
        
        if missing_fields:
            raise DataValidationException(
                f"Missing required contract fields: {missing_fields}",
                field="contract"
            )
    
    def normalize_contract_dict(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize contract dictionary for consistent processing."""
        normalized = contract.copy()
        
        # Ensure required fields have default values
        normalized.setdefault('exchange', 'SMART')
        normalized.setdefault('currency', 'USD')
        
        # Normalize symbol to uppercase
        if 'symbol' in normalized:
            normalized['symbol'] = str(normalized['symbol']).upper().strip()
            
        return normalized


def validate_date_range(start_date: str, end_date: str) -> None:
    """Validate that date range is reasonable for historical data requests."""
    # TODO: Implement date range validation
    pass


def calculate_rate_limit_delay(request_count: int) -> float:
    """Calculate appropriate delay between requests based on IB API limits."""
    # IB API limit: 2 requests per second for historical data
    base_delay = 0.5  # 500ms for 2/second
    
    # Add exponential backoff for burst requests
    if request_count > 10:
        return base_delay * (1.2 ** (request_count - 10))
    
    return base_delay