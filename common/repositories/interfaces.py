"""Repository interface protocols for type safety."""

from typing import Protocol, TypeVar, Generic, List, Dict, Any, Optional, Union
from datetime import date
import pandas as pd
import pyarrow as pa

T = TypeVar('T')


class Repository(Protocol[T]):
    """Base repository protocol defining core CRUD operations."""
    
    def save(self, data: pd.DataFrame, **kwargs) -> None:
        """Save data to repository.
        
        Args:
            data: DataFrame to save
            **kwargs: Additional save parameters (e.g., symbol, overwrite)
        """
        ...
        
    def load(self, filters: Optional[List[tuple]] = None, **kwargs) -> pd.DataFrame:
        """Load data from repository.
        
        Args:
            filters: List of filter tuples (column, operator, value)
            **kwargs: Additional load parameters
            
        Returns:
            DataFrame containing filtered data
        """
        ...
        
    def exists(self, **kwargs) -> bool:
        """Check if data exists in repository.
        
        Args:
            **kwargs: Criteria for existence check
            
        Returns:
            True if data exists, False otherwise
        """
        ...
        
    def delete(self, **kwargs) -> None:
        """Delete data from repository.
        
        Args:
            **kwargs: Criteria for deletion
        """
        ...
    
    def get_schema(self) -> pa.Schema:
        """Get the PyArrow schema for this repository.
        
        Returns:
            PyArrow schema definition
        """
        ...


class TimeSeriesRepository(Repository[T], Protocol):
    """Protocol for repositories storing time series data."""
    
    def load_date_range(
        self, 
        start_date: date, 
        end_date: date, 
        **kwargs
    ) -> pd.DataFrame:
        """Load data for a specific date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame containing data for the date range
        """
        ...
        
    def get_available_dates(self, **kwargs) -> List[date]:
        """Get list of available dates in the repository.
        
        Args:
            **kwargs: Filter criteria (e.g., symbol)
            
        Returns:
            List of dates for which data is available
        """
        ...
    
    def get_date_range(self, **kwargs) -> Optional[tuple[date, date]]:
        """Get the min/max date range available.
        
        Args:
            **kwargs: Filter criteria
            
        Returns:
            Tuple of (min_date, max_date) or None if no data
        """
        ...


class SymbolRepository(Repository[T], Protocol):
    """Protocol for repositories that organize data by trading symbols."""
    
    def load_symbol(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data for a specific symbol.
        
        Args:
            symbol: Trading symbol to load
            **kwargs: Additional parameters
            
        Returns:
            DataFrame containing data for the symbol
        """
        ...
        
    def get_symbols(self) -> List[str]:
        """Get list of available symbols in the repository.
        
        Returns:
            List of symbols for which data is available
        """
        ...
    
    def symbol_exists(self, symbol: str) -> bool:
        """Check if a symbol has data in the repository.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if symbol has data, False otherwise
        """
        ...


class PartitionedRepository(Repository[T], Protocol):
    """Protocol for repositories that use partitioning for performance."""
    
    def get_partitions(self) -> List[Dict[str, Any]]:
        """Get information about repository partitions.
        
        Returns:
            List of partition metadata dictionaries
        """
        ...
    
    def get_partition_columns(self) -> List[str]:
        """Get the columns used for partitioning.
        
        Returns:
            List of column names used for partitioning
        """
        ...
    
    def load_partition(self, partition_filter: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Load data from a specific partition.
        
        Args:
            partition_filter: Dictionary specifying partition values
            **kwargs: Additional parameters
            
        Returns:
            DataFrame containing partition data
        """
        ...


class CacheableRepository(Repository[T], Protocol):
    """Protocol for repositories that support caching."""
    
    def clear_cache(self, **kwargs) -> None:
        """Clear repository cache.
        
        Args:
            **kwargs: Cache clearing criteria
        """
        ...
    
    def cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        ...
    
    def is_cached(self, **kwargs) -> bool:
        """Check if specific data is cached.
        
        Args:
            **kwargs: Criteria for cache check
            
        Returns:
            True if data is cached, False otherwise
        """
        ...


class ValidationRepository(Repository[T], Protocol):
    """Protocol for repositories that validate data on save."""
    
    def validate_data(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """Validate data before saving.
        
        Args:
            data: DataFrame to validate
            **kwargs: Validation parameters
            
        Returns:
            List of validation error messages (empty if valid)
        """
        ...
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules for this repository.
        
        Returns:
            Dictionary describing validation rules
        """
        ...


class MetricsRepository(Repository[T], Protocol):
    """Protocol for repositories that provide usage metrics."""
    
    def get_metrics(self) -> Dict[str, Union[int, float, str]]:
        """Get repository usage metrics.
        
        Returns:
            Dictionary with metrics like record count, size, etc.
        """
        ...
    
    def get_size_bytes(self) -> int:
        """Get repository size in bytes.
        
        Returns:
            Size of repository data in bytes
        """
        ...
    
    def get_record_count(self, **kwargs) -> int:
        """Get number of records in repository.
        
        Args:
            **kwargs: Filter criteria for counting
            
        Returns:
            Number of records matching criteria
        """
        ...


# Combined protocols for common repository patterns
class MarketDataRepository(
    TimeSeriesRepository[T], 
    SymbolRepository[T], 
    PartitionedRepository[T], 
    Protocol
):
    """Combined protocol for market data repositories (bars, ticks, etc.)."""
    
    def load_symbol_date_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        **kwargs
    ) -> pd.DataFrame:
        """Load data for a symbol within a date range.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame containing symbol data for the date range
        """
        ...


class ContractRepository(
    SymbolRepository[T],
    ValidationRepository[T], 
    Protocol
):
    """Protocol for contract/instrument repositories."""
    
    def get_contract_types(self) -> List[str]:
        """Get available contract types (STK, OPT, etc.).
        
        Returns:
            List of contract type strings
        """
        ...
    
    def load_by_contract_type(self, contract_type: str, **kwargs) -> pd.DataFrame:
        """Load contracts by type.
        
        Args:
            contract_type: Contract type to filter by
            **kwargs: Additional filters
            
        Returns:
            DataFrame containing contracts of the specified type
        """
        ...