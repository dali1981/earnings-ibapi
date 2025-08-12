"""
Base repository class for trading data persistence with PyArrow/Parquet backend.
Provides common functionality for data normalization, schema validation, and Parquet operations.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Import reliability components with fallback for development
try:
    from reliability import (
        get_trading_logger, log_context, DataValidationException,
        StorageException, performance_log
    )
except ImportError:
    # Fallback implementations for development/testing
    import logging
    from contextlib import contextmanager
    
    def get_trading_logger(name: str):
        return logging.getLogger(name)
    
    @contextmanager
    def log_context(**kwargs):
        yield
    
    class DataValidationException(ValueError):
        def __init__(self, message: str, field: str = None, **kwargs):
            super().__init__(message)
            self.field = field
            self.context = kwargs
    
    class StorageException(Exception):
        def __init__(self, message: str, storage_type: str = None, **kwargs):
            super().__init__(message)
            self.storage_type = storage_type
            self.context = kwargs
    
    def performance_log(operation: str):
        def decorator(func):
            return func
        return decorator


class BaseRepository(ABC):
    """
    Abstract base class for trading data repositories.
    
    Provides common functionality for:
    - PyArrow schema management
    - Parquet dataset operations
    - Data normalization and validation
    - Hive partitioning
    - Error handling and logging
    """
    
    def __init__(self, base_path: Union[str, Path], dataset_name: str):
        """
        Initialize repository with base path and dataset name.
        
        Args:
            base_path: Base directory for storing data
            dataset_name: Name of the dataset (e.g., 'equity_bars', 'option_bars')
        """
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.dataset_path = self.base_path / dataset_name
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_trading_logger(f"{self.__class__.__name__}")
        
        # Initialize schema - must be implemented by subclasses
        self._schema = self._create_schema()
        
        # Create empty dataset if it doesn't exist
        if not any(self.dataset_path.rglob("*.parquet")):
            self._write_empty_dataset()
    
    @abstractmethod
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for the repository. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for the data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Normalize and validate data. Must be implemented by subclasses."""
        pass
    
    @property
    def schema(self) -> pa.Schema:
        """Get the PyArrow schema for this repository."""
        return self._schema
    
    def _write_empty_dataset(self) -> None:
        """Write an empty Parquet dataset to initialize the storage."""
        try:
            empty_df = pd.DataFrame(columns=[field.name for field in self._schema])
            empty_table = pa.Table.from_pandas(empty_df, schema=self._schema)
            
            pq.write_to_dataset(
                empty_table,
                root_path=str(self.dataset_path),
                partition_cols=[],
                use_dictionary=True,
                compression='snappy'
            )
            
            self.logger.debug(f"Created empty dataset at {self.dataset_path}")
            
        except Exception as e:
            raise StorageException(
                f"Failed to create empty dataset: {e}",
                storage_type="parquet",
                context={"dataset_path": str(self.dataset_path)}
            )
    
    def _build_filter_expression(self, **kwargs) -> Optional[ds.Expression]:
        """
        Build PyArrow dataset filter expression from keyword arguments.
        
        Handles date normalization automatically for *_date columns.
        """
        expr = None
        
        for key, value in kwargs.items():
            if value is None:
                continue
                
            # Normalize date-like columns
            if key.endswith('_date') or key == 'date':
                value = self._normalize_date_value(value)
            
            term = ds.field(key) == value
            expr = term if expr is None else (expr & term)
        
        return expr
    
    def _normalize_date_value(self, value: Union[str, date, datetime, pd.Timestamp]) -> str:
        """Normalize date value to ISO format string for Hive partitioning."""
        if isinstance(value, str):
            try:
                value = pd.to_datetime(value).date()
            except Exception:
                return value
        
        if isinstance(value, (pd.Timestamp, datetime)):
            value = value.date()
            
        if isinstance(value, date):
            return value.isoformat()
            
        return str(value)
    
    def _parse_ib_datetime_series(self, series: pd.Series) -> pd.Series:
        """
        Parse IB datetime strings to pandas datetime series.
        
        Handles formats like:
        - 'YYYYMMDD'
        - 'YYYYMMDD HH:MM:SS' 
        - 'YYYYMMDD HH:MM:SS TZ'
        """
        if series is None or series.empty:
            return pd.Series(pd.NaT, index=series.index if series is not None else [])
        
        series = series.astype(str)
        
        def canonicalize(x: str) -> str:
            """Convert to canonical format for parsing."""
            # Remove timezone info and extra whitespace
            parts = x.strip().split()
            if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
                # Check if second part is time format
                if re.match(r'^\d{2}:\d{2}:\d{2}$', parts[1]):
                    return f"{parts[0]} {parts[1]}"
                # Daily format
                return parts[0]
            return x
        
        canonical = series.map(canonicalize)
        
        # Try intraday format first
        ts = pd.to_datetime(canonical, format="%Y%m%d %H:%M:%S", errors="coerce")
        
        # Try daily format for failed parses
        mask = ts.isna()
        if mask.any():
            ts_daily = pd.to_datetime(canonical[mask], format="%Y%m%d", errors="coerce")
            ts[mask] = ts_daily
        
        # Final fallback
        mask = ts.isna()
        if mask.any():
            ts_fallback = pd.to_datetime(canonical[mask], errors="coerce")
            ts[mask] = ts_fallback
        
        return ts
    
    def _rename_standard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename standard columns to consistent naming."""
        columns = {c: c for c in df.columns}
        lower_map = {c.lower(): c for c in df.columns}
        
        # Standard renames
        rename_map = {
            'barcount': 'bar_count',
            'what_to_show': 'data_type',
            'whattosshow': 'data_type',
        }
        
        for std_name, actual_name in rename_map.items():
            if std_name in lower_map:
                columns[lower_map[std_name]] = actual_name
        
        return df.rename(columns=columns, errors='ignore')
    
    def _ensure_schema_columns(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Ensure all required columns exist, adding with NA values if missing."""
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    
    def _validate_numeric_columns(self, df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Convert and validate numeric columns."""
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    @performance_log("repository_save")
    def save(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Save DataFrame to Parquet dataset with proper partitioning.
        
        Args:
            df: DataFrame to save
            **kwargs: Additional parameters for normalization (symbol, bar_size, etc.)
        """
        if df is None or df.empty:
            self.logger.warning("Attempting to save empty DataFrame")
            return
        
        with log_context(
            operation="repository_save",
            repository=self.dataset_name,
            record_count=len(df)
        ):
            try:
                # Normalize and validate data
                normalized_df = self._normalize_data(df.copy(), **kwargs)
                
                # Validate against schema
                self._validate_dataframe_schema(normalized_df)
                
                # Create PyArrow table
                table = pa.Table.from_pandas(
                    normalized_df, 
                    schema=self._schema, 
                    preserve_index=False
                )
                
                # Determine partition columns
                partition_cols = self._get_partition_columns(**kwargs)
                
                # Write to dataset
                pq.write_to_dataset(
                    table,
                    root_path=str(self.dataset_path),
                    partition_cols=partition_cols,
                    use_dictionary=True,
                    compression='snappy',
                    existing_data_behavior='overwrite_or_ignore'
                )
                
                self.logger.info(
                    f"Saved {len(normalized_df)} records to {self.dataset_name}",
                    extra={
                        'record_count': len(normalized_df),
                        'partition_cols': partition_cols,
                        'table_size_bytes': table.nbytes
                    }
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to save data to {self.dataset_name}: {e}",
                    extra={'error_type': type(e).__name__}
                )
                if isinstance(e, (DataValidationException, StorageException)):
                    raise
                else:
                    raise StorageException(
                        f"Failed to save data: {e}",
                        storage_type="parquet",
                        context={
                            'dataset': self.dataset_name,
                            'record_count': len(df) if df is not None else 0
                        }
                    )
    
    @performance_log("repository_load")
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load data from Parquet dataset with filtering.
        
        Args:
            **kwargs: Filter parameters (symbol, date, etc.)
            
        Returns:
            Filtered DataFrame
        """
        with log_context(
            operation="repository_load",
            repository=self.dataset_name,
            filters=kwargs
        ):
            try:
                # Create dataset
                dataset = ds.dataset(
                    str(self.dataset_path), 
                    format="parquet",
                    partitioning="hive",
                    schema=self._schema
                )
                
                # Build filter expression
                filter_expr = self._build_filter_expression(**kwargs)
                
                # Load data
                table = dataset.to_table(filter=filter_expr)
                df = table.to_pandas()
                
                self.logger.debug(
                    f"Loaded {len(df)} records from {self.dataset_name}",
                    extra={
                        'record_count': len(df),
                        'filter_params': kwargs
                    }
                )
                
                return df
                
            except Exception as e:
                self.logger.error(
                    f"Failed to load data from {self.dataset_name}: {e}",
                    extra={'error_type': type(e).__name__, 'filters': kwargs}
                )
                raise StorageException(
                    f"Failed to load data: {e}",
                    storage_type="parquet",
                    context={
                        'dataset': self.dataset_name,
                        'filters': kwargs
                    }
                )
    
    def _validate_dataframe_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame against repository schema."""
        schema_columns = {field.name for field in self._schema}
        df_columns = set(df.columns)
        
        # Check for required columns (non-nullable fields)
        required_columns = {
            field.name for field in self._schema 
            if not field.nullable and field.name in df_columns
        }
        
        missing_required = required_columns - df_columns
        if missing_required:
            raise DataValidationException(
                f"Missing required columns: {missing_required}",
                context={
                    'missing_columns': list(missing_required),
                    'available_columns': list(df_columns),
                    'repository': self.dataset_name
                }
            )
        
        # Check for unexpected columns
        extra_columns = df_columns - schema_columns
        if extra_columns:
            self.logger.warning(
                f"DataFrame contains extra columns not in schema: {extra_columns}",
                extra={'extra_columns': list(extra_columns)}
            )
    
    def present_dates(
        self,
        start_date: Union[date, str, pd.Timestamp],
        end_date: Union[date, str, pd.Timestamp],
        **filter_kwargs
    ) -> Set[date]:
        """
        Get set of dates that have data in the specified date range.
        
        Args:
            start_date: Start date for range
            end_date: End date for range  
            **filter_kwargs: Additional filters (symbol, etc.)
            
        Returns:
            Set of dates with available data
        """
        try:
            # Normalize date inputs
            start_date = self._normalize_date_value(start_date)
            end_date = self._normalize_date_value(end_date)
            
            # Load data with filters
            df = self.load(**filter_kwargs)
            
            if df.empty:
                return set()
            
            # Find date column (try common names)
            date_col = None
            for col_name in ['trade_date', 'date', 'snapshot_date']:
                if col_name in df.columns:
                    date_col = col_name
                    break
            
            if date_col is None:
                self.logger.warning("No date column found for present_dates query")
                return set()
            
            # Filter by date range and return unique dates
            date_series = pd.to_datetime(df[date_col]).dt.date
            mask = (date_series >= pd.to_datetime(start_date).date()) & \
                   (date_series <= pd.to_datetime(end_date).date())
            
            return set(date_series[mask])
            
        except Exception as e:
            self.logger.error(f"Failed to get present dates: {e}")
            return set()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        try:
            dataset = ds.dataset(
                str(self.dataset_path), 
                format="parquet",
                partitioning="hive"
            )
            
            # Get basic stats
            total_rows = 0
            total_size = 0
            file_count = 0
            
            for fragment in dataset.get_fragments():
                metadata = fragment.metadata
                if metadata:
                    total_rows += metadata.num_rows
                    total_size += metadata.serialized_size
                    file_count += 1
            
            return {
                'dataset_name': self.dataset_name,
                'total_rows': total_rows,
                'total_size_bytes': total_size,
                'file_count': file_count,
                'dataset_path': str(self.dataset_path),
                'schema_columns': len(self._schema),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {e}")
            return {
                'dataset_name': self.dataset_name,
                'error': str(e)
            }