"""
Equity bars repository with BaseRepository integration.
Handles storage and retrieval of equity historical bar data.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa

from .base import BaseRepository

# Import with fallback
try:
    from reliability import DataValidationException
except ImportError:
    class DataValidationException(ValueError):
        def __init__(self, message: str, field: str = None, **kwargs):
            super().__init__(message)
            self.field = field


class EquityBarRepository(BaseRepository):
    """
    Repository for equity bar data with Hive partitioning.
    
    Partitioning strategy:
    - Daily bars (bar_size="1 day"): partitioned by [symbol] 
    - Intraday bars: partitioned by [symbol, trade_date]
    
    Expected DataFrame columns:
    - time: timestamp 
    - symbol: equity symbol
    - bar_size: bar size (e.g., "1 day", "1 min")
    - open, high, low, close: OHLC prices
    - volume: trading volume
    - Optional: wap, bar_count, data_type
    """
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path, "equity_bars")
    
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for equity bars."""
        return pa.schema([
            pa.field('time', pa.timestamp('ns')),
            pa.field('date', pa.date32()),
            pa.field('trade_date', pa.date32()),
            pa.field('symbol', pa.string()),
            pa.field('bar_size', pa.string()),
            pa.field('open', pa.float64()),
            pa.field('high', pa.float64()),
            pa.field('low', pa.float64()),
            pa.field('close', pa.float64()),
            pa.field('volume', pa.float64()),
            pa.field('wap', pa.float64(), nullable=True),
            pa.field('bar_count', pa.int64(), nullable=True),
            pa.field('data_type', pa.string(), nullable=True),
        ])
    
    def _get_partition_columns(self, symbol: str = None, bar_size: str = None, **kwargs) -> List[str]:
        """Get partition columns based on bar size."""
        if bar_size and bar_size.strip().lower() == "1 day":
            return ['symbol']
        else:
            return ['symbol', 'trade_date']
    
    def _normalize_data(self, df: pd.DataFrame, symbol: str = None, bar_size: str = None, 
                       data_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Normalize equity bar data for storage.
        
        Args:
            df: Raw DataFrame with bar data
            symbol: Equity symbol
            bar_size: Bar size (e.g., "1 day", "1 min")
            data_type: Data type (e.g., "TRADES", "MIDPOINT")
            
        Returns:
            Normalized DataFrame ready for storage
        """
        if df.empty:
            raise DataValidationException("Cannot normalize empty DataFrame")
        
        # Start with copy
        out = df.copy()
        
        # Rename standard columns
        out = self._rename_standard_columns(out)
        
        # Add metadata columns - use provided values or infer from DataFrame
        if symbol is not None:
            out['symbol'] = symbol
        elif 'symbol' not in out.columns:
            raise DataValidationException("Must provide 'symbol' parameter or column")
        
        if bar_size is not None:
            out['bar_size'] = bar_size
        elif 'bar_size' not in out.columns:
            raise DataValidationException("Must provide 'bar_size' parameter or column")
        
        if data_type:
            out['data_type'] = data_type
        
        # Handle datetime columns
        if 'datetime' in out.columns:
            # IB API typically returns 'datetime' column
            out['time'] = self._parse_ib_datetime_series(out['datetime'])
        elif 'time' in out.columns:
            # Check if time is already datetime or needs parsing
            if pd.api.types.is_datetime64_any_dtype(out['time']):
                # Already datetime, keep as is
                pass
            else:
                # Parse as IB datetime string
                out['time'] = self._parse_ib_datetime_series(out['time'])
        elif 'date' in out.columns:
            # Daily bars might only have date
            out['time'] = pd.to_datetime(out['date'])
        else:
            raise DataValidationException(
                "DataFrame must contain 'datetime', 'time', or 'date' column",
                field="time"
            )
        
        # Derive date columns
        out['date'] = out['time'].dt.date
        out['trade_date'] = out['time'].dt.date
        
        # Validate and convert OHLCV columns
        required_numeric = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [col for col in required_numeric if col not in out.columns]
        if missing_ohlcv:
            raise DataValidationException(
                f"Missing required OHLCV columns: {missing_ohlcv}",
                field="ohlcv"
            )
        
        out = self._validate_numeric_columns(out, required_numeric + ['wap'])
        
        # Handle bar_count as integer
        if 'bar_count' in out.columns:
            out['bar_count'] = pd.to_numeric(out['bar_count'], errors='coerce').astype('Int64')
        
        # Ensure all schema columns exist
        schema_columns = [field.name for field in self._schema]
        out = self._ensure_schema_columns(out, schema_columns)
        
        # Validate data quality
        self._validate_bar_data(out)
        
        return out
    
    def _validate_bar_data(self, df: pd.DataFrame) -> None:
        """Validate bar data quality."""
        # Check for negative volumes
        if 'volume' in df.columns:
            negative_volume = df['volume'] < 0
            if negative_volume.any():
                raise DataValidationException(
                    f"Found {negative_volume.sum()} bars with negative volume",
                    field="volume"
                )
        
        # Check for invalid OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Low, Close
            invalid_high = (df['high'] < df[['open', 'low', 'close']].max(axis=1))
            if invalid_high.any():
                self.logger.warning(
                    f"Found {invalid_high.sum()} bars where high < max(open, low, close)"
                )
            
            # Low should be <= Open, High, Close  
            invalid_low = (df['low'] > df[['open', 'high', 'close']].min(axis=1))
            if invalid_low.any():
                self.logger.warning(
                    f"Found {invalid_low.sum()} bars where low > min(open, high, close)"
                )
    
    def save_daily_bars(self, df: pd.DataFrame, symbol: str, 
                       data_type: str = "TRADES") -> None:
        """
        Convenience method for saving daily bars.
        
        Args:
            df: DataFrame with daily bar data
            symbol: Equity symbol
            data_type: Data type (default: "TRADES")
        """
        self.save(df, symbol=symbol, bar_size="1 day", data_type=data_type)
    
    def save_intraday_bars(self, df: pd.DataFrame, symbol: str, bar_size: str,
                          data_type: str = "TRADES") -> None:
        """
        Convenience method for saving intraday bars.
        
        Args:
            df: DataFrame with intraday bar data
            symbol: Equity symbol  
            bar_size: Bar size (e.g., "1 min", "5 mins")
            data_type: Data type (default: "TRADES")
        """
        self.save(df, symbol=symbol, bar_size=bar_size, data_type=data_type)
    
    def load_symbol_data(self, symbol: str, start_date: Optional[date] = None,
                        end_date: Optional[date] = None, bar_size: Optional[str] = None) -> pd.DataFrame:
        """
        Load all data for a specific symbol with optional filtering.
        
        Args:
            symbol: Equity symbol
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            bar_size: Bar size filter (optional)
            
        Returns:
            DataFrame with symbol's bar data
        """
        filters = {'symbol': symbol}
        if bar_size:
            filters['bar_size'] = bar_size
        
        df = self.load(**filters)
        
        # Apply date filtering if specified
        if not df.empty and (start_date or end_date):
            if start_date:
                df = df[df['trade_date'] >= start_date]
            if end_date:
                df = df[df['trade_date'] <= end_date]
        
        return df.sort_values('time') if not df.empty else df
    
    def present_dates(self, symbol: str, bar_size: str, start_date: date, end_date: date) -> set:
        """
        Get dates with available data for symbol and bar size.
        
        Args:
            symbol: Equity symbol
            bar_size: Bar size
            start_date: Start date
            end_date: End date
            
        Returns:
            Set of dates with available data
        """
        return super().present_dates(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            bar_size=bar_size
        )