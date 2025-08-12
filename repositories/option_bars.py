"""
Option bars repository with BaseRepository integration.
Handles storage and retrieval of option historical bar data.
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


class OptionBarRepository(BaseRepository):
    """
    Repository for option bar data with Hive partitioning.
    
    Partitioning strategy:
    - All bars: partitioned by [underlying, expiry, trade_date]
    
    Expected DataFrame columns:
    - time: timestamp
    - underlying: underlying symbol (e.g., "AAPL")
    - expiry: option expiry date
    - strike: strike price
    - right: option right ("C" or "P")
    - bar_size: bar size (e.g., "1 day", "1 min")
    - open, high, low, close: OHLC prices
    - volume: trading volume
    - Optional: wap, bar_count, data_type
    """
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path, "option_bars")
    
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for option bars."""
        return pa.schema([
            pa.field('time', pa.timestamp('ns')),
            pa.field('date', pa.date32()),
            pa.field('trade_date', pa.date32()),
            pa.field('underlying', pa.string()),
            pa.field('expiry', pa.date32()),
            pa.field('strike', pa.float64()),
            pa.field('right', pa.string()),
            pa.field('bar_size', pa.string()),
            pa.field('open', pa.float64()),
            pa.field('high', pa.float64()),
            pa.field('low', pa.float64()),
            pa.field('close', pa.float64()),
            pa.field('volume', pa.float64()),
            pa.field('wap', pa.float64(), nullable=True),
            pa.field('bar_count', pa.int64(), nullable=True),
            pa.field('data_type', pa.string(), nullable=True),
            pa.field('contract_id', pa.int64(), nullable=True),
        ])
    
    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for option bars."""
        return ['underlying', 'expiry', 'trade_date']
    
    def _normalize_data(self, df: pd.DataFrame, underlying: str = None, expiry: date = None,
                       strike: float = None, right: str = None, bar_size: str = None,
                       data_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Normalize option bar data for storage.
        
        Args:
            df: Raw DataFrame with bar data
            underlying: Underlying symbol (can be inferred from df)
            expiry: Option expiry date (can be inferred from df)
            strike: Strike price (can be inferred from df)
            right: Option right "C"/"P" (can be inferred from df)
            bar_size: Bar size (can be inferred from df)
            data_type: Data type (e.g., "TRADES")
            
        Returns:
            Normalized DataFrame ready for storage
        """
        if df.empty:
            raise DataValidationException("Cannot normalize empty DataFrame")
        
        # Start with copy
        out = df.copy()
        
        # Rename standard columns
        out = self._rename_standard_columns(out)
        
        # Handle option metadata - use provided values or infer from DataFrame
        if underlying is not None:
            out['underlying'] = underlying
        elif 'underlying' not in out.columns:
            raise DataValidationException("Must provide 'underlying' parameter or column")
        
        if expiry is not None:
            out['expiry'] = expiry
        elif 'expiry' in out.columns:
            # Normalize expiry dates
            out['expiry'] = pd.to_datetime(out['expiry']).dt.date
        else:
            raise DataValidationException("Must provide 'expiry' parameter or column")
        
        if strike is not None:
            out['strike'] = float(strike)
        elif 'strike' not in out.columns:
            raise DataValidationException("Must provide 'strike' parameter or column")
        else:
            out['strike'] = pd.to_numeric(out['strike'])
        
        if right is not None:
            out['right'] = right
        elif 'right' not in out.columns:
            raise DataValidationException("Must provide 'right' parameter or column")
        
        if bar_size is not None:
            out['bar_size'] = bar_size
        elif 'bar_size' not in out.columns:
            raise DataValidationException("Must provide 'bar_size' parameter or column")
        
        if data_type:
            out['data_type'] = data_type
        
        # Validate option right
        if 'right' in out.columns:
            invalid_rights = ~out['right'].isin(['C', 'P', 'CALL', 'PUT'])
            if invalid_rights.any():
                # Normalize common variants
                out['right'] = out['right'].replace({
                    'CALL': 'C', 'Call': 'C', 'call': 'C',
                    'PUT': 'P', 'Put': 'P', 'put': 'P'
                })
                # Check again
                invalid_rights = ~out['right'].isin(['C', 'P'])
                if invalid_rights.any():
                    raise DataValidationException(
                        f"Invalid option rights found: {out['right'][invalid_rights].unique()}. "
                        "Must be 'C', 'P', 'CALL', or 'PUT'",
                        field="right"
                    )
        
        # Handle datetime columns
        if 'datetime' in out.columns:
            out['time'] = self._parse_ib_datetime_series(out['datetime'])
        elif 'time' in out.columns:
            out['time'] = pd.to_datetime(out['time'])
        else:
            raise DataValidationException(
                "DataFrame must contain 'datetime' or 'time' column",
                field="time"
            )
        
        # Derive date columns
        out['date'] = out['time'].dt.date
        out['trade_date'] = out['time'].dt.date
        
        # Ensure expiry is date type
        if 'expiry' in out.columns:
            out['expiry'] = pd.to_datetime(out['expiry']).dt.date
        
        # Validate and convert OHLCV columns
        required_numeric = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [col for col in required_numeric if col not in out.columns]
        if missing_ohlcv:
            raise DataValidationException(
                f"Missing required OHLCV columns: {missing_ohlcv}",
                field="ohlcv"
            )
        
        out = self._validate_numeric_columns(out, required_numeric + ['wap', 'strike'])
        
        # Handle bar_count as integer
        if 'bar_count' in out.columns:
            out['bar_count'] = pd.to_numeric(out['bar_count'], errors='coerce').astype('Int64')
        
        # Handle contract_id if present
        if 'contract_id' in out.columns:
            out['contract_id'] = pd.to_numeric(out['contract_id'], errors='coerce').astype('Int64')
        
        # Ensure all schema columns exist
        schema_columns = [field.name for field in self._schema]
        out = self._ensure_schema_columns(out, schema_columns)
        
        # Validate option data quality
        self._validate_option_data(out)
        
        return out
    
    def _validate_option_data(self, df: pd.DataFrame) -> None:
        """Validate option-specific data quality."""
        # Basic bar validation
        self._validate_bar_data(df)
        
        # Check strike prices are positive
        if 'strike' in df.columns:
            negative_strikes = df['strike'] <= 0
            if negative_strikes.any():
                raise DataValidationException(
                    f"Found {negative_strikes.sum()} options with non-positive strike prices",
                    field="strike"
                )
        
        # Check expiry dates are valid
        if 'expiry' in df.columns and 'trade_date' in df.columns:
            expired_options = df['expiry'] < df['trade_date']
            if expired_options.any():
                self.logger.warning(
                    f"Found {expired_options.sum()} bars where expiry < trade_date"
                )
    
    def _validate_bar_data(self, df: pd.DataFrame) -> None:
        """Validate general bar data quality (copied from equity repo)."""
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
    
    def save_option_bars(self, df: pd.DataFrame, underlying: str, expiry: date,
                        strike: float, right: str, bar_size: str,
                        data_type: str = "TRADES") -> None:
        """
        Convenience method for saving option bars with explicit parameters.
        
        Args:
            df: DataFrame with option bar data
            underlying: Underlying symbol
            expiry: Option expiry date
            strike: Strike price
            right: Option right ("C" or "P")
            bar_size: Bar size (e.g., "1 day", "1 min")
            data_type: Data type (default: "TRADES")
        """
        self.save(
            df,
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            right=right,
            bar_size=bar_size,
            data_type=data_type
        )
    
    def load_option_data(self, underlying: str, expiry: Optional[date] = None,
                        strike: Optional[float] = None, right: Optional[str] = None,
                        start_date: Optional[date] = None, end_date: Optional[date] = None,
                        bar_size: Optional[str] = None) -> pd.DataFrame:
        """
        Load option data with filtering.
        
        Args:
            underlying: Underlying symbol
            expiry: Option expiry date (optional)
            strike: Strike price (optional)
            right: Option right (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            bar_size: Bar size filter (optional)
            
        Returns:
            DataFrame with option bar data
        """
        filters = {'underlying': underlying}
        
        if expiry:
            filters['expiry'] = expiry
        if strike is not None:
            filters['strike'] = strike
        if right:
            filters['right'] = right
        if bar_size:
            filters['bar_size'] = bar_size
        
        df = self.load(**filters)
        
        # Apply date filtering if specified
        if not df.empty and (start_date or end_date):
            if start_date:
                df = df[df['trade_date'] >= start_date]
            if end_date:
                df = df[df['trade_date'] <= end_date]
        
        return df.sort_values(['time', 'strike', 'right']) if not df.empty else df
    
    def present_dates_for_contract(self, underlying: str, expiry: date, right: str,
                                  strike: float, bar_size: str, start_date: date,
                                  end_date: date) -> set:
        """
        Get dates with available data for specific option contract.
        
        Args:
            underlying: Underlying symbol
            expiry: Option expiry date
            right: Option right
            strike: Strike price
            bar_size: Bar size
            start_date: Start date
            end_date: End date
            
        Returns:
            Set of dates with available data
        """
        return super().present_dates(
            start_date=start_date,
            end_date=end_date,
            underlying=underlying,
            expiry=expiry,
            right=right,
            strike=strike,
            bar_size=bar_size
        )