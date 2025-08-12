"""
Option chain repository with BaseRepository integration.
Handles storage and retrieval of option chain snapshot data.
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


class OptionChainSnapshotRepository(BaseRepository):
    """
    Repository for option chain snapshot data with Hive partitioning.
    
    Partitioning strategy:
    - All snapshots: partitioned by [underlying, snapshot_date]
    
    Expected DataFrame columns:
    - underlying: underlying symbol (e.g., "AAPL")
    - snapshot_date: date when snapshot was taken
    - expiry: option expiry date
    - strike: strike price
    - right: option right ("C" or "P")
    - bid, ask, last: option prices
    - volume, open_interest: market data
    - Optional: implied_volatility, delta, gamma, theta, vega (Greeks)
    """
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path, "option_chains")
    
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for option chain snapshots."""
        return pa.schema([
            pa.field('underlying', pa.string()),
            pa.field('snapshot_date', pa.date32()),
            pa.field('expiry', pa.date32()),
            pa.field('strike', pa.float64()),
            pa.field('right', pa.string()),
            pa.field('bid', pa.float64(), nullable=True),
            pa.field('ask', pa.float64(), nullable=True),
            pa.field('last', pa.float64(), nullable=True),
            pa.field('volume', pa.float64(), nullable=True),
            pa.field('open_interest', pa.float64(), nullable=True),
            pa.field('implied_volatility', pa.float64(), nullable=True),
            pa.field('delta', pa.float64(), nullable=True),
            pa.field('gamma', pa.float64(), nullable=True),
            pa.field('theta', pa.float64(), nullable=True),
            pa.field('vega', pa.float64(), nullable=True),
            pa.field('contract_id', pa.int64(), nullable=True),
            pa.field('local_symbol', pa.string(), nullable=True),
            pa.field('trading_class', pa.string(), nullable=True),
            pa.field('exchange', pa.string(), nullable=True),
            pa.field('currency', pa.string(), nullable=True),
            pa.field('multiplier', pa.int64(), nullable=True),
        ])
    
    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for option chain snapshots."""
        return ['underlying', 'snapshot_date']
    
    def _normalize_data(self, df: pd.DataFrame, underlying: str = None, 
                       snapshot_date: date = None, **kwargs) -> pd.DataFrame:
        """
        Normalize option chain snapshot data for storage.
        
        Args:
            df: Raw DataFrame with option chain data
            underlying: Underlying symbol (can be inferred from df)
            snapshot_date: Snapshot date (can be inferred from df)
            
        Returns:
            Normalized DataFrame ready for storage
        """
        if df.empty:
            raise DataValidationException("Cannot normalize empty DataFrame")
        
        # Start with copy
        out = df.copy()
        
        # Rename standard columns
        out = self._rename_standard_columns(out)
        
        # Handle common column name variations
        column_mapping = {
            'asof': 'snapshot_date',
            'as_of': 'snapshot_date',
            'date': 'snapshot_date',
            'openinterest': 'open_interest',
            'openInterest': 'open_interest',
            'impliedvolatility': 'implied_volatility',
            'impliedVolatility': 'implied_volatility',
            'iv': 'implied_volatility',
            'conid': 'contract_id',
            'contractid': 'contract_id',
            'localsymbol': 'local_symbol',
            'localSymbol': 'local_symbol',
            'tradingclass': 'trading_class',
            'tradingClass': 'trading_class',
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in out.columns and new_name not in out.columns:
                out[new_name] = out[old_name]
        
        # Handle underlying and snapshot_date
        if underlying is not None:
            out['underlying'] = underlying
        elif 'underlying' not in out.columns:
            raise DataValidationException("Must provide 'underlying' parameter or column")
        
        if snapshot_date is not None:
            out['snapshot_date'] = snapshot_date
        elif 'snapshot_date' not in out.columns:
            # Try to infer from other date columns
            for col in ['asof', 'date', 'as_of']:
                if col in out.columns:
                    out['snapshot_date'] = pd.to_datetime(out[col]).dt.date
                    break
            else:
                raise DataValidationException("Must provide 'snapshot_date' parameter or column")
        else:
            # Ensure snapshot_date is date type
            out['snapshot_date'] = pd.to_datetime(out['snapshot_date']).dt.date
        
        # Validate required columns
        required_columns = ['expiry', 'strike', 'right']
        missing_required = [col for col in required_columns if col not in out.columns]
        if missing_required:
            raise DataValidationException(
                f"Missing required columns: {missing_required}",
                field="required_columns"
            )
        
        # Normalize expiry dates
        if 'expiry' in out.columns:
            out['expiry'] = pd.to_datetime(out['expiry']).dt.date
        
        # Validate and normalize strikes
        if 'strike' in out.columns:
            out['strike'] = pd.to_numeric(out['strike'], errors='coerce')
            invalid_strikes = out['strike'].isna() | (out['strike'] <= 0)
            if invalid_strikes.any():
                raise DataValidationException(
                    f"Found {invalid_strikes.sum()} invalid strike prices",
                    field="strike"
                )
        
        # Validate and normalize option rights
        if 'right' in out.columns:
            # Normalize common variants
            out['right'] = out['right'].replace({
                'CALL': 'C', 'Call': 'C', 'call': 'C',
                'PUT': 'P', 'Put': 'P', 'put': 'P'
            })
            
            invalid_rights = ~out['right'].isin(['C', 'P'])
            if invalid_rights.any():
                raise DataValidationException(
                    f"Invalid option rights found: {out['right'][invalid_rights].unique()}. "
                    "Must be 'C', 'P', 'CALL', or 'PUT'",
                    field="right"
                )
        
        # Convert numeric columns
        numeric_columns = [
            'bid', 'ask', 'last', 'volume', 'open_interest',
            'implied_volatility', 'delta', 'gamma', 'theta', 'vega'
        ]
        out = self._validate_numeric_columns(out, numeric_columns)
        
        # Handle integer columns
        if 'contract_id' in out.columns:
            out['contract_id'] = pd.to_numeric(out['contract_id'], errors='coerce').astype('Int64')
        if 'multiplier' in out.columns:
            out['multiplier'] = pd.to_numeric(out['multiplier'], errors='coerce').astype('Int64')
        
        # Ensure all schema columns exist
        schema_columns = [field.name for field in self._schema]
        out = self._ensure_schema_columns(out, schema_columns)
        
        # Validate option chain data
        self._validate_option_chain_data(out)
        
        return out
    
    def _validate_option_chain_data(self, df: pd.DataFrame) -> None:
        """Validate option chain data quality."""
        
        # Check for valid bid/ask spreads
        if 'bid' in df.columns and 'ask' in df.columns:
            # Remove rows where both bid and ask are missing
            valid_quotes = ~(df['bid'].isna() & df['ask'].isna())
            
            if valid_quotes.any():
                # Check for negative bid/ask
                negative_bid = (df['bid'] < 0) & valid_quotes
                negative_ask = (df['ask'] < 0) & valid_quotes
                
                if negative_bid.any():
                    self.logger.warning(f"Found {negative_bid.sum()} negative bid prices")
                if negative_ask.any():
                    self.logger.warning(f"Found {negative_ask.sum()} negative ask prices")
                
                # Check for inverted bid/ask spreads
                inverted_spread = (df['bid'] > df['ask']) & valid_quotes
                if inverted_spread.any():
                    self.logger.warning(
                        f"Found {inverted_spread.sum()} options with bid > ask"
                    )
        
        # Check for reasonable implied volatility values
        if 'implied_volatility' in df.columns:
            valid_iv = ~df['implied_volatility'].isna()
            if valid_iv.any():
                # IV should be positive and typically < 5.0 (500%)
                extreme_iv = ((df['implied_volatility'] <= 0) | 
                             (df['implied_volatility'] > 5.0)) & valid_iv
                if extreme_iv.any():
                    self.logger.warning(
                        f"Found {extreme_iv.sum()} options with extreme implied volatility values"
                    )
        
        # Check expiry vs snapshot date
        if 'expiry' in df.columns and 'snapshot_date' in df.columns:
            expired_options = df['expiry'] <= df['snapshot_date']
            if expired_options.any():
                self.logger.warning(
                    f"Found {expired_options.sum()} expired options in chain"
                )
    
    def save_chain_snapshot(self, df: pd.DataFrame, underlying: str, 
                           snapshot_date: date) -> None:
        """
        Convenience method for saving option chain snapshot.
        
        Args:
            df: DataFrame with option chain data
            underlying: Underlying symbol
            snapshot_date: Date when snapshot was taken
        """
        self.save(df, underlying=underlying, snapshot_date=snapshot_date)
    
    def load_chain_snapshot(self, underlying: str, snapshot_date: date) -> pd.DataFrame:
        """
        Load option chain snapshot for specific underlying and date.
        
        Args:
            underlying: Underlying symbol
            snapshot_date: Snapshot date
            
        Returns:
            DataFrame with option chain data
        """
        df = self.load(underlying=underlying, snapshot_date=snapshot_date)
        return df.sort_values(['expiry', 'strike', 'right']) if not df.empty else df
    
    def load_chains_for_expiry(self, underlying: str, expiry: date,
                              start_date: Optional[date] = None,
                              end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Load option chains for specific expiry across date range.
        
        Args:
            underlying: Underlying symbol
            expiry: Option expiry date
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            DataFrame with option chain data
        """
        df = self.load(underlying=underlying)
        
        if df.empty:
            return df
        
        # Filter by expiry
        df = df[df['expiry'] == expiry]
        
        # Apply date filtering if specified
        if start_date:
            df = df[df['snapshot_date'] >= start_date]
        if end_date:
            df = df[df['snapshot_date'] <= end_date]
        
        return df.sort_values(['snapshot_date', 'strike', 'right']) if not df.empty else df
    
    def get_available_expiries(self, underlying: str, 
                              snapshot_date: Optional[date] = None) -> List[date]:
        """
        Get list of available option expiries for underlying.
        
        Args:
            underlying: Underlying symbol
            snapshot_date: Specific snapshot date (optional)
            
        Returns:
            List of available expiry dates
        """
        filters = {'underlying': underlying}
        if snapshot_date:
            filters['snapshot_date'] = snapshot_date
        
        df = self.load(**filters)
        
        if df.empty:
            return []
        
        expiries = df['expiry'].unique()
        return sorted([exp for exp in expiries if pd.notna(exp)])
    
    def get_strike_range(self, underlying: str, expiry: date,
                        snapshot_date: Optional[date] = None) -> tuple:
        """
        Get strike price range for specific expiry.
        
        Args:
            underlying: Underlying symbol
            expiry: Option expiry date
            snapshot_date: Specific snapshot date (optional)
            
        Returns:
            Tuple of (min_strike, max_strike)
        """
        filters = {'underlying': underlying}
        if snapshot_date:
            filters['snapshot_date'] = snapshot_date
        
        df = self.load(**filters)
        
        if df.empty:
            return (None, None)
        
        # Filter by expiry
        df = df[df['expiry'] == expiry]
        
        if df.empty:
            return (None, None)
        
        return (df['strike'].min(), df['strike'].max())