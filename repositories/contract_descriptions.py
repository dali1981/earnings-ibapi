"""
Contract descriptions repository with BaseRepository integration.
Handles storage and retrieval of IB contract description data.
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


class ContractDescriptionsRepository(BaseRepository):
    """
    Repository for IB contract descriptions with Hive partitioning.
    
    Partitioning strategy:
    - All descriptions: partitioned by [symbol, sec_type]
    
    Expected DataFrame columns:
    - symbol: contract symbol
    - sec_type: security type (STK, OPT, FUT, etc.)
    - contract_id: IB contract ID
    - exchange: primary exchange
    - currency: contract currency
    - Optional: local_symbol, trading_class, multiplier, etc.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path, "contract_descriptions")
    
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for contract descriptions."""
        return pa.schema([
            pa.field('symbol', pa.string()),
            pa.field('sec_type', pa.string()),
            pa.field('contract_id', pa.int64()),
            pa.field('exchange', pa.string(), nullable=True),
            pa.field('primary_exchange', pa.string(), nullable=True),
            pa.field('currency', pa.string(), nullable=True),
            pa.field('local_symbol', pa.string(), nullable=True),
            pa.field('trading_class', pa.string(), nullable=True),
            pa.field('multiplier', pa.int64(), nullable=True),
            pa.field('min_tick', pa.float64(), nullable=True),
            pa.field('price_magnifier', pa.int64(), nullable=True),
            pa.field('under_contract_id', pa.int64(), nullable=True),
            pa.field('long_name', pa.string(), nullable=True),
            pa.field('industry', pa.string(), nullable=True),
            pa.field('category', pa.string(), nullable=True),
            pa.field('subcategory', pa.string(), nullable=True),
            pa.field('timezone_id', pa.string(), nullable=True),
            pa.field('trading_hours', pa.string(), nullable=True),
            pa.field('liquid_hours', pa.string(), nullable=True),
            pa.field('market_data_size_multiplier', pa.int64(), nullable=True),
            pa.field('agg_group', pa.int64(), nullable=True),
            pa.field('under_symbol', pa.string(), nullable=True),
            pa.field('under_sec_type', pa.string(), nullable=True),
            pa.field('market_rule_ids', pa.string(), nullable=True),
            pa.field('real_expiration_date', pa.string(), nullable=True),
            pa.field('last_trade_time', pa.string(), nullable=True),
            pa.field('stock_type', pa.string(), nullable=True),
            pa.field('created_date', pa.date32()),
        ])
    
    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for contract descriptions."""
        return ['symbol', 'sec_type']
    
    def _normalize_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Normalize contract description data for storage.
        
        Args:
            df: Raw DataFrame with contract description data
            
        Returns:
            Normalized DataFrame ready for storage
        """
        if df.empty:
            raise DataValidationException("Cannot normalize empty DataFrame")
        
        # Start with copy
        out = df.copy()
        
        # Rename standard columns
        out = self._rename_standard_columns(out)
        
        # Handle common column name variations for IB contract details
        column_mapping = {
            'conid': 'contract_id',
            'conId': 'contract_id',
            'contractid': 'contract_id',
            'contractId': 'contract_id',
            'sectype': 'sec_type',
            'secType': 'sec_type',
            'primaryexchange': 'primary_exchange',
            'primaryExchange': 'primary_exchange',
            'localsymbol': 'local_symbol',
            'localSymbol': 'local_symbol',
            'tradingclass': 'trading_class',
            'tradingClass': 'trading_class',
            'mintick': 'min_tick',
            'minTick': 'min_tick',
            'pricemagnifier': 'price_magnifier',
            'priceMagnifier': 'price_magnifier',
            'undercontractid': 'under_contract_id',
            'underConId': 'under_contract_id',
            'longname': 'long_name',
            'longName': 'long_name',
            'timezoneid': 'timezone_id',
            'timezoneId': 'timezone_id',
            'tradinghours': 'trading_hours',
            'tradingHours': 'trading_hours',
            'liquidhours': 'liquid_hours',
            'liquidHours': 'liquid_hours',
            'marketdatasizemultiplier': 'market_data_size_multiplier',
            'mdSizeMultiplier': 'market_data_size_multiplier',
            'agggroup': 'agg_group',
            'aggGroup': 'agg_group',
            'undersymbol': 'under_symbol',
            'underSymbol': 'under_symbol',
            'undersectype': 'under_sec_type',
            'underSecType': 'under_sec_type',
            'marketruleids': 'market_rule_ids',
            'marketRuleIds': 'market_rule_ids',
            'realexpirationdate': 'real_expiration_date',
            'realExpirationDate': 'real_expiration_date',
            'lasttradetime': 'last_trade_time',
            'lastTradeTime': 'last_trade_time',
            'stocktype': 'stock_type',
            'stockType': 'stock_type',
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in out.columns and new_name not in out.columns:
                out[new_name] = out[old_name]
        
        # Validate required columns
        required_columns = ['symbol', 'contract_id']
        missing_required = [col for col in required_columns if col not in out.columns]
        if missing_required:
            raise DataValidationException(
                f"Missing required columns: {missing_required}",
                field="required_columns"
            )
        
        # Normalize sec_type
        if 'sec_type' not in out.columns:
            # Default to STK if not specified
            out['sec_type'] = 'STK'
            self.logger.warning("sec_type not provided, defaulting to 'STK'")
        
        # Validate contract_id
        if 'contract_id' in out.columns:
            out['contract_id'] = pd.to_numeric(out['contract_id'], errors='coerce')
            invalid_contract_ids = out['contract_id'].isna() | (out['contract_id'] <= 0)
            if invalid_contract_ids.any():
                raise DataValidationException(
                    f"Found {invalid_contract_ids.sum()} invalid contract IDs",
                    field="contract_id"
                )
            out['contract_id'] = out['contract_id'].astype('int64')
        
        # Convert numeric columns
        numeric_columns = [
            'multiplier', 'min_tick', 'price_magnifier', 'under_contract_id',
            'market_data_size_multiplier', 'agg_group'
        ]
        
        for col in numeric_columns:
            if col in out.columns:
                if col in ['multiplier', 'price_magnifier', 'under_contract_id', 
                          'market_data_size_multiplier', 'agg_group']:
                    # Integer columns
                    out[col] = pd.to_numeric(out[col], errors='coerce').astype('Int64')
                else:
                    # Float columns
                    out[col] = pd.to_numeric(out[col], errors='coerce')
        
        # Add created_date
        out['created_date'] = pd.Timestamp.now().date()
        
        # Ensure all schema columns exist
        schema_columns = [field.name for field in self._schema]
        out = self._ensure_schema_columns(out, schema_columns)
        
        # Validate contract data
        self._validate_contract_data(out)
        
        return out
    
    def _validate_contract_data(self, df: pd.DataFrame) -> None:
        """Validate contract description data quality."""
        
        # Check for duplicate contract IDs
        if 'contract_id' in df.columns:
            duplicate_contracts = df.duplicated(subset=['contract_id'])
            if duplicate_contracts.any():
                self.logger.warning(
                    f"Found {duplicate_contracts.sum()} duplicate contract IDs"
                )
        
        # Check for valid security types
        if 'sec_type' in df.columns:
            valid_sec_types = {'STK', 'OPT', 'FUT', 'CASH', 'BOND', 'CFD', 'FUND', 'CMDTY', 'IND', 'BILL'}
            invalid_sec_types = ~df['sec_type'].isin(valid_sec_types)
            if invalid_sec_types.any():
                unknown_types = df['sec_type'][invalid_sec_types].unique()
                self.logger.warning(
                    f"Found unknown security types: {unknown_types}. "
                    f"Valid types: {valid_sec_types}"
                )
        
        # Check for valid currencies (common ones)
        if 'currency' in df.columns and not df['currency'].isna().all():
            valid_currencies = {
                'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'HKD', 
                'KRW', 'SEK', 'NOK', 'DKK', 'SGD', 'CNH', 'MXN', 'INR'
            }
            invalid_currencies = ~df['currency'].isin(valid_currencies) & ~df['currency'].isna()
            if invalid_currencies.any():
                unknown_currencies = df['currency'][invalid_currencies].unique()
                self.logger.info(f"Found uncommon currencies: {unknown_currencies}")
    
    def save_contract_descriptions(self, df: pd.DataFrame) -> None:
        """
        Convenience method for saving contract descriptions.
        
        Args:
            df: DataFrame with contract description data
        """
        self.save(df)
    
    def load_contracts_by_symbol(self, symbol: str, 
                                sec_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load contract descriptions for specific symbol.
        
        Args:
            symbol: Contract symbol
            sec_type: Security type filter (optional)
            
        Returns:
            DataFrame with contract descriptions
        """
        filters = {'symbol': symbol}
        if sec_type:
            filters['sec_type'] = sec_type
        
        df = self.load(**filters)
        return df.sort_values(['contract_id']) if not df.empty else df
    
    def load_contract_by_id(self, contract_id: int) -> pd.DataFrame:
        """
        Load contract description by contract ID.
        
        Args:
            contract_id: IB contract ID
            
        Returns:
            DataFrame with single contract description
        """
        df = self.load()
        
        if df.empty:
            return df
        
        # Filter by contract_id
        result = df[df['contract_id'] == contract_id]
        return result
    
    def get_symbols_by_type(self, sec_type: str) -> List[str]:
        """
        Get list of symbols for specific security type.
        
        Args:
            sec_type: Security type
            
        Returns:
            List of symbols
        """
        df = self.load(sec_type=sec_type)
        
        if df.empty:
            return []
        
        symbols = df['symbol'].unique()
        return sorted([sym for sym in symbols if pd.notna(sym)])
    
    def search_contracts(self, symbol_pattern: str = None, 
                        exchange: str = None, currency: str = None,
                        sec_type: str = None) -> pd.DataFrame:
        """
        Search contracts with flexible criteria.
        
        Args:
            symbol_pattern: Symbol pattern to match (case-insensitive)
            exchange: Exchange filter
            currency: Currency filter
            sec_type: Security type filter
            
        Returns:
            DataFrame with matching contracts
        """
        df = self.load()
        
        if df.empty:
            return df
        
        # Apply filters
        if symbol_pattern:
            mask = df['symbol'].str.contains(symbol_pattern, case=False, na=False)
            df = df[mask]
        
        if exchange:
            df = df[df['exchange'] == exchange]
        
        if currency:
            df = df[df['currency'] == currency]
        
        if sec_type:
            df = df[df['sec_type'] == sec_type]
        
        return df.sort_values(['symbol', 'sec_type', 'contract_id']) if not df.empty else df
    
    def get_exchange_summary(self) -> pd.DataFrame:
        """
        Get summary of contracts by exchange.
        
        Returns:
            DataFrame with exchange statistics
        """
        df = self.load()
        
        if df.empty:
            return pd.DataFrame(columns=['exchange', 'contract_count', 'symbol_count'])
        
        summary = df.groupby('exchange').agg({
            'contract_id': 'count',
            'symbol': 'nunique'
        }).rename(columns={
            'contract_id': 'contract_count',
            'symbol': 'symbol_count'
        }).reset_index()
        
        return summary.sort_values('contract_count', ascending=False)