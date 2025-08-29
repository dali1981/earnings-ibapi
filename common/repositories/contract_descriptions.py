"""
Contract descriptions repository with BaseRepository integration.
Handles storage and retrieval of IB contract description data.
"""

from __future__ import annotations

from .base import BaseRepository
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa


# Import with fallback
try:
    from common.reliability import DataValidationException
except ImportError:

    class DataValidationException(ValueError):
        def __init__(self, message: str, field: str = None, **kwargs):
            super().__init__(message)
            self.field = field


class ContractDescriptionsRepository(BaseRepository):
    """
    Repository for IB contract descriptions with Hive partitioning.

    Partitioning strategy:
    - All descriptions: partitioned by [created_at]

    Canonical schema columns (see _create_schema):
    - symbol: contract symbol
    - sec_type: security type (STK, OPT, FUT, etc.)
    - conid: IB contract ID (int64)
    - primary_exchange: primary exchange (nullable)
    - currency: contract currency (nullable)
    - sec_id_type: security id type (nullable)
    - sec_id: security id (nullable)
    - derivative_sec_types: list<string>
    - created_at: date32 (partition key)
    """

    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path, "contract_descriptions")

    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for contract descriptions."""
        return pa.schema(
            [
                pa.field("symbol", pa.string()),
                pa.field("sec_type", pa.string()),
                pa.field("conid", pa.int64()),
                pa.field("primary_exchange", pa.string(), nullable=True),
                pa.field("currency", pa.string(), nullable=True),
                pa.field("sec_id_type", pa.string(), nullable=True),
                pa.field("sec_id", pa.string(), nullable=True),
                pa.field("derivative_sec_types", pa.list_(pa.string())),
                pa.field("created_at", pa.timestamp('s', 'UTC'), nullable=False),
            ]
        )

    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for contract descriptions."""
        return ["created_at"]

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

        # Handle common column name variations for IB contract details → canonical schema
        column_mapping = {
            # ID
            "conId": "conid",
            "contractId": "conid",
            "contractid": "conid",
            "con_id": "conid",
            "contract_id": "conid",
            # Security type / symbol
            "sectype": "sec_type",
            "secType": "sec_type",
            # Exchanges / currency
            "primaryexchange": "primary_exchange",
            "primaryExchange": "primary_exchange",
            "exchange": "primary_exchange",
            # Sec ID
            "secIdType": "sec_id_type",
            "secidtype": "sec_id_type",
            "secId": "sec_id",
            "secid": "sec_id",
            # Derivative types
            "derivativeSecTypes": "derivative_sec_types",
            # Legacy date naming
            "as_of_date": "created_at",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in out.columns and new_name not in out.columns:
                out[new_name] = out[old_name]

        # Validate required columns consistent with schema
        required_columns = ["symbol", "sec_type", "conid"]
        missing_required = [col for col in required_columns if col not in out.columns]
        if missing_required:
            raise DataValidationException(
                f"Missing required columns: {missing_required}",
                field="required_columns",
            )

        # Normalize/validate conid
        out["conid"] = pd.to_numeric(out["conid"], errors="coerce").astype("Int64")
        invalid_conid = out["conid"].isna() | (out["conid"] <= 0)
        if invalid_conid.any():
            raise DataValidationException(
                f"Found {int(invalid_conid.sum())} invalid conid values", field="conid"
            )
        out["conid"] = out["conid"].astype("int64")

        # Convert numeric columns
        # Normalize text fields to uppercase where appropriate
        for col in ["symbol", "sec_type", "currency", "primary_exchange"]:
            if col in out.columns:
                out[col] = out[col].astype(str).str.upper()

        # Derivative sec types: ensure list[str]
        def _to_list_str(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            if isinstance(x, str):
                return [s.strip().upper() for s in x.split(",") if s.strip()]
            if isinstance(x, (list, tuple, set)):
                return [str(s).upper() for s in x]
            return [str(x).upper()]

        out["derivative_sec_types"] = (
            out["derivative_sec_types"].map(_to_list_str)
            if "derivative_sec_types" in out.columns
            else [[]] * len(out)
        )

        # created_at: ensure present and as timestamp (schema expects timestamp, not date)
        if "created_at" not in out.columns:
            out["created_at"] = pd.Timestamp.now(tz='UTC').replace(microsecond=0)
        else:
            out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
            # Fill any null values with current timestamp
            out["created_at"] = out["created_at"].fillna(pd.Timestamp.now(tz='UTC').replace(microsecond=0))

        # Ensure all schema columns exist (PyArrow requires all schema fields present)
        schema_columns = [field.name for field in self._schema]
        out = self._ensure_schema_columns(out, schema_columns)

        # Validate contract data
        self._validate_contract_data(out)

        return out

    def _validate_contract_data(self, df: pd.DataFrame) -> None:
        """Validate contract description data quality."""

        # Check for duplicate contract IDs
        if "conid" in df.columns:
            duplicate_contracts = df.duplicated(subset=["conid"])
            if duplicate_contracts.any():
                self.logger.warning(
                    f"Found {duplicate_contracts.sum()} duplicate conid values"
                )

        # Check for valid security types
        if "sec_type" in df.columns:
            valid_sec_types = {
                "STK",
                "OPT",
                "FUT",
                "CASH",
                "BOND",
                "CFD",
                "FUND",
                "CMDTY",
                "IND",
                "BILL",
            }
            invalid_sec_types = ~df["sec_type"].isin(valid_sec_types)
            if invalid_sec_types.any():
                unknown_types = df["sec_type"][invalid_sec_types].unique()
                self.logger.warning(
                    f"Found unknown security types: {unknown_types}. "
                    f"Valid types: {valid_sec_types}"
                )

        # Check for valid currencies (common ones)
        if "currency" in df.columns and not df["currency"].isna().all():
            valid_currencies = {
                "USD",
                "EUR",
                "GBP",
                "JPY",
                "CHF",
                "CAD",
                "AUD",
                "HKD",
                "KRW",
                "SEK",
                "NOK",
                "DKK",
                "SGD",
                "CNH",
                "MXN",
                "INR",
            }
            invalid_currencies = (
                ~df["currency"].isin(valid_currencies) & ~df["currency"].isna()
            )
            if invalid_currencies.any():
                unknown_currencies = df["currency"][invalid_currencies].unique()
                self.logger.info(f"Found uncommon currencies: {unknown_currencies}")

    def save_contract_descriptions(self, df: pd.DataFrame) -> None:
        """
        Convenience method for saving contract descriptions.

        Args:
            df: DataFrame with contract description data
        """
        self.save(df)

    def load_contracts_by_symbol(
        self, symbol: str, sec_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load contract descriptions for specific symbol.

        Args:
            symbol: Contract symbol
            sec_type: Security type filter (optional)

        Returns:
            DataFrame with contract descriptions
        """
        filters = {"symbol": symbol}
        if sec_type:
            filters["sec_type"] = sec_type

        df = self.load(**filters)
        return df.sort_values(["conid"]) if not df.empty else df

    def load_contract_by_id(self, conid: int) -> pd.DataFrame:
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

        # Filter by conid
        result = df[df["conid"] == conid]
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

        symbols = df["symbol"].unique()
        return sorted([sym for sym in symbols if pd.notna(sym)])

    def search_contracts(
        self,
        symbol_pattern: str = None,
        exchange: str = None,
        currency: str = None,
        sec_type: str = None,
    ) -> pd.DataFrame:
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
            mask = df["symbol"].str.contains(symbol_pattern, case=False, na=False)
            df = df[mask]

        if exchange:
            df = df[df["primary_exchange"] == exchange]

        if currency:
            df = df[df["currency"] == currency]

        if sec_type:
            df = df[df["sec_type"] == sec_type]

        return (
            df.sort_values(["symbol", "sec_type", "conid"])
            if not df.empty
            else df
        )

    def get_exchange_summary(self) -> pd.DataFrame:
        """
        Get summary of contracts by exchange.

        Returns:
            DataFrame with exchange statistics
        """
        df = self.load()

        if df.empty:
            return pd.DataFrame(columns=["primary_exchange", "contract_count", "symbol_count"])

        summary = (
            df.groupby("primary_exchange")
            .agg({"conid": "count", "symbol": "nunique"})
            .rename(columns={"conid": "contract_count", "symbol": "symbol_count"})
            .reset_index()
        )

        return summary.sort_values("contract_count", ascending=False)
