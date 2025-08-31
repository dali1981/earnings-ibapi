"""
Contract repository for storing basic contract information.
Separate from ContractDescriptions which stores detailed IB contract descriptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Set, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from common.repositories.base import BaseRepository


class ContractRepository(BaseRepository):
    """Repository for basic contract information."""
    
    def __init__(self, base_path: Union[str, Path]):
        # Override base_path to match what tests expect
        super().__init__(base_path, "contracts")

    def _create_schema(self) -> pa.Schema:
        """Define the schema for contracts data."""
        return pa.schema(
            [
                pa.field("symbol", pa.string()),
                pa.field("sec_type", pa.string()),
                pa.field("conid", pa.int64()),
                pa.field("exchange", pa.string(), nullable=True),
                pa.field("primary_exchange", pa.string(), nullable=True),
                pa.field("currency", pa.string(), nullable=True),
                pa.field("local_symbol", pa.string(), nullable=True),
                pa.field("trading_class", pa.string(), nullable=True),
                pa.field("multiplier", pa.int64(), nullable=True),
                pa.field("min_tick", pa.float64(), nullable=True),
                pa.field("price_magnifier", pa.int64(), nullable=True),
                pa.field("under_conid", pa.int64(), nullable=True),
                pa.field("long_name", pa.string(), nullable=True),
                pa.field("industry", pa.string(), nullable=True),
                pa.field("category", pa.string(), nullable=True),
                pa.field("subcategory", pa.string(), nullable=True),
                pa.field("timezone_id", pa.string(), nullable=True),
                pa.field("trading_hours", pa.string(), nullable=True),
                pa.field("liquid_hours", pa.string(), nullable=True),
                pa.field("market_data_size_multiplier", pa.int64(), nullable=True),
                pa.field("agg_group", pa.int64(), nullable=True),
                pa.field("under_symbol", pa.string(), nullable=True),
                pa.field("under_sec_type", pa.string(), nullable=True),
                pa.field("market_rule_ids", pa.string(), nullable=True),
                pa.field("real_expiration_date", pa.string(), nullable=True),
                pa.field("last_trade_time", pa.string(), nullable=True),
                pa.field("stock_type", pa.string(), nullable=True),
                # Partition column (derived from created_at)
                pa.field("snapshot", pa.date32(), nullable=False),
                # Timestamp (UTC, nanosecond precision to match pandas)
                pa.field("created_at", pa.timestamp('ns', 'UTC'), nullable=False),
            ]
        )
    
    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for contract data."""
        # Partition by security type and snapshot date for efficient scans
        return ["sec_type", "snapshot"]
    
    def _normalize_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Normalize contract data and add partition columns."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Derive snapshot date from created_at timestamp
        if "created_at" in df.columns:
            try:
                ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
                df["snapshot"] = ts.dt.tz_convert("UTC").dt.date
            except Exception:
                # Fallback to naive date if tz conversion fails
                df["snapshot"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date
        else:
            # If created_at missing, use today's date to avoid partition errors
            df["snapshot"] = pd.Timestamp.utcnow().date()
        
        return df
    
    # Keep backward compatibility method
    @staticmethod
    def _contracts_schema() -> pa.Schema:
        """Legacy method for schema - use _create_schema() instead."""
        return ContractRepository(Path("/tmp"))._create_schema()
    
    def _with_partitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add partition columns to dataframe."""
        df = df.copy()
        if "snapshot" not in df.columns:
            if "created_at" in df.columns:
                ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
                df["snapshot"] = ts.dt.tz_convert("UTC").dt.date
            else:
                df["snapshot"] = pd.Timestamp.utcnow().date()
        return df
    
    def save(self, df: pd.DataFrame) -> None:
        """Save contracts to repository.
        
        Args:
            df: DataFrame with contract data
        """
        if df is None or df.empty:
            return
            
        df = df.copy()
        
        # Check required columns
        req = {"conid", "symbol", "sec_type", "currency"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(
                f"ContractRepository.save missing columns: {sorted(missing)}"
            )
        
        # Deduplicate by contract ID
        df = df.drop_duplicates(subset=["conid"])
        
        # Normalize data (adds partition columns)
        df = self._normalize_data(df)
        
        # Write to dataset using write_to_dataset for append behavior
        partition_cols = self._get_partition_columns()
        import pyarrow.parquet as pq
        
        table = pa.Table.from_pandas(df, schema=self._create_schema())
        pq.write_to_dataset(
            table,
            root_path=str(self.base_path),
            partition_cols=partition_cols,
            use_dictionary=True,
            compression="snappy",
        )
    
    # def read(self, columns: Optional[Sequence[str]] = None, **filters) -> pa.Table:
    #     """Read contracts with optional filters.
    #
    #     Args:
    #         columns: Optional list of columns to return
    #         **filters: Keyword arguments for filtering. Common filters:
    #             - symbol: filter by symbol(s)
    #             - conid: filter by contract ID(s)
    #             - sec_type: filter by security type(s)
    #             - currency: filter by currency(s)
    #             - primary_exchange: filter by primary exchange(s)
    #             - is_us_listing: filter by US listing status
    #
    #     Returns:
    #         DataFrame with filtered contracts
    #     """
    #     if not self.base_path.exists() or not any(self.base_path.rglob("*.parquet")):
    #         schema = self._create_schema()
    #         cols = list(columns) if columns else [f.name for f in schema]
    #         empty_df = pd.DataFrame(columns=cols)
    #         return pa.Table.from_pandas(empty_df, schema=schema, preserve_index=False)
    #
    #     dset = ds.dataset(
    #         self.base_path,  # Use base_path instead of dataset_path
    #         format="parquet",
    #         partitioning="hive",
    #         schema=self._create_schema(),
    #     )
    #
    #     schema = dset.schema
    #     from pyarrow.dataset import field
    #
    #     expr = None
    #     for k, v in filters.items():
    #         if k not in schema.names:
    #             continue
    #
    #         # Handle list/tuple/set filters with isin
    #         if isinstance(v, (list, tuple, set)):
    #             # Normalize symbol filters to uppercase
    #             if k == "symbol":
    #                 v = [str(s).upper() for s in v]
    #             cond = field(k).isin(list(v))
    #         else:
    #             # Single value filter
    #             if k == "symbol":
    #                 v = str(v).upper()
    #             cond = field(k) == v
    #
    #         expr = cond if expr is None else (expr & cond)
    #
    #     cols = [c for c in (columns or schema.names) if c in schema.names]
    #     tbl = dset.to_table(filter=expr, columns=cols)
    #     return tbl
    
    # def load(
    #     self, sec_type: Optional[str] = None, primary_exchange: Optional[str] = None
    # ) -> pa.Table:
    #     """Legacy method for backward compatibility. Use read() instead."""
    #     filters = {}
    #     if sec_type:
    #         filters["sec_type"] = sec_type
    #     if primary_exchange:
    #         filters["primary_exchange"] = primary_exchange
    #     return self.read(**filters)
    
    def present_conids(self) -> Set[int]:
        """Get set of contract IDs present in repository."""
        tbl = self.load(columns=["conid"])
        if tbl.num_rows == 0:
            return set()
        df = tbl.to_pandas()
        return set(df["conid"].dropna().astype(int))
