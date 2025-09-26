from __future__ import annotations

from ._util import _ensure_dir, _write_empty_dataset, read_dataset, write_dataset
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Set, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


class ContractRepository:
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base = (
            self.base_path
        )  # Alias for consistency with ContractDescriptionsRepository
        _ensure_dir(self.base_path)
        if not any(self.base_path.rglob("*.parquet")):
            _write_empty_dataset(self.base_path, ContractRepository._contracts_schema())

    @staticmethod
    def _contracts_schema() -> pa.schema:
        return pa.schema(
            [
                pa.field("conid", pa.int64()),
                pa.field("symbol", pa.string()),
                pa.field("sec_type", pa.string()),
                pa.field("currency", pa.string()),
                pa.field("primary_exchange", pa.string()).with_nullable(True),
                pa.field("exchange", pa.string()).with_nullable(True),
                pa.field("local_symbol", pa.string()).with_nullable(True),
                pa.field("trading_class", pa.string()).with_nullable(True),
                pa.field("long_name", pa.string()).with_nullable(True),
                pa.field("category", pa.string()).with_nullable(True),
                pa.field("sub_category", pa.string()).with_nullable(True),
                pa.field("industry", pa.string()).with_nullable(True),
                pa.field("time_zone", pa.string()).with_nullable(True),
                pa.field("is_us_listing", pa.bool_()).with_nullable(True),
                pa.field("sym_prefix", pa.string()).with_nullable(True),
            ]
        )

    def _with_partitions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prefix = df["symbol"].astype(str).str[0].str.upper().fillna("")
        prefix = prefix.where(prefix.str.match(r"[A-Z]"), "#")
        df["sym_prefix"] = prefix
        return df

    def save(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df = df.copy()
        req = {"conid", "symbol", "sec_type", "currency"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(
                f"ContractRepository.save missing columns: {sorted(missing)}"
            )
        df = df.drop_duplicates(subset=["conid"])
        df = self._with_partitions(df)
        write_dataset(
            df,
            self.base_path,
            self._contracts_schema(),
            partition_cols=[
                "sec_type",
                "is_us_listing",
                "primary_exchange",
                "sym_prefix",
            ],
        )

    def read(self, columns: Optional[Sequence[str]] = None, **filters) -> pd.DataFrame:
        """Read contracts with optional filters.

        Args:
            columns: Optional list of columns to return
            **filters: Keyword arguments for filtering. Common filters:
                - symbol: filter by symbol(s)
                - conid: filter by contract ID(s)
                - sec_type: filter by security type(s)
                - currency: filter by currency(s)
                - primary_exchange: filter by primary exchange(s)

        Returns:
            DataFrame with filtered contracts
        """
        if not self.base_path.exists() or not any(self.base_path.rglob("*.parquet")):
            schema = self._contracts_schema()
            cols = list(columns) if columns else [f.name for f in schema]
            return pd.DataFrame(columns=cols)

        dset = ds.dataset(
            self.base_path,
            format="parquet",
            partitioning="hive",
            schema=self._contracts_schema(),
        )
        schema = dset.schema
        from pyarrow.dataset import field

        expr = None
        for k, v in filters.items():
            if k not in schema.names:
                continue
            # Handle list/tuple/set filters with isin
            if isinstance(v, (list, tuple, set)):
                # Normalize symbol filters to uppercase
                if k == "symbol":
                    v = [str(s).upper() for s in v]
                cond = field(k).isin(list(v))
            else:
                # Single value filter
                if k == "symbol":
                    v = str(v).upper()
                cond = field(k) == v

            expr = cond if expr is None else (expr & cond)

        cols = [c for c in (columns or schema.names) if c in schema.names]
        tbl = dset.to_table(filter=expr, columns=cols)
        return tbl.to_pandas() if tbl.num_rows else pd.DataFrame(columns=cols)

    def load(
        self, sec_type: Optional[str] = None, primary_exchange: Optional[str] = None
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility. Use read() instead."""
        filters = {}
        if sec_type:
            filters["sec_type"] = sec_type
        if primary_exchange:
            filters["primary_exchange"] = primary_exchange
        return self.read(**filters)

    def present_conids(self) -> Set[int]:
        df = self.read(columns=["conid"])
        return set(df["conid"].astype(int)) if not df.empty else set()
