"""
Base repository class for trading data persistence with PyArrow/Parquet backend.
Provides common functionality for data normalization, schema validation, and Parquet operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import re

from functools import reduce
import operator
from datetime import date, datetime

from jedi.inference.value.iterable import Sequence

from common.utils.repositories import (
    list_contains_mask,
    list_contains_any_mask,
    list_contains_all_mask,
)

Filter = Tuple[str, str, Any]

def _ensure_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (str, bytes)):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]

def _build_arrow_expression(filters: Sequence[Filter]) -> Optional[ds.Expression]:
    if not filters:
        return None

    F = ds.field
    exprs = []

    for field, op, value in filters:
        op = op.lower().strip()

        if op == "==":
            e = (F(field) == value)
        elif op == "!=":
            e = (F(field) != value)
        elif op == ">=":
            e = (F(field) >= value)
        elif op == ">":
            e = (F(field) > value)
        elif op == "<=":
            e = (F(field) <= value)
        elif op == "<":
            e = (F(field) < value)
        elif op == "in":
            vals = _ensure_list(value)
            e = F(field).isin(vals)  # empty list => always False (fine)
        elif op == "not in":
            vals = _ensure_list(value)
            if not vals:
                continue  # empty NOT IN => no-op (always True)
            e = ~F(field).isin(vals)
        elif op == "between":
            lo, hi = value
            parts = []
            if lo is not None:
                parts.append(F(field) >= lo)
            if hi is not None:
                parts.append(F(field) <= hi)
            if not parts:
                continue
            e = reduce(lambda a, b: a & b, parts)
        else:
            raise ValueError(f"Unsupported op: {op}")

        exprs.append(e)

    if not exprs:
        return None
    return reduce(lambda a, b: a & b, exprs)


# Lightweight instrumentation shim (no-op in tests when LIGHTWEIGHT_REPOS=1)
from common.instrumentation import (
    DataValidationException,
    StorageException,
    get_trading_logger,
    log_context,
    performance_log,
)


from common.instrumentation import OperationType, track_repository_operation

LINEAGE_AVAILABLE = True


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x]
    return [str(x)]

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

    @property
    def base(self) -> Path:
        return self.base_path

    @property
    def schema(self) -> pa.Schema:
        return self._schema

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
        """
        Initialize the dataset folder without writing empty Parquet files.

        Writing 0-row Parquet files at the dataset root can confuse partition
        discovery and lead to read-time errors. We only ensure the directory
        exists and let the first real save() create partitioned files.
        """
        try:
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            # Optional: create a sentinel file for humans/tools
            # (kept disabled to avoid extra filesystem noise)
            # (self.dataset_path / ".dataset_initialized").touch(exist_ok=True)
            self.logger.debug(f"Initialized dataset directory at {self.dataset_path}")
        except Exception as e:
            raise StorageException(
                f"Failed to initialize dataset directory: {e}",
                storage_type="parquet",
                context={"dataset_path": str(self.dataset_path)},
            )

    # def _build_filter_expression(self, **kwargs) -> Optional[ds.Expression]:
    #     """
    #     Build PyArrow dataset filter expression from keyword arguments.
    #
    #     Handles date normalization automatically for *_date columns.
    #     """
    #     expr = None
    #
    #     for key, value in kwargs.items():
    #         if value is None:
    #             continue
    #
    #         # Normalize date-like columns
    #         if key.endswith("_date") or key == "date":
    #             value = self._normalize_date_value(value)
    #
    #         term = ds.field(key) == value
    #         expr = term if expr is None else (expr & term)
    #
    #     return expr

    # def _build_filter_expression(self, **kwargs):
    #     exprs = []
    #
    #     # symbol | symbols -> string or list
    #     symbols = kwargs.get("symbol")
    #     if symbols is None:
    #         symbols = kwargs.get("symbols")
    #     if symbols is not None:
    #         sym_list = _as_list(symbols)
    #         if len(sym_list) == 1:
    #             exprs.append(ds.field("symbol") == sym_list[0])
    #         elif len(sym_list) > 1:
    #             exprs.append(ds.field("symbol").isin(sym_list))
    #
    #     # Optional: exact date or range (adjust field names to your schema)
    #     # Accepts Python date/datetime; Arrow handles the coercion.
    #     if "date" in kwargs and kwargs["date"] is not None:
    #         d = kwargs["date"]
    #         if isinstance(d, (list, tuple, set)):
    #             exprs.append(ds.field("date").isin(list(d)))
    #         else:
    #             exprs.append(ds.field("date") == d)
    #
    #     start = kwargs.get("start") or kwargs.get("start_date")
    #     end = kwargs.get("end") or kwargs.get("end_date")
    #     if start is not None:
    #         exprs.append(ds.field("date") >= start)
    #     if end is not None:
    #         exprs.append(ds.field("date") <= end)
    #
    #     # Combine all with AND; return None if no filters (read all)
    #     if not exprs:
    #         return None
    #     return reduce(operator.and_, exprs)

    def _normalize_date_value(
        self, value: Union[str, date, datetime, pd.Timestamp]
    ) -> str:
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
                if re.match(r"^\d{2}:\d{2}:\d{2}$", parts[1]):
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
            "barcount": "bar_count",
            "what_to_show": "data_type",
            "whattosshow": "data_type",
        }

        for std_name, actual_name in rename_map.items():
            if std_name in lower_map:
                columns[lower_map[std_name]] = actual_name

        return df.rename(columns=columns, errors="ignore")

    def _ensure_schema_columns(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> pd.DataFrame:
        """Ensure all required columns exist, adding with NA values if missing."""
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def _validate_numeric_columns(
        self, df: pd.DataFrame, numeric_columns: List[str]
    ) -> pd.DataFrame:
        """Convert and validate numeric columns."""
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @performance_log("repository_save")
    @track_repository_operation(OperationType.WRITE)
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
            record_count=len(df),
        ):
            try:
                # Safety guard: block accidental production writes from standalone scripts
                try:
                    from common.utils.safety import guard_production_write

                    guard_production_write(self.base_path)
                except ImportError:
                    # Safety helper not available; ignore
                    pass

                # Normalize and validate data
                normalized_df = self._normalize_data(df.copy(), **kwargs)

                # Validate against schema
                self._validate_dataframe_schema(normalized_df)

                # Create PyArrow table
                table = pa.Table.from_pandas(
                    normalized_df, schema=self._schema, preserve_index=False
                )

                # Determine partition columns
                partition_cols = self._get_partition_columns(**kwargs)

                # Write to dataset. Default to insert-only semantics at the file level
                # to avoid wiping unrelated partitions. Repositories that need merge
                # semantics should implement custom save methods.
                pq.write_to_dataset(
                    table,
                    root_path=str(self.dataset_path),
                    partition_cols=partition_cols,
                    use_dictionary=True,
                    compression="snappy",
                    existing_data_behavior="overwrite_or_ignore",
                )

                self.logger.info(
                    f"Saved {len(normalized_df)} records to {self.dataset_name}",
                    extra={
                        "record_count": len(normalized_df),
                        "partition_cols": partition_cols,
                        "table_size_bytes": table.nbytes,
                    },
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to save data to {self.dataset_name}: {e}",
                    extra={"error_type": type(e).__name__},
                )
                if isinstance(e, (DataValidationException, StorageException)):
                    raise
                else:
                    raise StorageException(
                        f"Failed to save data: {e}",
                        storage_type="parquet",
                        context={
                            "dataset": self.dataset_name,
                            "record_count": len(df) if df is not None else 0,
                        },
                    )

    @performance_log("repository_load")
    @track_repository_operation(OperationType.READ)
    def load(self,
             **kwargs
             # filters: Optional[Sequence[Filter]]=None,
             # columns: Optional[Iterable[str]] = None,
    ) -> pa.Table:
        """
        Load data from Parquet dataset with filtering.

        Args:
            filters: Filters to apply to the query
            columns: Columns to include in the result table

        Returns:
            Filtered Arrow table
        """
        with log_context(
            operation="repository_load", repository=self.dataset_name, filters=kwargs
        ):
            filters = kwargs.pop("filters", None)
            columns = kwargs.pop("columns", None)
            try:
                # Create dataset
                dataset = ds.dataset(
                    str(self.dataset_path),
                    format="parquet",
                    partitioning="hive",
                    schema=self._schema,
                )

                # Split filters into pushdown vs post filters (e.g. list contains)
                push_filters: List[Filter] = []
                post_filters: List[Filter] = []
                if filters:
                    for f in filters:
                        try:
                            field, op, value = f
                        except Exception:
                            # Ignore malformed filters; let Arrow raise later if needed
                            continue
                        op_norm = str(op).lower().strip()
                        if op_norm in {"==", "!=", ">=", ">", "<=", "<", "in", "not in", "between"}:
                            push_filters.append((field, op, value))
                        elif op_norm in {"contains", "contains_any", "contains_all"}:
                            post_filters.append((field, op_norm, value))
                        else:
                            # Unknown op → treat as post filter to avoid hard failure
                            post_filters.append((field, op_norm, value))

                # Build filter expression: explicit pushdown filters + simple equality kwargs
                expr = _build_arrow_expression(push_filters or [])

                # Support equality kwargs like symbol="AAPL", trade_date=...
                if kwargs:
                    kw_expr = None
                    F = ds.field
                    for key, value in kwargs.items():
                        if value is None:
                            continue
                        # Normalize date-like keys
                        if key.endswith("_date") or key == "date":
                            if isinstance(value, str):
                                try:
                                    value = pd.to_datetime(value).date()
                                except Exception:
                                    pass
                            elif hasattr(value, "date") and not isinstance(value, date):
                                try:
                                    value = value.date()
                                except Exception:
                                    pass
                        term = (F(key) == value)
                        kw_expr = term if kw_expr is None else (kw_expr & term)
                    if kw_expr is not None:
                        expr = kw_expr if expr is None else (expr & kw_expr)

                table = dataset.to_table(columns=columns, filter=expr)

                # Apply post-filters that Arrow can't push down (e.g., list contains)
                if post_filters:
                    import pyarrow.compute as pc
                    for field, op_norm, value in post_filters:
                        # Skip if column not present
                        if field not in table.column_names:
                            continue
                        col = table.column(field)
                        if op_norm == "contains":
                            mask = list_contains_mask(col, value)
                        elif op_norm == "contains_any":
                            vals = _ensure_list(value)
                            mask = None
                            for v in vals:
                                m = list_contains_mask(col, v)
                                mask = m if mask is None else pc.or_(mask, m)
                            if mask is None:
                                # No values → keep all
                                continue
                        elif op_norm == "contains_all":
                            vals = _ensure_list(value)
                            mask = None
                            for v in vals:
                                m = list_contains_mask(col, v)
                                mask = m if mask is None else pc.and_(mask, m)
                            if mask is None:
                                continue
                        else:
                            # Unknown op: skip
                            continue
                        table = pc.filter(table, mask)

                self.logger.debug(
                    f"Loaded {int(table.num_rows)} records from {self.dataset_name}",
                    extra={"record_count": int(table.num_rows), "filter_params": kwargs},
                )

                return table

            except Exception as e:
                self.logger.error(
                    f"Failed to load data from {self.dataset_name}: {e}",
                    extra={"error_type": type(e).__name__, "filters": kwargs},
                )
                raise StorageException(
                    f"Failed to load data: {e}",
                    storage_type="parquet",
                    context={"dataset": self.dataset_name, "filters": kwargs},
                )

    def _validate_dataframe_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame against repository schema."""
        schema_columns = {field.name for field in self._schema}
        df_columns = set(df.columns)

        # Check for required columns (non-nullable fields)
        required_columns = {
            field.name
            for field in self._schema
            if not field.nullable and field.name in df_columns
        }

        missing_required = required_columns - df_columns
        if missing_required:
            raise DataValidationException(
                f"Missing required columns: {missing_required}",
                context={
                    "missing_columns": list(missing_required),
                    "available_columns": list(df_columns),
                    "repository": self.dataset_name,
                },
            )

        # Check for unexpected columns
        extra_columns = df_columns - schema_columns
        if extra_columns:
            self.logger.warning(
                f"DataFrame contains extra columns not in schema: {extra_columns}",
                extra={"extra_columns": list(extra_columns)},
            )

    def present_dates(
        self,
        start_date: Union[date, str, pd.Timestamp],
        end_date: Union[date, str, pd.Timestamp],
        **filter_kwargs,
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
            table = self.load(**filter_kwargs)

            if table.num_rows == 0:
                return set()

            # Find date column (try common names)
            date_col = None
            for col_name in ["trade_date", "date", "snapshot_date", "created_at"]:
                if col_name in table.column_names:
                    date_col = col_name
                    break

            if date_col is None:
                self.logger.warning("No date column found for present_dates query")
                return set()

            # Filter by date range and return unique dates
            date_series = pd.to_datetime(table.select([date_col]).to_pandas()[date_col]).dt.date
            mask = (date_series >= pd.to_datetime(start_date).date()) & (
                date_series <= pd.to_datetime(end_date).date()
            )

            return set(date_series[mask])

        except Exception as e:
            self.logger.error(f"Failed to get present dates: {e}")
            return set()

    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        try:
            dataset = ds.dataset(
                str(self.dataset_path), format="parquet", partitioning="hive"
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
                "dataset_name": self.dataset_name,
                "total_rows": total_rows,
                "total_size_bytes": total_size,
                "file_count": file_count,
                "dataset_path": str(self.dataset_path),
                "schema_columns": len(self._schema),
            }

        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {e}")
            return {"dataset_name": self.dataset_name, "error": str(e)}
