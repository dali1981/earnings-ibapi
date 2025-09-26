"""
Equity bars repository with BaseRepository integration.
Handles storage and retrieval of equity historical bar data.
"""

from __future__ import annotations

from .base import BaseRepository
from datetime import date
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common.exceptions import DataValidationException
from common.observability import monitor_repository_operation


class EquityBarRepository(BaseRepository):
    """
    Repository for equity bar data following a split dataset layout:

    - Root dataset: `<base>/equity_bars/`
    - Intraday dataset under: `<base>/equity_bars/intraday/`
      Partition columns: ["bar_size", "date"]
    - Daily dataset under: `<base>/equity_bars/daily/`
      Partition columns: ["year", "month"]

    Notes:
    - Keep a single schema/class for both, but store to separate datasets.
    - Do NOT partition by symbol. Keep `symbol` as a regular column.
    - Timestamps are stored in UTC; `date` is derived from `time` for pruning.

    Expected DataFrame columns (normalized):
    - time: timestamp (UTC)
    - date: date32 (derived from time)
    - symbol: equity symbol
    - bar_size: bar size (e.g., "1 min", "5 mins", "1 day")
    - open, high, low, close, volume; optional wap, bar_count, data_type
    """

    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path, "equity_bars")
        # Ensure sub-datasets exist for daily and intraday
        self.intraday_path = self.dataset_path / "intraday"
        self.daily_path = self.dataset_path / "daily"
        self.intraday_path.mkdir(parents=True, exist_ok=True)
        self.daily_path.mkdir(parents=True, exist_ok=True)

    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for equity bars."""
        # Include partition helper fields `year`, `month` so we can materialize
        # them when writing daily bars while still validating against schema.
        return pa.schema(
            [
                pa.field("time", pa.timestamp("ns")),
                pa.field("date", pa.date32()),
                pa.field("symbol", pa.string()),
                pa.field("exchange", pa.string(), nullable=True),
                pa.field("conid", pa.int32()),
                pa.field("bar_size", pa.string()),
                pa.field("open", pa.float64()),
                pa.field("high", pa.float64()),
                pa.field("low", pa.float64()),
                pa.field("close", pa.float64()),
                pa.field("volume", pa.float64()),
                pa.field("wap", pa.float64(), nullable=True),
                pa.field("bar_count", pa.int64(), nullable=True),
                pa.field("data_type", pa.string()),
                pa.field("created_at", pa.timestamp("ns")),
                # helper partition fields for daily dataset (optional on read)
                pa.field("year", pa.int16(), nullable=True),
                pa.field("month", pa.int8(), nullable=True),
            ]
        )

    def _get_partition_columns(
        self, symbol: str = None, bar_size: str = None, **kwargs
    ) -> List[str]:
        """Backward-compat method (unused by overridden save)."""
        # We override save() to choose the appropriate dataset and partitioning.
        # Keep a harmless default here to avoid misuse elsewhere.
        return ["bar_size", "date"]

    def _normalize_data(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        bar_size: str = None,
        data_type: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
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
            out["symbol"] = symbol
        elif "symbol" not in out.columns:
            raise DataValidationException("Must provide 'symbol' parameter or column")

        if bar_size is not None:
            out["bar_size"] = bar_size
        elif "bar_size" not in out.columns:
            raise DataValidationException("Must provide 'bar_size' parameter or column")

        if data_type:
            out["data_type"] = data_type

        # Handle datetime columns
        if "datetime" in out.columns:
            # IB API typically returns 'datetime' column
            out["time"] = self._parse_ib_datetime_series(out["datetime"])
        elif "time" in out.columns:
            # Check if time is already datetime or needs parsing
            if pd.api.types.is_datetime64_any_dtype(out["time"]):
                # Already datetime, keep as is
                pass
            else:
                # Parse as IB datetime string
                out["time"] = self._parse_ib_datetime_series(out["time"])
        elif "date" in out.columns:
            # Daily bars might only have date
            out["time"] = pd.to_datetime(out["date"])
        else:
            raise DataValidationException(
                "DataFrame must contain 'datetime', 'time', or 'date' column",
                field="time",
            )

        # Derive date columns
        out["date"] = out["time"].dt.date
        # Stable ordering for deterministic files
        try:
            out = out.sort_values(["symbol", "time"]).reset_index(drop=True)
        except Exception:
            pass

        # Validate and convert OHLCV columns
        required_numeric = ["open", "high", "low", "close", "volume"]
        missing_ohlcv = [col for col in required_numeric if col not in out.columns]
        if missing_ohlcv:
            raise DataValidationException(
                f"Missing required OHLCV columns: {missing_ohlcv}", field="ohlcv"
            )

        out = self._validate_numeric_columns(out, required_numeric + ["wap"])

        # Handle bar_count as integer
        if "bar_count" in out.columns:
            out["bar_count"] = pd.to_numeric(out["bar_count"], errors="coerce").astype(
                "Int64"
            )

        # Ensure all schema columns exist
        # Add partition helper cols for daily dataset; harmless for intraday
        if "date" in out.columns:
            out["year"] = pd.to_datetime(out["date"]).dt.year.astype("Int16")
            out["month"] = pd.to_datetime(out["date"]).dt.month.astype("Int8")

        schema_columns = [field.name for field in self._schema]
        out = self._ensure_schema_columns(out, schema_columns)

        # Validate data quality
        self._validate_bar_data(out)

        return out

    def _validate_bar_data(self, df: pd.DataFrame) -> None:
        """Validate bar data quality."""
        # Check for negative volumes
        if "volume" in df.columns:
            negative_volume = df["volume"] < 0
            if negative_volume.any():
                raise DataValidationException(
                    f"Found {negative_volume.sum()} bars with negative volume",
                    field="volume",
                )

        # Check for invalid OHLC relationships
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # High should be >= Open, Low, Close
            invalid_high = df["high"] < df[["open", "low", "close"]].max(axis=1)
            if invalid_high.any():
                self.logger.warning(
                    f"Found {invalid_high.sum()} bars where high < max(open, low, close)"
                )

            # Low should be <= Open, High, Close
            invalid_low = df["low"] > df[["open", "high", "close"]].min(axis=1)
            if invalid_low.any():
                self.logger.warning(
                    f"Found {invalid_low.sum()} bars where low > min(open, high, close)"
                )

    def save_daily_bars(
        self, df: pd.DataFrame, symbol: str, data_type: str = "TRADES"
    ) -> None:
        """
        Convenience method for saving daily bars.

        Args:
            df: DataFrame with daily bar data
            symbol: Equity symbol
            data_type: Data type (default: "TRADES")
        """
        self.save(df, symbol=symbol, bar_size="1 day", data_type=data_type)

    def save_intraday_bars(
        self, df: pd.DataFrame, symbol: str, bar_size: str, data_type: str = "TRADES"
    ) -> None:
        """
        Convenience method for saving intraday bars.

        Args:
            df: DataFrame with intraday bar data
            symbol: Equity symbol
            bar_size: Bar size (e.g., "1 min", "5 mins")
            data_type: Data type (default: "TRADES")
        """
        self.save(df, symbol=symbol, bar_size=bar_size, data_type=data_type)

    # Override save to route to the correct sub-dataset with new partitions
    @monitor_repository_operation(
        "equity_bars_save",
        extract_symbol=lambda args, kwargs: kwargs.get('symbol'),
        extract_metadata=lambda args, kwargs: {
            'bar_size': kwargs.get('bar_size'),
            'data_type': kwargs.get('data_type', 'TRADES'),
            'records_count': len(args[1]) if len(args) > 1 and hasattr(args[1], '__len__') else 0
        }
    )
    def save(self, df: pd.DataFrame, **kwargs) -> None:  # type: ignore[override]
        if df is None or df.empty:
            self.logger.warning("Attempting to save empty DataFrame")
            return

        symbol = kwargs.get("symbol")
        bar_size = kwargs.get("bar_size")

        normalized_df = self._normalize_data(df.copy(), **kwargs)

        # Decide dataset by frequency
        bs = str(bar_size or normalized_df.get("bar_size").iloc[0]).strip().lower()
        is_daily = bs in ("1 day", "1d")

        if is_daily:
            # Daily: partition by year/month; avoid symbol partitioning
            table = pa.Table.from_pandas(normalized_df, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=str(self.daily_path),
                partition_cols=["year", "month"],
                use_dictionary=True,
                compression="snappy",
                existing_data_behavior="overwrite_or_ignore",
            )
        else:
            # Intraday: partition by bar_size then date
            table = pa.Table.from_pandas(normalized_df, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=str(self.intraday_path),
                partition_cols=["bar_size", "date"],
                use_dictionary=True,
                compression="snappy",
                existing_data_behavior="overwrite_or_ignore",
            )

    # Convenience loader for a date range; prunes by partitions
    def load_range(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        bar_size: str = "1 day",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        import pyarrow.dataset as ds

        bs = str(bar_size).strip().lower()
        is_daily = bs in ("1 day", "1d")

        root = self.daily_path if is_daily else self.intraday_path

        dataset = ds.dataset(str(root), format="parquet", partitioning="hive")

        F = ds.field
        expr = F("symbol") == symbol
        # Date filter (on `date` column) for both datasets
        expr = (
            expr
            & (F("date") >= pd.to_datetime(start_date).date())
            & (F("date") <= pd.to_datetime(end_date).date())
        )

        # Also prune by bar_size in intraday dataset
        if not is_daily:
            expr = expr & (F("bar_size") == bar_size)

        table = dataset.to_table(columns=columns, filter=expr)
        return table.to_pandas()

    def load_symbol_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        bar_size: Optional[str] = None,
    ) -> pd.DataFrame:
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
        filters = {"symbol": symbol}
        if bar_size:
            filters["bar_size"] = bar_size

        df = self.load(**filters)

        # Apply date filtering if specified
        if not df.empty and (start_date or end_date):
            if start_date:
                df = df[df["trade_date"] >= start_date]
            if end_date:
                df = df[df["trade_date"] <= end_date]

        return df.sort_values("time") if not df.empty else df

    def present_dates(
        self, symbol: str, bar_size: str, start_date: date, end_date: date
    ) -> set:
        """Return set of dates already stored for symbol and bar_size in range."""
        import pyarrow.dataset as ds

        bs = str(bar_size).strip().lower()
        is_daily = bs in ("1 day", "1d")
        root = self.daily_path if is_daily else self.intraday_path

        dataset = ds.dataset(str(root), format="parquet", partitioning="hive")
        F = ds.field
        expr = F("symbol") == symbol
        expr = (
            expr
            & (F("date") >= pd.to_datetime(start_date).date())
            & (F("date") <= pd.to_datetime(end_date).date())
        )
        if not is_daily:
            expr = expr & (F("bar_size") == bar_size)

        table = dataset.to_table(columns=["date"], filter=expr)
        if table.num_rows == 0:
            return set()
        df = table.to_pandas()
        return set(pd.to_datetime(df["date"]).dt.date)
