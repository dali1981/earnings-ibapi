"""
Option chain repository with BaseRepository integration.
Handles storage and retrieval of option chain snapshot data.
"""

from __future__ import annotations

from .base import BaseRepository
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
import pyarrow as pa


from common.exceptions import DataValidationException


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
        return pa.schema(
            [
                pa.field("underlying", pa.string()),
                pa.field("underlying_conid", pa.int64()),
                pa.field("exchange", pa.string()),
                pa.field("trading_class", pa.string()),
                pa.field("multiplier", pa.string()),
                pa.field("expirations", pa.list_(pa.string())),
                pa.field("strikes", pa.list_(pa.float64())),
                pa.field("as_of", pa.date32()),
            ]
        )

    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for option chain snapshots."""
        return ["as_of"]

    def _normalize_data(
        self,
        df: pd.DataFrame,
        underlying: str | None = None,
        as_of: date | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Normalize aggregated option chain snapshot data for storage.

        Expects one row per parameter set with list fields for expirations and strikes.
        """
        if df is None or df.empty:
            raise DataValidationException("Cannot normalize empty DataFrame")

        out = df.copy()

        # Standardize column names
        rename_map = {
            "snapshot_date": "as_of",
            "date": "as_of",
            "asof": "as_of",
            "tradingClass": "trading_class",
            "under_conid": "underlying_conid",
            "conId": "underlying_conid",
        }
        for k, v in rename_map.items():
            if k in out.columns and v not in out.columns:
                out[v] = out[k]

        # Inject required params
        if underlying is not None:
            out["underlying"] = underlying
        if as_of is not None:
            out["as_of"] = as_of

        # Validate required columns
        required = ["underlying", "expirations", "strikes", "as_of"]
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise DataValidationException(
                f"Missing required columns: {missing}", field="required_columns"
            )

        # Normalize types
        out["underlying"] = out["underlying"].astype(str).str.upper()

        # as_of to date
        out["as_of"] = pd.to_datetime(out["as_of"]).dt.date

        # List coercion helpers
        def _to_list(x) -> list:
            if isinstance(x, (list, tuple, set)):
                return list(x)
            if pd.isna(x):
                return []
            return [x]

        def _norm_expirations(seq: Iterable) -> list[str]:
            vals = []
            for e in _to_list(seq):
                if pd.isna(e):
                    continue
                s = str(e)
                # Normalize 20240920 -> 2024-09-20 if looks like yyyymmdd
                if len(s) == 8 and s.isdigit():
                    try:
                        dt = pd.to_datetime(s, format="%Y%m%d").date()
                        vals.append(dt.isoformat())
                        continue
                    except Exception:
                        pass
                try:
                    dt = pd.to_datetime(s).date()
                    vals.append(dt.isoformat())
                except Exception:
                    vals.append(s)
            # dedupe + sort
            return sorted(set(vals))

        def _norm_strikes(seq: Iterable) -> list[float]:
            vals = []
            for x in _to_list(seq):
                try:
                    vals.append(float(x))
                except Exception:
                    continue
            return sorted(set(vals))

        out["expirations"] = out["expirations"].apply(_norm_expirations)
        out["strikes"] = out["strikes"].apply(_norm_strikes)

        # Ensure optional fields exist with proper types
        if "underlying_conid" in out.columns:
            out["underlying_conid"] = pd.to_numeric(
                out["underlying_conid"], errors="coerce"
            ).astype("Int64")
        else:
            out["underlying_conid"] = pd.NA

        for col in ["exchange", "trading_class", "multiplier"]:
            if col not in out.columns:
                out[col] = pd.NA
            else:
                out[col] = out[col].astype(str)

        # Ensure schema columns presence
        out = self._ensure_schema_columns(out, [f.name for f in self._schema])

        return out

    def _validate_option_chain_data(self, df: pd.DataFrame) -> None:
        """Validate option chain snapshot data quality (list form)."""
        # Ensure expirations/strikes lists are not empty
        if "expirations" in df.columns:
            empties = df["expirations"].map(lambda x: len(x) == 0)
            if empties.any():
                self.logger.warning("Found snapshots with empty expirations list")
        if "strikes" in df.columns:
            empties = df["strikes"].map(lambda x: len(x) == 0)
            if empties.any():
                self.logger.warning("Found snapshots with empty strikes list")

    # Convenience helpers
    def get_latest_snapshot_date(self, underlying: str) -> Optional[date]:
        """Return the most recent snapshot_date for an underlying, or None."""
        try:
            tbl = self.load(underlying=underlying.upper())
            if tbl is None or tbl.num_rows == 0:
                return None
            # Compute max snapshot_date via Arrow to avoid full Pandas conversion
            import pyarrow.compute as pc

            val = pc.max(tbl.column("as_of")).as_py()
            # Arrow date32 returns datetime.date already
            return val
        except Exception:
            return None

    def save_chain_snapshot(
        self, df: pd.DataFrame, underlying: str, snapshot_date: date
    ) -> None:
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
        return df.sort_values(["expiry", "strike", "right"]) if not df.empty else df

    def load_chains_for_expiry(
        self,
        underlying: str,
        expiry: date,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
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
        df = df[df["expiry"] == expiry]

        # Apply date filtering if specified
        if start_date:
            df = df[df["snapshot_date"] >= start_date]
        if end_date:
            df = df[df["snapshot_date"] <= end_date]

        return (
            df.sort_values(["snapshot_date", "strike", "right"]) if not df.empty else df
        )

    def get_available_expiries(
        self, underlying: str, snapshot_date: Optional[date] = None
    ) -> List[date]:
        """
        Get list of available option expiries for underlying.

        Args:
            underlying: Underlying symbol
            snapshot_date: Specific snapshot date (optional)

        Returns:
            List of available expiry dates
        """
        filters = {"underlying": underlying}
        if snapshot_date:
            filters["snapshot_date"] = snapshot_date

        df = self.load(**filters)

        if df.empty:
            return []

        expiries = df["expiry"].unique()
        return sorted([exp for exp in expiries if pd.notna(exp)])

    def get_strike_range(
        self, underlying: str, expiry: date, snapshot_date: Optional[date] = None
    ) -> tuple:
        """
        Get strike price range for specific expiry.

        Args:
            underlying: Underlying symbol
            expiry: Option expiry date
            snapshot_date: Specific snapshot date (optional)

        Returns:
            Tuple of (min_strike, max_strike)
        """
        filters = {"underlying": underlying}
        if snapshot_date:
            filters["snapshot_date"] = snapshot_date

        df = self.load(**filters)

        if df.empty:
            return (None, None)

        # Filter by expiry
        df = df[df["expiry"] == expiry]

        if df.empty:
            return (None, None)

        return (df["strike"].min(), df["strike"].max())
