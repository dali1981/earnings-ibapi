"""
Repository for detailed option contracts (one row per contract).

Partitioning strategy:
- Partition by [symbol, expiry] for efficient selection and lookups.

Schema captures key contract-identifying fields and common IB metadata.
"""

from __future__ import annotations

from .base import BaseRepository
from datetime import date
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa


class OptionContractsRepository(BaseRepository):
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path, "option_contracts")

    def _create_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("symbol", pa.string()),
                pa.field("sec_type", pa.string()),  # OPT
                pa.field("conid", pa.int64()),
                pa.field("exchange", pa.string(), nullable=True),
                pa.field("primary_exchange", pa.string(), nullable=True),
                pa.field("currency", pa.string()),
                pa.field("local_symbol", pa.string(), nullable=True),
                pa.field("trading_class", pa.string(), nullable=True),
                pa.field("multiplier", pa.int64(), nullable=True),
                # Option specifics
                pa.field("strike", pa.float64()),
                pa.field("right", pa.string()),  # C/P
                pa.field("expiry", pa.date32()),
                pa.field("expiry_raw", pa.string(), nullable=True),
                # IB/market metadata
                pa.field("min_tick", pa.float64(), nullable=True),
                pa.field("valid_exchanges", pa.string(), nullable=True),
                pa.field("long_name", pa.string(), nullable=True),
                pa.field("category", pa.string(), nullable=True),
                pa.field("subcategory", pa.string(), nullable=True),
                pa.field("industry", pa.string(), nullable=True),
                pa.field("timezone_id", pa.string(), nullable=True),
                pa.field("trading_hours", pa.string(), nullable=True),
                pa.field("liquid_hours", pa.string(), nullable=True),
                pa.field("market_rule_ids", pa.string(), nullable=True),
                pa.field("contract_month", pa.string(), nullable=True),
                # Partitions / timestamps
                pa.field("snapshot", pa.date32(), nullable=False),
                pa.field("created_at", pa.timestamp("ns", "UTC"), nullable=False),
            ]
        )

    def _get_partition_columns(self, **kwargs) -> List[str]:
        # Partition by underlying symbol and expiry for efficient scans
        return ["symbol", "expiry"]

    def _normalize_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        out = df.copy()

        # Standardize column names from mixed sources
        rename_map = {
            "sectype": "sec_type",
            "tradingClass": "trading_class",
            "localSymbol": "local_symbol",
            "conId": "conid",
            "primaryExchange": "primary_exchange",
            "timeZoneId": "timezone_id",
        }
        for k, v in rename_map.items():
            if k in out.columns and v not in out.columns:
                out[v] = out[k]

        # Ensure required fields exist
        for col in [
            "symbol",
            "sec_type",
            "conid",
            "currency",
            "strike",
            "right",
        ]:
            if col not in out.columns:
                out[col] = pd.NA

        # Set sec_type to 'OPT' if missing
        out["sec_type"] = out["sec_type"].fillna("OPT").astype(str)

        # Normalize symbol and right
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out["right"] = out["right"].astype(str).str.upper()

        # Strike as float
        out["strike"] = pd.to_numeric(out["strike"], errors="coerce")

        # Conid as int
        out["conid"] = pd.to_numeric(out["conid"], errors="coerce").astype("Int64")

        # Multiplier as int if present
        if "multiplier" in out.columns:
            out["multiplier"] = pd.to_numeric(
                out["multiplier"], errors="coerce"
            ).astype("Int64")

        # Expiry normalization
        # Accept either 'expiry' as date/string or 'lastTradeDateOrContractMonth' or 'expiry_raw'
        if "expiry" not in out.columns:
            out["expiry"] = pd.NA
        if "expiry_raw" not in out.columns:
            out["expiry_raw"] = pd.NA

        def _norm_exp(val) -> Optional[date]:
            if pd.isna(val):
                return None
            s = str(val)
            try:
                # Handle forms like "YYYYMMDD", "YYYY-MM-DD", or with time/tz suffix
                parts = s.strip().split()
                token = parts[0] if parts else s
                if len(token) == 8 and token.isdigit():
                    return pd.to_datetime(token, format="%Y%m%d").date()
                try:
                    return pd.to_datetime(token).date()
                except Exception:
                    return pd.to_datetime(s).date()
            except Exception:
                return None

        # Pull raw expiry if present under common names
        for src in ("lastTradeDateOrContractMonth", "expiry", "expiry_raw"):
            if src in out.columns and out["expiry_raw"].isna().all():
                out["expiry_raw"] = out[src]

        # Derive expiry date column from raw or local_symbol
        if out["expiry"].isna().any():
            mask = out["expiry"].isna()
            out.loc[mask, "expiry"] = out.loc[mask, "expiry_raw"].map(_norm_exp)
            # Fallback to parse from local_symbol (e.g., "BABA  250905C00135000")
            ls = out.get("local_symbol")
            if ls is not None:
                import re

                def _from_local(sym: str) -> Optional[date]:
                    if not isinstance(sym, str):
                        return None
                    m = re.search(r"(\d{6})[CP]", sym)
                    if not m:
                        return None
                    yymmdd = m.group(1)
                    try:
                        # Assume 20yy for modern options
                        return pd.to_datetime("20" + yymmdd, format="%Y%m%d").date()
                    except Exception:
                        return None

                mask2 = out["expiry"].isna()
                out.loc[mask2, "expiry"] = out.loc[mask2, "local_symbol"].map(
                    _from_local
                )

        # Timestamps: created_at and snapshot (date)
        if "created_at" not in out.columns or out["created_at"].isna().all():
            out["created_at"] = pd.Timestamp.now(tz="UTC")
        else:
            out["created_at"] = pd.to_datetime(
                out["created_at"], utc=True, errors="coerce"
            )
        out["snapshot"] = out["created_at"].dt.tz_convert("UTC").dt.date

        # Cast optional metadata to pandas StringDtype to preserve missing as <NA>
        for c in [
            "exchange",
            "primary_exchange",
            "currency",
            "local_symbol",
            "trading_class",
            "valid_exchanges",
            "long_name",
            "category",
            "subcategory",
            "industry",
            "timezone_id",
            "trading_hours",
            "liquid_hours",
            "market_rule_ids",
            "contract_month",
        ]:
            if c in out.columns:
                out[c] = out[c].astype("string")
            else:
                out[c] = pd.NA

        # Ensure schema columns presence/order
        out = self._ensure_schema_columns(out, [f.name for f in self._schema])

        return out
