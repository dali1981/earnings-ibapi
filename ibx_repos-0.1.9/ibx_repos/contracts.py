from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Union
import pandas as pd, pyarrow as pa
from ._util import write_dataset, read_dataset, _ensure_dir, _write_empty_dataset


class ContractRepository:
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        _ensure_dir(self.base_path)
        if not any(self.base_path.rglob("*.parquet")):
            _write_empty_dataset(self.base_path, ContractRepository._contracts_schema())

    @staticmethod
    def _contracts_schema()-> pa.schema:
        return pa.schema([
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
        ])

    def _with_partitions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prefix = df["symbol"].astype(str).str[0].str.upper().fillna("")
        prefix = prefix.where(prefix.str.match(r"[A-Z]"), "#")
        df["sym_prefix"] = prefix; return df

    def save(self, df: pd.DataFrame) -> None:
        if df is None or df.empty: return
        df = df.copy()
        req = {"conid","symbol","sec_type","currency"}
        missing = req - set(df.columns)
        if missing: raise ValueError(f"ContractRepository.save missing columns: {sorted(missing)}")
        df = df.drop_duplicates(subset=["conid"]); df = self._with_partitions(df)
        write_dataset(df, self.base_path, self.schema, partition_cols=["sec_type","is_us_listing","primary_exchange","sym_prefix"])

    def load(self, sec_type: Optional[str]=None, primary_exchange: Optional[str]=None) -> pd.DataFrame:
        filters = {}
        if sec_type: filters["sec_type"] = sec_type
        if primary_exchange: filters["primary_exchange"] = primary_exchange
        return read_dataset(self.base_path, **filters)

    def present_conids(self) -> Set[int]:
        df = read_dataset(self.base_path, columns=["conid"], schema=ContractRepository._contracts_schema())
        return set(df["conid"].astype(int)) if not df.empty else set()
