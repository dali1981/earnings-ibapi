
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Set
from datetime import date
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from ._util import _write_empty_dataset

import pyarrow.dataset as ds
import pandas as pd
from datetime import date, datetime

def _dataset_hive(path: str):
    return ds.dataset(path, format="parquet", partitioning="hive")

def _build_filter(**kwargs):
    """
    Build a pyarrow.dataset.Expression from equality filters.
    NOTE: Hive partitions (e.g., trade_date, snapshot_date) are strings "YYYY-MM-DD".
    We coerce *_date values to ISO strings.
    """
    expr = None
    for key, value in kwargs.items():
        if value is None:
            continue
        if key.endswith("date"):
            if isinstance(value, (pd.Timestamp, datetime)):
                value = value.date().isoformat()
            elif isinstance(value, date):
                value = value.isoformat()
            else:
                # best-effort parse to date then isoformat
                try:
                    value = pd.to_datetime(value).date().isoformat()
                except Exception:
                    value = str(value)
        term = ds.field(key) == value
        expr = term if expr is None else (expr & term)
    return expr


import pyarrow.dataset as ds

import pyarrow.dataset as ds
import pandas as pd
from datetime import date, datetime

def _dataset_hive(path: str):
    return ds.dataset(path, format="parquet", partitioning="hive")

def _build_filter(**kwargs):
    """
    Build a pyarrow.dataset.Expression from equality filters.
    NOTE: Hive partitions (e.g., trade_date, snapshot_date) are strings "YYYY-MM-DD".
    We coerce *_date values to ISO strings.
    """
    expr = None
    for key, value in kwargs.items():
        if value is None:
            continue
        if key.endswith("date"):
            if isinstance(value, (pd.Timestamp, datetime)):
                value = value.date().isoformat()
            elif isinstance(value, date):
                value = value.isoformat()
            else:
                # best-effort parse to date then isoformat
                try:
                    value = pd.to_datetime(value).date().isoformat()
                except Exception:
                    value = str(value)
        term = ds.field(key) == value
        expr = term if expr is None else (expr & term)
    return expr

import pandas as pd
from datetime import date

def _build_filter(**kwargs):
    """
    Build a pyarrow.dataset filter expression from keyword equality tests.
    Values of type str/float/int are used as-is; for dates, accept datetime.date or str (ISO) -> date.
    Returns None if no filters.
    """
    expr = None
    for key, value in kwargs.items():
        if value is None:
            continue
        # normalize date-like
        if key.endswith("date"):
            if isinstance(value, str):
                try:
                    value = pd.to_datetime(value).date()
                except Exception:
                    pass
            if hasattr(value, "isoformat"):
                # date or datetime
                value = value if isinstance(value, date) else getattr(value, "date", lambda: value)()
        term = ds.field(key) == value
        expr = term if expr is None else (expr & term)
    return expr

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq  # used by save()

class OptionChainSnapshotRepository:
    """
    Stores option chain snapshots (as returned from SecDefService.option_params)
    Partition by underlying, snapshot_date.
      base/underlying=AAPL/snapshot_date=2025-08-09/*.parquet
    Schema mirrors: exchange,trading_class,multiplier,expirations[],strikes[] plus underlying info.
    """
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema([
            pa.field('underlying', pa.string()),
            pa.field('underlying_conid', pa.int64()),
            pa.field('exchange', pa.string()),
            pa.field('trading_class', pa.string()),
            pa.field('multiplier', pa.string()),
            pa.field('expirations', pa.list_(pa.string())),
            pa.field('strikes', pa.list_(pa.float64())),
            pa.field('snapshot_date', pa.date32()),
        ])
        if not any(self.base_path.rglob("*.parquet")):
            _write_empty_dataset(self.base_path, self.schema)

    # ----- your save() is fine; keep it as-is -----
    def save(self, underlying: str, underlying_conid: int, snapshots: List[Dict[str, Any]], snapshot_date: Optional[date] = None) -> None:
        df = pd.DataFrame(snapshots)
        df['underlying'] = underlying
        df['underlying_conid'] = int(underlying_conid)
        df['snapshot_date'] = snapshot_date or date.today()
        # Normalize types
        df['trading_class'] = df['tradingClass'] if 'tradingClass' in df.columns else df.get('trading_class')
        if 'tradingClass' in df.columns: df.drop(columns=['tradingClass'], inplace=True)
        df['multiplier'] = df['multiplier'].astype(str)
        df['expirations'] = df['expirations'].apply(lambda x: list(x))
        df['strikes'] = df['strikes'].apply(lambda x: [float(s) for s in x])
        table = pa.Table.from_pandas(
            df[['underlying','underlying_conid','exchange','trading_class','multiplier','expirations','strikes','snapshot_date']],
            schema=self.schema, preserve_index=False
        )

        pq.write_to_dataset(
            table,
            root_path=str(self.base_path),
            partition_cols=['underlying','snapshot_date'],
            use_dictionary=True,
            compression='snappy'
        )

    # ----- improved dataset builder (typed Hive partitions) -----
    def _dataset(self) -> ds.Dataset:
        part_schema = pa.schema([
            pa.field("underlying", pa.string()),
            pa.field("snapshot_date", pa.date32()),
        ])
        return ds.dataset(
            str(self.base_path),
            format="parquet",
            partitioning=ds.partitioning(flavor="hive", schema=part_schema),
            schema=self.schema,
        )

    # ----- robust load() that casts scalars to column types -----
    def load(self, underlying: Optional[str] = None, snapshot_date: Optional[Union[str, date]] = None) -> pd.DataFrame:
        dset = self._dataset()
        schema = dset.schema

        expr = None

        # underlying filter (string)
        if underlying is not None and "underlying" in schema.names:
            expr_u = pc.equal(pc.field("underlying"), pa.scalar(underlying, type=schema.field("underlying").type))
            expr = expr_u if expr is None else pc.and_(expr, expr_u)

        # snapshot_date filter (date32)
        if snapshot_date is not None and "snapshot_date" in schema.names:
            if isinstance(snapshot_date, str):
                # accept "YYYY-MM-DD" or "YYYYMMDD"
                try:
                    dt = pd.to_datetime(snapshot_date, format="%Y-%m-%d", errors="raise").date()
                except Exception:
                    dt = pd.to_datetime(snapshot_date, format="%Y%m%d", errors="coerce").date()
            else:
                dt = snapshot_date
            expr_d = pc.equal(pc.field("snapshot_date"), pa.scalar(dt, type=schema.field("snapshot_date").type))
            expr = expr_d if expr is None else pc.and_(expr, expr_d)

        table = dset.to_table(filter=expr) if expr is not None else dset.to_table()
        df = table.to_pandas()
        return df

    def present_dates(self, underlying: str, start: date, end: date) -> Set[date]:
        """Return the set of dates for which snapshots already exist."""
        df = self.load(underlying=underlying, asof_start=start, asof_end=end)
        if df.empty:
            return set()
        return set(pd.to_datetime(df["asof"]).dt.date)