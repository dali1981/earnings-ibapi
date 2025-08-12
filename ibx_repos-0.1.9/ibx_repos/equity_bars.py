
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List, Iterable, Dict
from datetime import date, datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


import pandas as pd

def _rename_standard_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}
    # barCount -> bar_count (case-insensitive)
    if 'barcount' in lower:
        cols[ lower['barcount'] ] = 'bar_count'
    # WAP -> wap (case-insensitive)
    if 'wap' in lower and lower['wap'] != 'wap':
        cols[ lower['wap'] ] = 'wap'
    # ensure rename
    return df.rename(columns=cols, errors='ignore')

def _ensure_schema_columns(df: pd.DataFrame, required: list) -> pd.DataFrame:
    for name in required:
        if name not in df.columns:
            df[name] = pd.NA
    return df


import pandas as pd
import re

def _parse_ib_datetime_series(s):
    """Parse IB 'date' strings like 'YYYYMMDD' or 'YYYYMMDD HH:MM:SS [TZ]'. Returns pandas.DatetimeIndex."""
    if s is None:
        return pd.Series(pd.NaT)
    s = s.astype(str)
    # Fast path: tokenise and rebuild canonical form
    def _canon(x: str) -> str:
        # e.g., '20250807 04:00:00 US/Eastern' -> '20250807 04:00:00'
        parts = x.split()
        if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
            # parts[1] expected HH:MM:SS
            if re.match(r'^\d{2}:\d{2}:\d{2}$', parts[1]):
                return parts[0] + " " + parts[1]
            # daily
            return parts[0]
        return x
    canon = s.map(_canon)
    # Two attempts: intraday, then daily
    ts = pd.to_datetime(canon, format="%Y%m%d %H:%M:%S", errors="coerce")
    mask = ts.isna()
    if mask.any():
        ts2 = pd.to_datetime(canon[mask], format="%Y%m%d", errors="coerce")
        ts[mask] = ts2
    # Final fallback for any stubborn values
    mask2 = ts.isna()
    if mask2.any():
        ts3 = pd.to_datetime(canon[mask2], errors="coerce")
        ts[mask2] = ts3
    return ts


class EquityBarRepository:
    """
    Persist and read equity bars as a Parquet dataset.
    Partitioning rules:
      - Intraday (bar_size != "1 day"):
          partition_cols = ["symbol","trade_date"]
      - Daily (bar_size == "1 day"):
          partition_cols = ["symbol"]
    Expected columns in `df`:
      - For intraday: ['time','symbol','open','high','low','close','volume'] (time tz-naive ok)
      - For daily:    ['date','symbol','open','high','low','close','volume']
    We'll derive/normalize 'time'/'date' and add 'bar_size' as a column.
    """
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema([
            pa.field('time', pa.timestamp('ns')),
            pa.field('date', pa.date32()),
            pa.field('trade_date', pa.date32()),
            pa.field('symbol', pa.string()),
            pa.field('bar_size', pa.string()),
            pa.field('open', pa.float64()),
            pa.field('high', pa.float64()),
            pa.field('low', pa.float64()),
            pa.field('close', pa.float64()),
            pa.field('volume', pa.float64()),
            pa.field('wap', pa.float64()).with_nullable(True),
            pa.field('bar_count', pa.int64()).with_nullable(True),
            pa.field('what_to_show', pa.string()).with_nullable(True),
        ])

    def _normalize(self, df: pd.DataFrame, symbol: str, bar_size: str, what_to_show: Optional[str]) -> pd.DataFrame:
        out = df.copy()
        out = _rename_standard_cols(out)
        out['symbol'] = symbol
        out['bar_size'] = bar_size
        if what_to_show:
            out['what_to_show'] = what_to_show

        # Normalize time/date
        if 'time' not in out.columns and 'datetime' in out.columns:
            out['time'] = pd.to_datetime(out['datetime'])
        elif 'time' in out.columns:
            out['time'] = pd.to_datetime(out['time'])
        else:
            # daily bars may only have 'date'
            pass

        if 'date' not in out.columns:
            if 'time' in out.columns:
                out['date'] = out['time'].dt.date
            elif 'datetime' in out.columns:
                out['date'] = pd.to_datetime(out['datetime']).dt.date

        ts = _parse_ib_datetime_series(out['date'])
        out['time'] = ts
        out['date'] = ts.dt.date
        out['trade_date'] = ts.dt.date
        # Cast numerics
        for c in ['open','high','low','close','volume']:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors='coerce')
        if 'bar_count' in out.columns:
            out['bar_count'] = pd.to_numeric(out['bar_count'], errors='coerce').astype('Int64')
        if 'wap' in out.columns:
            out['wap'] = pd.to_numeric(out['wap'], errors='coerce')
        # Ensure all schema columns exist
        required = ['time','date','trade_date','symbol','bar_size','open','high','low','close','volume','wap','bar_count','what_to_show']
        out = _ensure_schema_columns(out, required)
        return out

    def save(self, df: pd.DataFrame, symbol: str, bar_size: str, what_to_show: Optional[str] = None) -> None:
        out = self._normalize(df, symbol, bar_size, what_to_show)
        table = pa.Table.from_pandas(out, schema=self.schema, preserve_index=False)
        if bar_size.strip().lower() == "1 day":
            parts = ['symbol']
        else:
            parts = ['symbol','trade_date']
        pq.write_to_dataset(
            table,
            root_path=str(self.base_path),
            partition_cols=parts,
            use_dictionary=True,
            compression='snappy'
        )

    def load(self, symbol: Optional[str] = None, trade_date: Optional[Union[str,date]] = None) -> pd.DataFrame:
        dataset = _dataset_hive(str(self.base_path))
        expr = _build_filter(symbol=symbol)
        table = dataset.to_table(filter=expr)
        df = table.to_pandas()
        if trade_date is not None and not df.empty:
            import pandas as pd
            try:
                td = pd.to_datetime(trade_date).date()
                df = df[df['trade_date'] == td]
            except Exception:
                pass
        return df
