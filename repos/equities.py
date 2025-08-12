from pathlib import Path
from datetime import date
from typing import List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from ibx_time import parse_ib_datetime_series



class EquityBarRepository:
    """
    Repository for storing and loading historical equity bars as Parquet.
    Directory layout: base_path/equity_bars/symbol=<symbol>/trade_date=YYYY-MM-DD/*.parquet
    Similar schema to option bars but keyed by 'symbol' instead of 'contract_id'.
    """

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path) / "equity_bars"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema([
            pa.field('time', pa.timestamp('ns')),
            pa.field('symbol', pa.string()),
            pa.field('open', pa.float64()),
            pa.field('high', pa.float64()),
            pa.field('low', pa.float64()),
            pa.field('close', pa.float64()),
            pa.field('volume', pa.int64()),
            pa.field('bar_size', pa.string()),
            pa.field('data_type', pa.string()),
            pa.field('date', pa.date32()),
        ])

    def save(self, df: pd.DataFrame) -> None:
        """
        Save equity bar DataFrame to Parquet, partitioned by symbol and trade_date.
        Expects columns: ['trade_time','symbol','open','high','low','close','volume']
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("EquityBarRepository.save requires a pandas DataFrame")
        df = df.copy()

        out = df.copy()
        # bar_size = out.attrs.get('bar_size')
        bar_size = out['bar_size'][0]

        # Daily bars
        if bar_size == '1 day':
            if 'datetime' not in out.columns:
                raise ValueError("Daily bars require 'date' column")
            # normalize
            # if pd.api.types.is_datetime64_any_dtype(out['datetime']):
            ts = parse_ib_datetime_series(out['datetime'])
            out['date'] = ts.dt.date
            # set a dummy time at midnight
            out['time'] = ts
        else:
            # Intraday bars
            if "datetime" not in out.columns:
                raise ValueError("Intraday bars require 'time' column")
            ts = parse_ib_datetime_series(out["datetime"])
            out["time"] = ts
            out["date"] = ts.dt.date

        out.rename(columns={'what_to_show': 'data_type'}, inplace=True)
        # out['symbol'] = out.attrs.get('symbol').astype(str)
        # out['bar_size'] = out.attrs.get('bar_size').astype(str)

        # choose partitions
        if bar_size == '1 day':
            parts = ['symbol']
        else:
            parts = ['symbol', 'date']

        table = pa.Table.from_pandas(out, schema=self.schema, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(self.base_path),
            partition_cols=parts,
            use_dictionary=True,
            compression='snappy'
        )

    def load(
        self,
        trade_date: Union[str, date],
        symbol: Optional[str] = None,
        bar_size: Optional[str] = None,
        data_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load equity bars filtered by date and optional symbol, bar_size, data_type.
        """
        if isinstance(trade_date, date):
            trade_date = trade_date.isoformat()
        path = self.base_path
        if symbol:
            path = path / f"symbol={symbol}"
        if bar_size:
            path = path / f"bar_size={bar_size}"
        if data_type:
            path = path / f"data_type={data_type}"
        path = path / f"trade_date={trade_date}"
        dataset = ds.dataset(str(path), format='parquet')
        filters = []
        if symbol:
            filters.append(("symbol","=",symbol))
        table = dataset.to_table(filter=filters if filters else None)
        return table.to_pandas()
