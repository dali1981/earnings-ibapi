from pathlib import Path
from datetime import date
from typing import List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import duckdb
import pandas as pd



class ChainRepository:
    """
    Repository for storing and loading option-chain metadata as Parquet.
    Directory layout: base_path/chains/underlying=<symbol>/*.parquet
    """

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path) / "chains"
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Define Arrow schema
        self.schema = pa.schema([
            pa.field('underlying_conid', pa.int64()),
            pa.field('exchange', pa.string()),
            pa.field('trading_class', pa.string()),
            pa.field('multiplier', pa.int64()),
            pa.field('expirations', pa.list_(pa.string())),
            pa.field('strikes', pa.list_(pa.float64())),
            pa.field('date', pa.date32()),  # date when data was fetched
        ])

    def save(self, data: Union[pd.DataFrame, List[dict]]) -> None:
        """
        Save the chain DataFrame to Parquet, partitioned by underlying.
        """
        # Convert raw list to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Cast types and add ingestion_date
        df['underlying_conid'] = df['underlying_conid'].astype(int)
        df['multiplier'] = df['multiplier'].astype(int)
        # Ensure expirations are strings and strikes floats
        df['expirations'] = df['expirations'].apply(lambda x: list(x))
        df['strikes'] = df['strikes'].apply(lambda x: [float(s) for s in x])
        df['date'] = date.today()

        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(self.base_path),
            partition_cols=['underlying_conid'],
            use_dictionary=True,
            compression='snappy'
        )

    def load(
        self,
        underlying: Optional[str] = None,
        expiry: Optional[Union[str, date]] = None
    ) -> pd.DataFrame:
        """
        Load chains, optionally filtering by underlying and expiry.
        """
        dataset = ds.dataset(str(self.base_path), format='parquet')
        filters = []
        if underlying:
            filters.append(("underlying", "=", underlying))
        if expiry:
            if isinstance(expiry, date):
                expiry = expiry.isoformat()
            filters.append(("expiry", "=", expiry))
        table = dataset.to_table(filter=filters if filters else None)
        return table.to_pandas()

    # def load(self, underlying_conid: Optional[int] = None) -> pd.DataFrame:
    #     """
    #     Load raw chain parameters, optionally filtering by underlying_conid.
    #     """
    #     dataset = ds.dataset(str(self.base_path), format='parquet')
    #     filters = []
    #     if underlying_conid is not None:
    #         filters.append(("underlying_conid", "=", underlying_conid))
    #     table = dataset.to_table(filter=filters if filters else None)
    #     return table.to_pandas()
class BarRepository:
    """
    Repository for storing and loading 1-minute option bars as Parquet.
    Directory layout: base_path/bars/trade_date=YYYY-MM-DD/*.parquet
    """

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path) / "bars"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema([
            pa.field('trade_time', pa.timestamp('ns')),
            pa.field('contract_id', pa.int64()),
            pa.field('open', pa.float64()),
            pa.field('high', pa.float64()),
            pa.field('low', pa.float64()),
            pa.field('close', pa.float64()),
            pa.field('volume', pa.int64()),
            pa.field('open_interest', pa.int64()),
            pa.field('trade_date', pa.date32()),  # extracted for partitioning
        ])

    def save(self, df: pd.DataFrame) -> None:
        """
        Save minute-bar DataFrame to Parquet, partitioned by trade_date.
        Assumes df contains 'trade_time' datetime64[ns].
        """
        # extract date for partition
        df['trade_date'] = df['trade_time'].dt.date
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(self.base_path),
            partition_cols=['trade_date'],
            use_dictionary=True,
            compression='snappy'
        )

    def load(
        self,
        trade_date: Union[str, date],
        contract_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load bars for a given date and optional contract filter.
        """
        if isinstance(trade_date, date):
            trade_date = trade_date.isoformat()
        partition_path = self.base_path / f"trade_date={trade_date}"
        dataset = ds.dataset(str(partition_path), format='parquet')
        filters = []
        if contract_id:
            filters.append(("contract_id", "=", contract_id))
        table = dataset.to_table(filter=filters if filters else None)
        return table.to_pandas()


# class IBOptionDataService:
#     """
#     Service to fetch from IB and persist using repositories.
#     """
#
#     def __init__(
#         self,
#         ib_client: IB,
#         chain_repo: OptionChainSnapshotRepository,
#         bar_repo: BarRepository
#     ):
#         self.ib = ib_client
#         self.chain_repo = chain_repo
#         self.bar_repo = bar_repo
#
#     def update_chains(self, symbols: List[str]) -> None:
#         """
#         Fetch contract details for each symbol and store metadata.
#         """
#         records = []
#         for sym in symbols:
#             details = self.ib.reqContractDetails(Option(sym, '', 0, ''))
#             for d in details:
#                 c = d.contract
#                 records.append({
#                     'contract_id': c.conId,
#                     'underlying': c.symbol,
#                     'strike': c.strike,
#                     'right': c.right,
#                     'expiry': pd.to_datetime(c.lastTradeDateOrContractMonth).date(),
#                     'multiplier': int(d.contract.multiplier),
#                     'currency': c.currency,
#                     'exchange': c.exchange,
#                 })
#         df = pd.DataFrame(records)
#         self.chain_repo.save(df)
#
#     def fetch_and_store_bars(
#         self,
#         contract_ids: List[int],
#         as_of_date: date
#     ) -> None:
#         """
#         Fetch 1-minute bars for given contracts on as_of_date and store.
#         """
#         for cid in contract_ids:
#             # build IB contract object
#             # here we assume chain_repo has metadata to rebuild Option
#             meta = self.chain_repo.load().query(f"contract_id == {cid}").iloc[0]
#             opt = Option(
#                 meta['underlying'],
#                 meta['expiry'].strftime('%Y%m%d'),
#                 meta['strike'],
#                 meta['right']
#             )
#             bars = self.ib.reqHistoricalData(
#                 opt,
#                 endDateTime=f"{as_of_date.strftime('%Y%m%d')} 23:59:59",
#                 durationStr='1 D',
#                 barSizeSetting='1 min',
#                 whatToShow='TRADES',
#                 useRTH=True
#             )
#             df = util.df(bars)
#             df['contract_id'] = cid
#             self.bar_repo.save(df)
#
# # Example usage
# if __name__ == '__main__':
#     ib = IB()
#     ib.connect('127.0.0.1', 7497, clientId=1)
#
#     base = 'data'
#     chain_repo = OptionChainSnapshotRepository(base)
#     bar_repo = BarRepository(base)
#     service = IBOptionDataService(ib, chain_repo, bar_repo)
#
#     # update chains for selected symbols
#     service.update_chains(['AAPL', 'MSFT', 'GOOG'])
#
#     # fetch and store bars for today
#     service.fetch_and_store_bars(
#         contract_ids=[123456, 234567],
#         as_of_date=date.today()
#     )
