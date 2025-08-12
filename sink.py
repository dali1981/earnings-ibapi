import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from option_repositories import ChainRepository, BarRepository
from repos.equities import EquityBarRepository
from utils import append_parquet


class Sink:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.buffers: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        self.chain_repo = ChainRepository(self.out_dir)  # Placeholder for chain repository if needed
        self.equity_repo = EquityBarRepository(self.out_dir)

    def add_value(self, req_id: int, value: Any) -> None:
        """
        Add a request ID and tag to the buffer.
        """
        self.buffers[req_id].append(value)


    def historical_data(self, req_id: int, bar):
        # logging.info("BAR %d %s %.2f", req_id, bar.date, bar.close)
        # TODO: append to parquet / csv here
        self.buffers[req_id].append(
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "wap": bar.wap,
            }
        )

    def historical_data_end(self, req_id: int, start: str, end: str):
        logging.info("END %d", req_id)
        append_parquet(self.out_dir / f"bars_{req_id}.parquet", self.buffers.pop(req_id, []))

    def tick_option_computation(self, req_id: int, tick_type: int, *_):
        iv, delta, gamma, vega, theta = _[:5]
        self.buffers[req_id].append(
            {
                "tickType": tick_type,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
            }
        )

    def tick_snapshot_end(self, req_id: int):
        logging.info("SNAPSHOT END %d", req_id)
        append_parquet(self.out_dir / f"snapshot_{req_id}.parquet", self.buffers.pop(req_id, []))

    def option_chain(self, req_id: int):
        """
        Save option chain data to repository.
        """
        chain_data = self.buffers.pop(req_id, None)
        self.chain_repo.save(chain_data)
        logging.info("Saved option chain for req_id %d", req_id)



class OptionBarSink:
    """
    Sink for option bar DataFrames. Knows about OptionBarRepository.
    Adds underlying/expiry/strike/right columns from df.attrs before saving.
    """
    def __init__(self, repo: BarRepository):
        self._repo = repo

    def __call__(self, df: pd.DataFrame):
        # Enrich frame with option identifiers from metadata for persistence
        out = df.copy()
        meta = out.attrs or {}
        # Attach identifiers
        out['underlying'] = meta.get('underlying') or meta.get('symbol')
        # expiry may be a date or string; normalize to date
        exp = meta.get('expiry')
        try:
            if exp is not None:
                out['expiry'] = pd.to_datetime(exp).date()
            else:
                out['expiry'] = pd.NaT
        except Exception:
            out['expiry'] = pd.NaT

        strike = meta.get('strike')

        out['strike'] = float(strike) if strike is not None else None
        out['right'] = meta.get('right')
        # Carry bar_size & what_to_show for repos that read from attrs
        out.attrs['bar_size'] = meta.get('bar_size')
        out.attrs['what_to_show'] = meta.get('what_to_show')
        # Ensure date column exists for intraday partitioning
        if 'date' not in out.columns and 'time' in out.columns:
            out['date'] = pd.to_datetime(out['time']).dt.date
        # Save to repository
        self._repo.save(out)

class EquityBarSink:
    """
    Sink for equity bar DataFrames. Knows about EquityBarRepository.
    """
    def __init__(self, repo: EquityBarRepository):
        self._repo = repo

    def __call__(self, df: pd.DataFrame):
        # Save bar data to equity repository
        self._repo.save(df)