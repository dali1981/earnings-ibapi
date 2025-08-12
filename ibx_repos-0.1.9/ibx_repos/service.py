
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import date
import pandas as pd
from ibx import IBRuntime, ContractDetailsService, SecDefService, HistoricalService, make_stock, make_option
from .equity_bars import EquityBarRepository
from .option_bars import OptionBarRepository, OptionMeta
from .chains import OptionChainSnapshotRepository

@dataclass
class FetchSpec:
    symbol: str
    bar_size: str
    duration: str
    what_to_show: str = "TRADES"
    use_rth: int = 0

class IBXDataService:
    """High-level helpers to fetch from IB via ibx services and persist via repositories."""
    def __init__(self, rt: IBRuntime, equity_repo: EquityBarRepository, option_repo: OptionBarRepository, chain_repo: OptionChainSnapshotRepository):
        self.rt = rt
        self.cds = ContractDetailsService(rt)
        self.secdef = SecDefService(rt)
        self.hist = HistoricalService(rt)
        self.eq_repo = equity_repo
        self.opt_repo = option_repo
        self.chain_repo = chain_repo

    # ---------- Equity ----------
    def fetch_equity_bars(self, spec: FetchSpec, end_datetime: str = "") -> pd.DataFrame:
        stk = make_stock(spec.symbol)
        rows = self.hist.bars(stk, end_datetime, spec.duration, spec.bar_size, spec.what_to_show, spec.use_rth, timeout=60.0)
        df = pd.DataFrame(rows)
        self.eq_repo.save(df, symbol=spec.symbol, bar_size=spec.bar_size, what_to_show=spec.what_to_show)
        return df

    # ---------- Chains ----------
    def snapshot_chain(self, symbol: str) -> pd.DataFrame:
        # Get conid for the primary listing
        cds = self.cds.fetch(make_stock(symbol))
        conid = cds[0].contract.conId
        snap = self.secdef.option_params(symbol, conid, sec_type="STK", timeout=30.0)
        self.chain_repo.save(symbol, conid, snap)
        return pd.DataFrame(snap)

    # ---------- Options ----------
    def fetch_option_bars(self, meta: OptionMeta, end_datetime: str = "", duration: str = "1 D", use_rth: int = 0) -> pd.DataFrame:
        opt = make_option(meta.underlying, meta.expiry, meta.strike, meta.right)
        rows = self.hist.bars(opt, end_datetime, duration, meta.bar_size, meta.what_to_show or "TRADES", use_rth, timeout=60.0)
        df = pd.DataFrame(rows)
        self.opt_repo.save(df, meta)
        return df
