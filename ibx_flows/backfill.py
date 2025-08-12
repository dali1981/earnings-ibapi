from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import pandas as pd
# Use new unified repository system
from repositories import EquityBarRepository, OptionChainSnapshotRepository, OptionBarRepository
from .source_interface import MarketDataSource
from .select_strikes import StrikeSelectionConfig, select_contracts
from .windows import missing_windows

@dataclass
class BackfillConfig:
    underlying: str
    start: date
    end: date
    bar_size: str = "1 day"
    expiries_max: int = 8
    strikes_per_side: int = 6
    selection_mode: str = "k_around_atm"

def backfill_equity_bars(src: MarketDataSource, repo: EquityBarRepository, cfg: BackfillConfig) -> None:
    # Use new repository interface for present_dates
    present = repo.present_dates(cfg.underlying, cfg.bar_size, cfg.start, cfg.end)
    windows = missing_windows(present, cfg.start, cfg.end)
    if not windows: return
    frames = []
    for ws, we in windows:
        df = src.get_equity_bars(cfg.underlying, ws, we, cfg.bar_size)
        if not df.empty: frames.append(df)
    # The new repository can infer symbol and bar_size from the DataFrame
    if frames: repo.save(pd.concat(frames, ignore_index=True))

def backfill_option_chain_daily(src: MarketDataSource, repo: OptionChainSnapshotRepository, cfg: BackfillConfig) -> None:
    asofs = pd.bdate_range(cfg.start, cfg.end).date; frames = []
    for d in asofs:
        try: snap = src.get_option_chain(cfg.underlying, d)
        except Exception: continue
        frames.append(snap)
    if frames: repo.save(pd.concat(frames, ignore_index=True))

def backfill_option_bars(src: MarketDataSource, repo: OptionBarRepository, chain_repo: OptionChainSnapshotRepository|None, eq_repo: EquityBarRepository|None, cfg: BackfillConfig) -> None:
    chain = src.get_option_chain(cfg.underlying, cfg.end)
    eq = src.get_equity_bars(cfg.underlying, cfg.end, cfg.end, cfg.bar_size)
    if eq.empty: raise RuntimeError("No equity bars for spot selection on end date")
    spot = float(eq.iloc[-1]["close"])
    filtered = select_contracts(chain_snapshot=chain, spot_price=spot, cfg=StrikeSelectionConfig(expiries_max=cfg.expiries_max, strikes_per_side=cfg.strikes_per_side), mode=cfg.selection_mode)
    frames = []
    for _, row in filtered.iterrows():
        present = repo.present_dates_for_contract(row["underlying"], row["expiry"], row["right"], float(row["strike"]), cfg.bar_size, pd.to_datetime(cfg.start), pd.to_datetime(cfg.end))
        windows = missing_windows(present, cfg.start, cfg.end)
        if not windows: continue
        contract_df = pd.DataFrame([{k: row[k] for k in ["underlying","expiry","right","strike","conid"] if k in row}])
        for ws, we in windows:
            bars = src.get_option_bars(contract_df, ws, we, cfg.bar_size)
            if not bars.empty: frames.append(bars)
    if frames: repo.save(pd.concat(frames, ignore_index=True))
