"""
Local data access helpers enforcing a read-from-repos, write-then-notify pattern.

- Consumers: read via repository classes only (no direct API calls).
- Producers: fetch externally, persist to repo, then notify via EventBus.
"""

from __future__ import annotations

from pathlib import Path
from repositories.equity_bars import EquityBarRepository
from typing import Any, Dict, Optional
from utils.events.bus import EventBus, FileEventBus

import config
import pandas as pd


def get_equity_bars(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bar_size: Optional[str] = None,
    base_path: Path | str = config.DATA_ROOT,
) -> pd.DataFrame:
    """Read-only convenience accessor for local equity bars.

    Example:
        df = get_equity_bars("AAPL", start_date="2024-01-01", bar_size="1 day")
    """
    repo = EquityBarRepository(base_path)
    return repo.load_symbol_data(
        symbol=symbol, start_date=start_date, end_date=end_date, bar_size=bar_size
    )


def persist_equity_bars_and_notify(
    df: pd.DataFrame,
    symbol: str,
    bar_size: str,
    data_type: str = "TRADES",
    base_path: Path | str = config.DATA_ROOT,
    bus: Optional[EventBus] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    dataset_topic: Optional[str] = None,
) -> None:
    """Persist externally-fetched equity bars locally, then notify downstream services.

    This function encodes the repository-first policy: all external fetches must be
    written to local storage before other services consume the data.
    """
    repo = EquityBarRepository(base_path)
    repo.save(df, symbol=symbol, bar_size=bar_size, data_type=data_type)

    bus = bus or FileEventBus()
    meta = {"symbol": symbol, "bar_size": bar_size, "rows": int(len(df))}
    if extra_meta:
        meta.update(extra_meta)
    topic = dataset_topic or "equity_bars"
    bus.notify_data_available(dataset=topic, **meta)
