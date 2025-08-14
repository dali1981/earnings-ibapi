"""
Example: Ingest external bars -> write to repo -> notify -> read locally.

Run:
    python scripts/example_local_data_flow.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import random
import sys


# Ensure project root on sys.path when run directly
sys.path.append(str(Path(__file__).parent.parent))

from utils.local_data import get_equity_bars, persist_equity_bars_and_notify

import config


def _fake_external_daily_bars(symbol: str, days: int = 3) -> pd.DataFrame:
    today = datetime.utcnow().date()
    rows = []
    price = 100.0
    for i in range(days):
        d = today - timedelta(days=i + 1)
        change = random.uniform(-2, 2)
        o = price
        c = max(1.0, price + change)
        h = max(o, c) + random.uniform(0, 1)
        l = min(o, c) - random.uniform(0, 1)
        v = random.randint(1_000_000, 5_000_000)
        rows.append(
            {"date": d, "open": o, "high": h, "low": l, "close": c, "volume": v}
        )
        price = c
    return pd.DataFrame(rows)


def main():
    symbol = "DEMO"
    # Use an isolated demo base path under data/exports to avoid mixing with production datasets
    demo_base = config.EXPORTS_PATH / "demo"

    # 1) External fetch (simulated here)
    df_ext = _fake_external_daily_bars(symbol, days=3)

    # 2) Persist to local repo + notify
    persist_equity_bars_and_notify(
        df_ext,
        symbol=symbol,
        bar_size="1 day",
        data_type="TRADES",
        base_path=demo_base,
        dataset_topic="equity_bars.demo",
    )

    # 3) Downstream consumers read locally only
    df_local = get_equity_bars(symbol, bar_size="1 day", base_path=demo_base)
    print(
        f"Local rows for {symbol}: {len(df_local)}; stored under {demo_base / 'equity_bars'}"
    )


if __name__ == "__main__":
    main()
