#!/usr/bin/env python3
"""
Example: Inspect contract descriptions from the local repository.

Loads contract descriptions and prints unique stock symbols available in the
local DB. This is a read-only example; it does not call external APIs.

Usage:
  python examples/get_with_options_example.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root on path
sys.path.append(str(Path(__file__).parent.parent))

import config
import pandas as pd
from common.repositories.contract_descriptions import ContractDescriptionsRepository


def main() -> None:
    repo = ContractDescriptionsRepository(config.DATA_ROOT)
    try:
        df = repo.load()
    except Exception as e:
        print(f"Failed to load contract descriptions: {e}")
        return

    if df is None or df.empty:
        print("No contract descriptions found in local DB.")
        return

    # Normalize types for safety
    if "sec_type" in df.columns:
        stk = df[df["sec_type"].str.upper() == "STK"].copy()
    else:
        stk = df.copy()

    symbols = (
        stk[["symbol"]].dropna().drop_duplicates().sort_values("symbol").symbol.tolist()
    )
    print(f"Total unique stock symbols in DB: {len(symbols)}")
    print("Sample:")
    for s in symbols[:50]:
        print(f"  {s}")


if __name__ == "__main__":
    main()

