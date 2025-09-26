#!/usr/bin/env python3
"""
Show upcoming earnings from the local repository (no external calls).

Usage examples:
  python scripts/show_upcoming_earnings.py --days 14
  python scripts/show_upcoming_earnings.py --symbols AAPL,MSFT,GOOGL --days 21
  python scripts/show_upcoming_earnings.py --demo --days 14
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import argparse
import sys


# Ensure project root on path
sys.path.append(str(Path(__file__).parent.parent))

from common.repositories.earnings import EarningsCalendarRepository

import config
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Display upcoming earnings from the local DB"
    )
    p.add_argument(
        "--days", type=int, default=30, help="Days ahead to display (default: 30)"
    )
    p.add_argument(
        "--symbols", type=str, help="Comma-separated list of symbols to filter"
    )
    p.add_argument("--source", type=str, help="Filter by source (e.g., nasdaq)")
    p.add_argument(
        "--limit", type=int, default=100, help="Max rows to show (default: 100)"
    )
    p.add_argument(
        "--demo", action="store_true", help="Read from demo base (data/exports/demo)"
    )
    return p.parse_args()


def trim(text: str, width: int) -> str:
    s = text or ""
    return s if len(s) <= width else s[: width - 1] + "…"


def main() -> None:
    args = parse_args()
    base_root = config.EXPORTS_PATH / "demo" if args.demo else config.DATA_ROOT
    repo = EarningsCalendarRepository(base_root)

    today = date.today()
    end = today + timedelta(days=args.days)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    df = repo.get_earnings_by_date_range(
        start_date=today, end_date=end, symbols=symbols
    )

    if args.source:
        df = df[df["source"].str.lower() == args.source.lower()]

    if df.empty:
        print("No upcoming earnings found in local DB for the given filters.")
        return

    # Keep only future or today
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
    df = df[df["earnings_date"] >= today]
    df = df.sort_values(["earnings_date", "symbol"]).head(args.limit)

    # Pretty print
    print(
        f"📅 Upcoming earnings (next {args.days} days){' [demo]' if args.demo else ''}"
    )
    print("=" * 60)
    header = f"{'Date':<12} {'Symbol':<8} {'Time':<5} {'Source':<8} Company"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        day = row["earnings_date"].isoformat()
        sym = row["symbol"]
        time_str = (row.get("time") or "").upper()[:5]
        src = (row.get("source") or "")[:8]
        company = trim(str(row.get("company_name") or ""), 60)
        print(f"{day:<12} {sym:<8} {time_str:<5} {src:<8} {company}")


if __name__ == "__main__":
    main()
