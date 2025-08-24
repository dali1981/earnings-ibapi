#!/usr/bin/env python3
"""
Bootstrap market data for symbols with earning dates.

Reads the latest earning-date CSV, selects symbols, and for each symbol:
- Ensures an option chain snapshot is present (refreshes if stale)
- Backfills equity history
- Backfills option history

Notes
- This script uses NASDAQ-derived earning dates exported by earnings.discovery.
- It requires an active IB Gateway/TWS for chain/equity/option requests.

Usage examples
  python jobs/run_from_earnings.py --max-symbols 25 --lookback-days 365
  python jobs/run_from_earnings.py --min-days-until 0 --max-symbols 10 --data-path data
  python jobs/run_from_earnings.py --symbols AAPL,MSFT,NVDA --lookback-days 180
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import argparse
import logging
import pandas as pd
import sys


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    CONTRACTS_PATH,
    DATA_ROOT,
    IB_CLIENT_ID,
    IB_HOST,
    IB_PORT,
    OPTION_CHAINS_PATH,
)
from earnings.discovery import EarningsDiscoveryEngine
from jobs.tasks import BackfillEquityBarsTask, BackfillOptionBarsTask
from common.repositories.option_chains import OptionChainSnapshotRepository


try:
    # Prefer ibx-flows source (unified IB access)
    from ibx_flows.backfill import backfill_option_chain_daily, BackfillConfig
    from ibx_flows.contracts_backfill import (
        backfill_us_equity_contracts,
        ContractsConfig,
    )
    from ibx_flows.source_ib import IBSource
    from ibx_repos.contracts import ContractRepository
except Exception as e:  # pragma: no cover
    print(
        "ibx_flows is required for chain/equity/option backfills:", e, file=sys.stderr
    )
    sys.exit(2)


log = logging.getLogger("jobs.run_from_earnings")


def load_symbols_from_earnings(
    max_age_days: int = 1,
    min_days_until: Optional[int] = None,
    max_symbols: Optional[int] = None,
) -> List[str]:
    eng = EarningsDiscoveryEngine()
    df = eng.get_recent_discoveries(max_age_days=max_age_days)
    if df is None or df.empty:
        return []

    if min_days_until is not None:
        df = df[df["days_until"] >= min_days_until]

    # Keep simple uppercase symbols without special chars
    df = df[df["symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
    syms = df["symbol"].dropna().drop_duplicates().tolist()
    if max_symbols:
        syms = syms[:max_symbols]
    return syms


def ensure_chain_snapshot(
    symbol: str, chain_repo: OptionChainSnapshotRepository, refresh_hours: int = 24
) -> None:
    try:
        latest = chain_repo.get_latest_snapshot_date(symbol)
    except Exception:
        latest = None

    need_refresh = True
    today = date.today()
    if latest is not None:
        age = (today - latest).days * 24
        need_refresh = age >= refresh_hours

    if need_refresh:
        log.info("Refreshing option chain snapshot for %s", symbol)
        src = IBSource(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID)
        cfg = BackfillConfig(underlying=symbol, start=today, end=today)
        backfill_option_chain_daily(src, chain_repo, cfg)
    else:
        log.info("Option chain snapshot is recent for %s (latest=%s)", symbol, latest)


def process_symbol(
    symbol: str, base_path: Path, lookback_days: int, option_bar_size: str = "1 min"
) -> None:
    end = date.today()
    start = end - timedelta(days=lookback_days)

    # 0) Contracts (targeted backfill if missing)
    try:
        repo = ContractRepository(CONTRACTS_PATH)
        existing = (
            repo.find_by_symbol(symbol) if hasattr(repo, "find_by_symbol") else None
        )
        need_contracts = existing is None or (
            isinstance(existing, pd.DataFrame) and existing.empty
        )
    except Exception:
        need_contracts = True

    if need_contracts:
        log.info("Backfilling contract descriptions for %s", symbol)
        src = IBSource(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID)
        cfg = ContractsConfig(
            exchanges=["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"],
            currency="USD",
            patterns=[symbol],
        )
        backfill_us_equity_contracts(src, ContractRepository(CONTRACTS_PATH), cfg)
    else:
        log.info("Contract descriptions present for %s", symbol)

    # 1) Chains
    chain_repo = OptionChainSnapshotRepository(base_path / "option_chains")
    ensure_chain_snapshot(symbol, chain_repo, refresh_hours=24)

    # 2) Equity bars
    BackfillEquityBarsTask(
        symbol=symbol,
        start=start,
        end=end,
        bar_size="1 day",
        out=base_path / "equity_bars",
    ).run()

    # 3) Option bars (default: one-minute granularity)
    BackfillOptionBarsTask(
        symbol=symbol,
        start=start,
        end=end,
        bar_size=option_bar_size,
        out=base_path / "option_bars",
        chain_base=base_path,
        expiries_max=8,
        strikes_per_side=6,
        selection_mode="k_around_atm",
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill data for symbols with earning dates"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols; if empty, read from latest earning-date CSV",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=20,
        help="Limit symbols selected from earnings table",
    )
    parser.add_argument(
        "--min-days-until",
        type=int,
        default=0,
        help="Only include symbols with days_until >= this value (default: 0 for upcoming)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Historical lookback for equity/option bars (default: 365)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_ROOT,
        help="Base path for repositories (default: config.DATA_ROOT)",
    )
    parser.add_argument(
        "--option-bar-size",
        type=str,
        default="1 min",
        help="Bar size for option history (default: '1 min')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned symbols and actions without making IB requests",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    base_path = Path(args.data_path)
    base_path.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols_from_earnings(
            max_age_days=1,
            min_days_until=args.min_days_until,
            max_symbols=args.max_symbols,
        )

    if not symbols:
        print("No symbols to process (check earnings export or --symbols)")
        sys.exit(0)

    log.info("Processing %d symbols: %s", len(symbols), ", ".join(symbols))

    if args.dry_run:
        print("DRY RUN - planned actions:\n")
        print(f"Data path: {base_path}")
        print(f"Lookback days: {args.lookback_days}")
        print(f"Option bar size: {args.option_bar_size}")
        print("Symbols:", ", ".join(symbols))
        print("\nPer symbol steps:")
        print("  1) Ensure contracts present (targeted backfill if missing)")
        print("  2) Ensure option chain snapshot (refresh if >24h old)")
        print("  3) Backfill equity bars [1 day]")
        print(f"  4) Backfill option bars [{args.option_bar_size}]")
        sys.exit(0)
    for sym in symbols:
        try:
            process_symbol(
                sym, base_path, args.lookback_days, option_bar_size=args.option_bar_size
            )
        except Exception as e:
            log.error("Failed processing %s: %s", sym, e)


if __name__ == "__main__":
    main()
