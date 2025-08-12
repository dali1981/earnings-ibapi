# jobs/backfill_option_chains.py (Refactored)
from __future__ import annotations
import argparse
import logging
from datetime import datetime
from ibx_repos import OptionChainSnapshotRepository
from ibx_flows.backfill import BackfillConfig, backfill_option_chain_daily
from ibx_flows.source_ib import IBSource
from config import (
    OPTION_CHAINS_PATH,
    IB_HOST,
    IB_PORT,
    IB_CLIENT_ID
)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, help="underlying symbol")
    p.add_argument("--start", type=str, default="2025-07-01")
    p.add_argument("--end", type=str, default="2025-08-31")
    args = p.parse_args()

    log.info(f"Starting option chain backfill for {args.symbol} from {args.start} to {args.end}")

    cfg = BackfillConfig(
        underlying=args.symbol,
        start=datetime.fromisoformat(args.start).date(),
        end=datetime.fromisoformat(args.end).date()
    )

    # Paths and connection details are now from the central config
    repo = OptionChainSnapshotRepository(OPTION_CHAINS_PATH)
    src = IBSource(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID)

    backfill_option_chain_daily(src, repo, cfg)
    log.info("Option chain backfill complete.")