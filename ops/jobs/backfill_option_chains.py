#!/usr/bin/env python3
# jobs/backfill_option_chains.py (Refactored)
from __future__ import annotations

from pathlib import Path

import socket
import sys


# Ensure project root on sys.path for config and local imports
sys.path.append(str(Path(__file__).parent.parent))

from common.repositories.option_chains import OptionChainSnapshotRepository
from config import DATA_ROOT, EXPORTS_PATH, IB_CLIENT_ID, IB_HOST, IB_PORT
from datetime import datetime, timedelta
from ibx_flows.backfill import backfill_option_chain_daily, BackfillConfig
from ibx_flows.source_ib import IBSource

import argparse
import logging
import pandas as pd


log = logging.getLogger(__name__)


def can_connect(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class FakeSource:
    """Offline test source that returns a simple chain snapshot (no network)."""

    def get_option_chain(self, underlying: str, asof_date):
        return pd.DataFrame(
            [
                {
                    "underlying": underlying,
                    "date": asof_date,
                    "expiry": asof_date + timedelta(days=30),
                    "right": "C",
                    "strike": 100.0,
                    "conid": 111,
                }
            ]
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Backfill option chain snapshots")
    p.add_argument("--symbol", type=str, required=True, help="underlying symbol")
    p.add_argument(
        "--start",
        type=str,
        required=True,
        help="YYYY-MM-DD (deprecated; snapshots are current-only)",
    )
    p.add_argument(
        "--end",
        type=str,
        required=True,
        help="YYYY-MM-DD (deprecated; snapshots are current-only)",
    )
    p.add_argument(
        "--offline", action="store_true", help="Run without IB (fake source)"
    )
    p.add_argument("--host", default=IB_HOST)
    p.add_argument("--port", type=int, default=IB_PORT)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    log.info(
        f"Starting option chain backfill for {args.symbol} from {args.start} to {args.end}"
    )

    # Deprecated behavior: IB does not provide historical chains; treat as a labeled snapshot (start=end)
    log.warning(
        "backfill_option_chains is deprecated. Use jobs/snapshot_option_chains.py --date instead."
    )
    snap_date = datetime.fromisoformat(args.end).date()
    cfg = BackfillConfig(underlying=args.symbol, start=snap_date, end=snap_date)

    if args.offline:
        log.info("Running in OFFLINE mode (fake source)")
        src = FakeSource()
        base_root = EXPORTS_PATH / "demo"
        log.info("Offline write base: %s", base_root)
    else:
        if not can_connect(args.host, args.port, timeout=2.0):
            log.error(
                "Cannot connect to IB at %s:%s. Use --offline for a local test or start IB Gateway/TWS.",
                args.host,
                args.port,
            )
            sys.exit(2)
        try:
            src = IBSource(host=args.host, port=args.port, client_id=IB_CLIENT_ID)
        except Exception as e:
            log.error(
                "IB runtime failed to start (%s). Use --offline for a local test or check IB settings.",
                e,
            )
            sys.exit(2)
        base_root = DATA_ROOT

    repo = OptionChainSnapshotRepository(base_root)
    backfill_option_chain_daily(src, repo, cfg)
    log.info("Option chain backfill complete.")
