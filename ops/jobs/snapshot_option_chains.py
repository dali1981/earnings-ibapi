#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import socket
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import DATA_ROOT, EXPORTS_PATH, IB_CLIENT_ID, IB_HOST, IB_PORT
from ibx_flows.backfill import BackfillConfig, backfill_option_chain_daily
from ibx_flows.source_ib import IBSource
from common.repositories.option_chains import OptionChainSnapshotRepository


log = logging.getLogger(__name__)


def can_connect(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class FakeSource:
    """Offline source that returns one minimal chain row for the requested date."""

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot current option chain")
    ap.add_argument("--symbol", required=True, help="Underlying symbol")
    ap.add_argument("--date", default=datetime.today().date().isoformat(), help="Snapshot date label (YYYY-MM-DD)")
    ap.add_argument("--offline", action="store_true", help="Run without IB (safe demo write)")
    ap.add_argument("--host", default=IB_HOST)
    ap.add_argument("--port", type=int, default=IB_PORT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("Snapshotting option chain for %s (label date=%s)", args.symbol, args.date)

    snap_date = datetime.fromisoformat(args.date).date()
    cfg = BackfillConfig(underlying=args.symbol, start=snap_date, end=snap_date)

    if args.offline:
        base = EXPORTS_PATH / "demo"
        src = FakeSource()
        log.info("OFFLINE mode: writing under %s", base)
    else:
        if not can_connect(args.host, args.port):
            log.error("Cannot connect to IB at %s:%s; use --offline or start IB.", args.host, args.port)
            sys.exit(2)
        src = IBSource(host=args.host, port=args.port, client_id=IB_CLIENT_ID)
        base = DATA_ROOT

    repo = OptionChainSnapshotRepository(base)
    backfill_option_chain_daily(src, repo, cfg)
    log.info("Snapshot complete: %s", base / "option_chains")


if __name__ == "__main__":
    main()

