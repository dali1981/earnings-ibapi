#!/usr/bin/env python3
"""
Option Chain Backfill Consumer

Consumes option_chains events from the local file-based event bus and backfills
historical option bars for the referenced underlying and date window.

Workflow per event:
  1) Read event payload (underlying, start, end, optional bar_size)
  2) Run backfill_option_bars via IBSource -> writes to repository
  3) Backfill code emits its own data.available event for option_bars
  4) Mark idempotency key as processed and advance checkpoint

Usage:
  python jobs/option_chain_backfill_consumer.py \
    --service backfill-broker --topic option_chains --bar-size "1 day" --limit 50 --once
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import argparse
import sys


# Ensure project root on path
sys.path.append(str(Path(__file__).parent.parent))

from ibx_flows.backfill import backfill_option_bars, BackfillConfig
from ibx_flows.source_ib import IBSource
from repositories.option_bars import OptionBarRepository
from utils.event_bus_consumer import JSONLConsumer
from utils.processed_index import ProcessedIndex

import config
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consume option_chains events and backfill option bars"
    )
    p.add_argument(
        "--service",
        default="backfill-broker",
        help="Service name for checkpointing (default: backfill-broker)",
    )
    p.add_argument(
        "--topic",
        default="option_chains",
        help="Event topic (option_chains or option_chains.demo)",
    )
    p.add_argument(
        "--bar-size",
        default="1 day",
        help="Bar size for option bars (default: '1 day')",
    )
    p.add_argument(
        "--limit", type=int, default=50, help="Max events per batch (default: 50)"
    )
    p.add_argument(
        "--base-path",
        type=Path,
        default=config.DATA_ROOT,
        help="Repository base path for writing (default: config.DATA_ROOT)",
    )
    p.add_argument("--once", action="store_true", help="Process one batch and exit")
    return p.parse_args()


def handle_event(payload: dict, repo: OptionBarRepository, bar_size: str) -> None:
    underlying = payload.get("underlying")
    if not underlying:
        return
    start = payload.get("start")
    end = payload.get("end")
    # Parse dates with fallback to today
    start_d = pd.to_datetime(start).date() if start else date.today()
    end_d = pd.to_datetime(end).date() if end else start_d

    cfg = BackfillConfig(
        underlying=underlying,
        start=start_d,
        end=end_d,
        bar_size=bar_size,
    )
    src = IBSource()
    backfill_option_bars(src, repo, chain_repo=None, eq_repo=None, cfg=cfg)


def main() -> None:
    args = parse_args()
    cons = JSONLConsumer(args.topic)
    processed = ProcessedIndex(service=args.service, topic=args.topic)
    repo = OptionBarRepository(args.base_path)

    try:
        while True:
            events = cons.read(service=args.service, limit=args.limit)
            if not events:
                if args.once:
                    break
                # Real service could sleep here
                break
            for evt in events:
                payload = evt.get("payload", {})
                key = f"{payload.get('underlying')}:{payload.get('start')}:{payload.get('end')}:{args.bar_size}"
                if processed.is_processed(key):
                    continue
                handle_event(payload, repo, args.bar_size)
                processed.mark_processed(key)
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
