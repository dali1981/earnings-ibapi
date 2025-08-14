#!/usr/bin/env python3
"""
Consume equity_bars events from the local JSONL event bus and process idempotently.

Usage:
  python scripts/consume_equity_bars_events.py \
    --service analytics --topic equity_bars --limit 100 --once

By default reads events from data/cache/events/<topic>.jsonl and tracks
per-service checkpoints + processed keys under data/cache/state/.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import argparse
import sys


# Ensure project root on path
sys.path.append(str(Path(__file__).parent.parent))

from repositories.equity_bars import EquityBarRepository
from utils.events.consumer import JSONLConsumer
from utils.events.processed_index import ProcessedIndex

import config
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consume local equity_bars events and process idempotently"
    )
    p.add_argument(
        "--service",
        default="analytics",
        help="Consumer service name for checkpointing (default: analytics)",
    )
    p.add_argument(
        "--topic",
        default="equity_bars",
        help="Event topic (e.g., equity_bars or equity_bars.demo)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max events to read per batch (default: 100)",
    )
    p.add_argument(
        "--base-path",
        type=Path,
        default=config.DATA_ROOT,
        help="Repository base path for reading (default: config.DATA_ROOT)",
    )
    p.add_argument("--once", action="store_true", help="Process one batch and exit")
    return p.parse_args()


def process_equity_bars_event(repo: EquityBarRepository, payload: dict) -> None:
    symbol = payload.get("symbol")
    bar_size = payload.get("bar_size")
    start = payload.get("start")
    end = payload.get("end")

    df = repo.load(symbol=symbol, bar_size=bar_size) if symbol else repo.load()
    if start:
        start_d = pd.to_datetime(start).date()
        df = df[df["trade_date"] >= start_d]
    if end:
        end_d = pd.to_datetime(end).date()
        df = df[df["trade_date"] <= end_d]

    # Example processing: print a small summary
    count = len(df)
    if count:
        tmin = df["trade_date"].min()
        tmax = df["trade_date"].max()
        print(f"[{symbol or 'ALL'} {bar_size or ''}] rows={count} range={tmin}..{tmax}")
    else:
        print(
            f"[{symbol or 'ALL'} {bar_size or ''}] no rows in repo for window {start}..{end}"
        )


def main() -> None:
    args = parse_args()
    consumer = JSONLConsumer(args.topic)
    processed = ProcessedIndex(service=args.service, topic=args.topic)
    repo = EquityBarRepository(args.base_path)

    try:
        while True:
            events = consumer.read(service=args.service, limit=args.limit)
            if not events:
                if args.once:
                    break
                # In a real daemon we would sleep here; keep example simple
                break
            for evt in events:
                payload = evt.get("payload", {})
                key = f"{payload.get('symbol')}:{payload.get('bar_size')}:{payload.get('start')}:{payload.get('end')}"
                if processed.is_processed(key):
                    continue
                process_equity_bars_event(repo, payload)
                processed.mark_processed(key)
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
