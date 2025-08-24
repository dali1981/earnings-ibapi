#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence

from celery import group

from ibx_config.settings import get_settings
from ibx_celery.app import app as celery_app  # ensure app is importable
from ops.tasks.contracts import backfill_contracts_task


def _chunk(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def main() -> None:
    p = argparse.ArgumentParser(description="Launch Celery backfill jobs for US contracts")
    settings = get_settings()
    p.add_argument("--root", type=Path, default=settings.repo.contracts_path)
    p.add_argument(
        "--exchanges",
        nargs="*",
        default=["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"],
        help="Primary exchanges to include",
    )
    p.add_argument("--currency", default="USD")
    p.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Optional symbol search patterns; if omitted, worker will expand defaults",
    )
    p.add_argument("--chunk-size", type=int, default=200, help="Patterns per task")
    p.add_argument("--use-service", action="store_true", help="Reuse long-lived IB runtime per worker")
    p.add_argument("--via-api", action="store_true", help="Use FastAPI service instead of direct IB")
    p.add_argument("--api-base", default=os.environ.get("IB_API_BASE", "http://localhost:9010"), help="Base URL for IB API service")
    p.add_argument("--host", default=settings.ib.host)
    p.add_argument("--port", type=int, default=settings.ib.port)
    p.add_argument(
        "--client-id",
        type=int,
        default=settings.ib.client_id_contracts,
    )
    p.add_argument("--wait", action="store_true", help="Wait for all tasks to complete")
    args = p.parse_args()

    root = str(args.root)
    exchanges = [str(e).upper() for e in args.exchanges]
    patterns: List[str] | None = args.patterns

    # If patterns not provided, let worker expand defaults to avoid generating 17k signatures here
    if patterns is None:
        # Schedule a single task that will expand default patterns internally
        sig = backfill_contracts_task.s(
            root=root,
            exchanges=exchanges,
            currency=args.currency,
            patterns=None,
            use_service=bool(args.use_service) if not args.via_api else False,
            use_api=bool(args.via_api),
            api_base=args.api_base,
            ib_host=args.host,
            ib_port=int(args.port),
            client_id=int(args.client_id),
        )
        result = sig.apply_async()
        print(f"Scheduled contracts backfill task: {result.id}")
        if args.wait:
            result.get(disable_sync_subtasks=False)
        return

    # Chunk provided patterns and schedule a group for parallelism
    tasks = []
    for chunk in _chunk(patterns, args.chunk_size):
        tasks.append(
            backfill_contracts_task.s(
                root=root,
                exchanges=exchanges,
                currency=args.currency,
                patterns=chunk,
                use_service=bool(args.use_service) if not args.via_api else False,
                use_api=bool(args.via_api),
                api_base=args.api_base,
                ib_host=args.host,
                ib_port=int(args.port),
                client_id=int(args.client_id),
            )
        )

    job = group(tasks)
    result = job.apply_async()
    print(f"Scheduled {len(tasks)} contracts backfill tasks as group: {result.id}")
    if args.wait:
        result.join()


if __name__ == "__main__":
    # Ensure Celery picks up configuration if set via env vars when called via python
    os.environ.setdefault("CELERY_BROKER_URL", os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"))
    os.environ.setdefault(
        "CELERY_RESULT_BACKEND", os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    )
    main()
