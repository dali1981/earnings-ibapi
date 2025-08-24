from __future__ import annotations

from datetime import datetime
from jobs.tasks import BackfillEquityBarsTask
from pathlib import Path

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--bar", default="1 day")
    parser.add_argument("--out", type=Path, default=Path("data/equity_bars"))
    parser.add_argument(
        "--demo", action="store_true", help="Write under data/exports/demo/equity_bars"
    )
    args = parser.parse_args()

    if args.demo:
        from common.utils.demo_mode import demo_base

        args.out = demo_base() / "equity_bars"

    task = BackfillEquityBarsTask(
        symbol=args.symbol,
        start=datetime.fromisoformat(args.start).date(),
        end=datetime.fromisoformat(args.end).date(),
        bar_size=args.bar,
        out=args.out,
    )
    task.run()
