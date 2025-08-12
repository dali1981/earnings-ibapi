from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from jobs.tasks import BackfillEquityBarsTask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--bar", default="1 day")
    parser.add_argument("--out", type=Path, default=Path("data/equity_bars"))
    args = parser.parse_args()

    task = BackfillEquityBarsTask(
        symbol=args.symbol,
        start=datetime.fromisoformat(args.start).date(),
        end=datetime.fromisoformat(args.end).date(),
        bar_size=args.bar,
        out=args.out,
    )
    task.run()
