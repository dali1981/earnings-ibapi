from __future__ import annotations

from datetime import datetime
from jobs.tasks import BackfillOptionBarsTask
from pathlib import Path

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--bar", default="1 day")
    parser.add_argument("--out", type=Path, default=Path("data/option_bars"))
    parser.add_argument("--expiries-max", type=int, default=8)
    parser.add_argument("--strikes-per-side", type=int, default=6)
    parser.add_argument(
        "--mode", choices=["k_around_atm", "moneyness_bands"], default="k_around_atm"
    )
    parser.add_argument(
        "--chains",
        type=Path,
        default=Path("data"),
        help="Base path for chain repository",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Write under data/exports/demo"
    )
    args = parser.parse_args()

    if args.demo:
        from common.utils.demo_mode import demo_base

        demo = demo_base()
        args.out = demo / "option_bars"
        args.chains = demo

    task = BackfillOptionBarsTask(
        symbol=args.symbol,
        start=datetime.fromisoformat(args.start).date(),
        end=datetime.fromisoformat(args.end).date(),
        bar_size=args.bar,
        out=args.out,
        chain_base=args.chains,
        expiries_max=args.expiries_max,
        strikes_per_side=args.strikes_per_side,
        selection_mode=args.mode,
    )
    task.run()
