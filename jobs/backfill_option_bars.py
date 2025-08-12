from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
from ibx_repos.option_bars import OptionBarRepository
from ibx_flows.backfill import BackfillConfig, backfill_option_bars
from ibx_flows.source_ib import IBSource
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("symbol"); p.add_argument("start"); p.add_argument("end")
    p.add_argument("--bar", default="1 day"); p.add_argument("--out", type=Path, default=Path("data/option_bars"))
    p.add_argument("--expiries-max", type=int, default=8); p.add_argument("--strikes-per-side", type=int, default=6)
    p.add_argument("--mode", choices=["k_around_atm","moneyness_bands"], default="k_around_atm")
    args = p.parse_args()
    cfg = BackfillConfig(underlying=args.symbol, start=datetime.fromisoformat(args.start).date(), end=datetime.fromisoformat(args.end).date(), bar_size=args.bar, expiries_max=args.expiries_max, strikes_per_side=args.strikes_per_side, selection_mode=args.mode)
    repo = OptionBarRepository(args.out); src = IBSource(); backfill_option_bars(src, repo, chain_repo=None, eq_repo=None, cfg=cfg)
