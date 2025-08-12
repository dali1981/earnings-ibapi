from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
from ibx_repos.equity_bars import EquityBarRepository
from ibx_flows.backfill import BackfillConfig, backfill_equity_bars
from ibx_flows.source_ib import IBSource


if __name__ == "__main__":
    p = argparse.ArgumentParser();
    p.add_argument("symbol")
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("--bar", default="1 day");
    p.add_argument("--out", type=Path, default=Path("data/equity_bars"))
    args = p.parse_args()
    cfg = BackfillConfig(underlying=args.symbol, start=datetime.fromisoformat(args.start).date(), end=datetime.fromisoformat(args.end).date(), bar_size=args.bar)
    repo = EquityBarRepository(args.out); src = IBSource(); backfill_equity_bars(src, repo, cfg)
