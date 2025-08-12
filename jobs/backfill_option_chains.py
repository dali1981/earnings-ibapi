from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
from ibx_repos.option_chains import OptionChainSnapshotRepository
from ibx_flows.backfill import BackfillConfig, backfill_option_chain_daily
from ibx_flows.source_ib import IBSource
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("symbol"); p.add_argument("start"); p.add_argument("end"); p.add_argument("--out", type=Path, default=Path("data/option_chains")); args = p.parse_args()
    cfg = BackfillConfig(underlying=args.symbol, start=datetime.fromisoformat(args.start).date(), end=datetime.fromisoformat(args.end).date())
    repo = OptionChainSnapshotRepository(args.out); src = IBSource(); backfill_option_chain_daily(src, repo, cfg)
