from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from ibx_repos.equity_bars import EquityBarRepository
from ibx_repos.option_bars import OptionBarRepository
from ibx_repos.option_chains import OptionChainSnapshotRepository
from ibx_flows.windows import missing_windows
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dry-run coverage check (no IB calls)")
    sub = p.add_subparsers(dest="mode", required=True)
    pe = sub.add_parser("equity")
    pe.add_argument("symbol"); pe.add_argument("start"); pe.add_argument("end")
    pe.add_argument("--bar", default="1 day"); pe.add_argument("--root", type=Path, default=Path("data/equity_bars"))
    po = sub.add_parser("options")
    po.add_argument("symbol"); po.add_argument("start"); po.add_argument("end")
    po.add_argument("--bar", default="1 day"); po.add_argument("--root", type=Path, default=Path("data/option_bars"))
    po.add_argument("--from-chain-root", type=Path, default=None); po.add_argument("--expiry"); po.add_argument("--right", choices=["C","P"]); po.add_argument("--strike", type=float)
    args = p.parse_args()
    if args.mode == "equity":
        repo = EquityBarRepository(args.root)
        present = repo.present_dates(args.symbol, args.bar, pd.to_datetime(args.start), pd.to_datetime(args.end))
        gaps = missing_windows(present, pd.to_datetime(args.start).date(), pd.to_datetime(args.end).date())
        print({"symbol": args.symbol, "bar_size": args.bar, "gaps": gaps, "present_days": len(present)})
    else:
        repo = OptionBarRepository(args.root)
        start = pd.to_datetime(args.start).date(); end = pd.to_datetime(args.end).date()
        contracts = []
        if args.from_chain_root:
            chain_repo = OptionChainSnapshotRepository(args.from_chain_root)
            chain = chain_repo.load(args.symbol, asof_start=args.end, asof_end=args.end)
            if args.expiry: chain = chain[chain["expiry"] == pd.to_datetime(args.expiry).date()]
            if args.right: chain = chain[chain["right"] == args.right]
            if args.strike is not None: chain = chain[abs(chain["strike"].astype(float) - float(args.strike)) < 1e-9]
            if hasattr(chain, "to_dict"):
                contracts = chain[["underlying","expiry","right","strike"]].drop_duplicates().to_dict("records")
        else:
            df = repo.load(args.symbol, args.bar)
            if not df.empty:
                if args.expiry: df = df[df["expiry"] == pd.to_datetime(args.expiry).date()]
                if args.right: df = df[df["right"] == args.right]
                if args.strike is not None: df = df[abs(df["strike"].astype(float) - float(args.strike)) < 1e-9]
                contracts = df[["underlying","expiry","right","strike"]].drop_duplicates().to_dict("records")
        out = []
        for c in contracts:
            present = repo.present_dates_for_contract(c["underlying"], c["expiry"], c["right"], float(c["strike"]), args.bar, pd.to_datetime(args.start), pd.to_datetime(args.end))
            gaps = missing_windows(present, start, end)
            out.append({**c, "gaps": gaps, "present_days": len(present)})
        print(out)
