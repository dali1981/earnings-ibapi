from __future__ import annotations
import argparse
from pathlib import Path
from ibx_repos.option_bars import OptionBarRepository
from ibx_flows.resample import resample_option_minute_to_daily
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("symbol"); p.add_argument("start"); p.add_argument("end")
    p.add_argument("--in-root", type=Path, default=Path("data/option_bars"))
    p.add_argument("--out-root", type=Path, default=Path("data/option_bars"))
    p.add_argument("--expiry"); p.add_argument("--right", choices=["C","P"]); p.add_argument("--strike", type=float)
    args = p.parse_args()
    in_repo = OptionBarRepository(args.in_root); out_repo = OptionBarRepository(args.out_root)
    df = in_repo.load(underlying=args.symbol, bar_size="1 min", expiry=args.expiry, right=args.right, strike=args.strike, start=args.start, end=args.end)
    daily = resample_option_minute_to_daily(df)
    if not daily.empty: out_repo.save(daily)
