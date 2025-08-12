from __future__ import annotations
import argparse
from pathlib import Path
from ibx_repos.contracts import ContractRepository
from ibx_flows.contracts_backfill import ContractsConfig, backfill_us_equity_contracts
try:
    from ibx import IBRuntime  # type: ignore
except Exception:
    from ibx.runtime import IBRuntime  # type: ignore


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/contracts"))
    p.add_argument("--exchanges", nargs="*", default=["NYSE","NASDAQ","AMEX","ARCA","BATS","IEX"])
    p.add_argument("--currency", default="USD")
    p.add_argument("--patterns", nargs="*", default=None)
    args = p.parse_args()
    cfg = ContractsConfig(exchanges=args.exchanges, currency=args.currency, patterns=args.patterns)
    repo = ContractRepository(args.root)
    with IBRuntime(port=4002) as rt:
        backfill_us_equity_contracts(rt, repo, cfg)
