from common.repositories.contract_descriptions import ContractDescriptionsRepository
from ibx import IBRuntime
from ibx_flows.contracts_backfill import (
    ContractDescriptionsSnapshotJob,
    ContractsConfig,
)

import argparse


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Snapshot contract descriptions (option support)"
    )
    p.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Symbol patterns or exact symbols to include (e.g., AAPL NVDA)",
    )
    p.add_argument(
        "--out",
        default="data/contract_descriptions",
        help="Output path for contract descriptions",
    )
    p.add_argument("--port", type=int, default=4002)
    args = p.parse_args()

    cfg = ContractsConfig(patterns=args.patterns)
    repo = ContractDescriptionsRepository(args.out)
    with IBRuntime(port=args.port) as rt:
        job = ContractDescriptionsSnapshotJob(cfg, rt, repo)
        job.run()
