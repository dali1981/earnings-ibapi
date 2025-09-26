import warnings


warnings.warn(
    "contracts_snapshot_one.py is deprecated. Use jobs/contracts_snapshot.py --patterns instead.",
    DeprecationWarning,
    stacklevel=2,
)

from common.repositories.contract_descriptions import ContractDescriptionsRepository
from ibx import IBRuntime
from ibx_flows.contracts_backfill import (
    ContractDescriptionsSnapshotJob,
    ContractsConfig,
)


if __name__ == "__main__":
    cfg = ContractsConfig(patterns=["DEC", "CMCL", "NIU"])
    repo = ContractDescriptionsRepository("data/contract_descriptions")
    with IBRuntime(port=4002) as rt:
        job = ContractDescriptionsSnapshotJob(cfg, rt, repo)
        job.run()
