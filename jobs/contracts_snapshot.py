from ibx import IBRuntime
from ibx_flows.contracts_backfill import (
    ContractDescriptionsSnapshotJob,
    ContractsConfig,
)
from ibx_repos import ContractDescriptionsRepository


if __name__ == "__main__":
    cfg = ContractsConfig()
    repo = ContractDescriptionsRepository("data/contract_descriptions")
    with IBRuntime(port=4002) as rt:
        job = ContractDescriptionsSnapshotJob(cfg, rt, repo)
        job.run()
