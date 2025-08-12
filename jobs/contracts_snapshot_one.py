from ibx_flows.contracts_backfill import ContractsConfig, ContractDescriptionsSnapshotJob
from ibx_repos import ContractDescriptionsRepository
from ibx import IBRuntime


if __name__ == "__main__":
    cfg = ContractsConfig(patterns=["DEC", "CMCL", "NIU"])
    repo = ContractDescriptionsRepository("data/contract_descriptions")
    with IBRuntime(port=4002) as rt:
        job = ContractDescriptionsSnapshotJob(cfg, rt, repo)
        job.run()