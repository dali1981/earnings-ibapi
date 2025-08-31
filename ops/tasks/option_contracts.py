from __future__ import annotations

from celery import shared_task
from typing import Any, Dict

# Thin wrappers around option_bars tasks to present a cohesive API surface
from ops.tasks.option_bars import (
    get_option_contracts_for_bars_task as _select_from_chains,
    fetch_option_contract_details_task as _enrich_details,
    save_selected_option_contracts_to_repo_task as _save_contracts,
)


@shared_task(name="option_contracts.select", queue="option_chains")
def select_option_contracts_task(**kwargs) -> Dict[str, Any]:
    """Select option contracts from aggregated chain snapshots.

    Wrapper that forwards keyword args to get_option_contracts_for_bars_task.
    """
    return _select_from_chains.run(_select_from_chains, **kwargs)


@shared_task(name="option_contracts.enrich", queue="option_chains")
def enrich_option_contracts_task(contracts_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Optionally enrich selected contracts with conId/localSymbol via IB service.

    Not called automatically; orchestrator can trigger explicitly.
    """
    return _enrich_details.run(_enrich_details, contracts_result=contracts_result, **kwargs)


@shared_task(name="option_contracts.save", queue="option_chains")
def save_option_contracts_task(contracts_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Persist option contracts that already have conId into the contracts repo."""
    return _save_contracts.run(_save_contracts, contracts_result=contracts_result, **kwargs)

