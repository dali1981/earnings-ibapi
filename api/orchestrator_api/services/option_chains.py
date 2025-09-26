from __future__ import annotations

from celery import chain
from celery.canvas import Signature
from typing import Optional


def build_snapshot_chain(
    symbol: str,
    snapshot_date: Optional[str] = None,
    root: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Signature:
    """Construct the Celery canvas for option chain snapshot collection.

    Pipeline:
      1) check_has_options (repo-only)
      2) ensure_has_options (gate)
      3) get_underlying_contract (repo-only)
      4) fetch_option_params (secdef)
      5) build_chain_snapshot (aggregate params into rows)
      6) save_chain_snapshot (persist to option_chains repo)
    """
    from ops.tasks.option_chains import (
        build_chain_snapshot_task,
        check_has_options_task,
        fetch_option_params_task,
        get_underlying_contract_task,
        save_chain_snapshot_task,
    )

    return (
        check_has_options_task.si(
            symbol=symbol, root=root, correlation_id=correlation_id
        )
        | get_underlying_contract_task.si(
            symbol=symbol, root=root, correlation_id=correlation_id
        )
        | fetch_option_params_task.s(correlation_id=correlation_id)
        | build_chain_snapshot_task.s(snapshot_date)
        | save_chain_snapshot_task.s(root=root)
    )
