"""
Utilities package for trading project.

Provides common utilities like logging setup, configuration helpers, etc.
Also exposes light-weight IO helpers expected by callers (e.g., append_parquet).
"""

from .logging_setup import (
    setup_logging,
    get_logger,
    setup_script_logging,
    disable_noisy_loggers,
)

# Minimal IO helper re-exposed here to avoid import confusion between the
# top-level `utils.py` module and this `utils` package.
from typing import List, Dict, Any
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional in some environments
    pa = None
    pq = None


def append_parquet(path: Path, records: List[Dict[str, Any]]):
    """Append records to a parquet file (create if missing).

    This mirrors the implementation in the root-level `utils.py` to provide a
    stable import path: `from utils import append_parquet`.
    """
    if not records:
        return
    if pa is None or pq is None:
        raise RuntimeError("pyarrow not available for append_parquet")
    table = pa.Table.from_pylist(records)
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, path)


__all__ = [
    'setup_logging',
    'get_logger',
    'setup_script_logging',
    'disable_noisy_loggers',
    'append_parquet',
]
