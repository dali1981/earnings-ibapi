from __future__ import annotations
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_table(df: pd.DataFrame, schema: pa.schema | None) -> pa.Table:
    if schema is None:
        return pa.Table.from_pandas(df, preserve_index=False)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def write_dataset(df: pd.DataFrame, root_path: Path, schema: pa.schema | None, partition_cols: list[str]) -> None:
    _ensure_dir(root_path)
    table = _to_table(df, schema)
    pq.write_to_dataset(
        table,
        root_path=str(root_path),
        partition_cols=partition_cols,
        use_dictionary=True,
        compression="snappy",
    )


def _dataset_hive(root_path: Path) -> ds.Dataset:
    return ds.dataset(str(root_path), format="parquet", partitioning="hive")


def _build_filter(**kwargs):
    """
    Build a pyarrow.dataset filter expression from keyword equality tests.
    Accepts scalars or iterables (treated as isin). For dates, accepts str or date.
    """
    expr = None
    for key, value in kwargs.items():
        if value is None:
            continue
        field = ds.field(key)
        if isinstance(value, (list, tuple, set)):
            term = field.isin(list(value))
        else:
            if key in {"asof","date"}:
                try:
                    value = pd.to_datetime(value).date()
                except Exception:
                    pass
            term = field == value
        expr = term if expr is None else (expr & term)
    return expr


def read_dataset(root_path: Path, columns: list[str] | None = None, **filters) -> pd.DataFrame:
    ds_ = _dataset_hive(root_path)
    expr = _build_filter(**filters)
    tbl = ds_.to_table(filter=expr, columns=columns)
    return tbl.to_pandas() if tbl.num_rows else pd.DataFrame(columns=columns or [])


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_table(df: pd.DataFrame, schema: pa.schema | None) -> pa.Table:
    if schema is None:
        return pa.Table.from_pandas(df, preserve_index=False)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def write_dataset(df: pd.DataFrame, root_path: Path, schema: pa.schema | None, partition_cols: list[str]) -> None:
    _ensure_dir(root_path)
    table = _to_table(df, schema)
    pq.write_to_dataset(
        table,
        root_path=str(root_path),
        partition_cols=partition_cols,
        use_dictionary=True,
        compression="snappy",
    )
