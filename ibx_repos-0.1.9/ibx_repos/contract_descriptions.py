
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple
import pandas as pd

class ContractDescriptionsRepository:
    REQUIRED = [
        "conid", "symbol", "sec_type", "currency",
        "primary_exchange", "derivative_sec_types", "as_of_date"
    ]

    def __init__(self, base_path: str | Path):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self._ensure_initialized()

    # ---------------- Schema & init ----------------
    @staticmethod
    def _schema():
        import pyarrow as pa
        return pa.schema([
            ("conid", pa.int64()),
            ("symbol", pa.string()),
            ("sec_type", pa.string()),
            ("currency", pa.string()),
            ("primary_exchange", pa.string()),
            ("derivative_sec_types", pa.list_(pa.string())),
            ("as_of_date", pa.date32()),
        ])

    def _ensure_initialized(self) -> None:
        import pyarrow as pa, pyarrow.parquet as pq
        if any(self.base.rglob("*.parquet")):
            return
        schema = self._schema()
        empty_cols = {f.name: pa.array([], type=f.type) for f in schema}
        empty_tbl = pa.Table.from_pydict(empty_cols, schema=schema)
        pq.write_table(empty_tbl, self.base / "_empty.parquet")

    # ---------------- Internal helpers ----------------
    def _existing_keys(self, dates: Sequence[dt.date], conids: Sequence[int]) -> Set[Tuple[int, dt.date]]:
        """Return set of (query, conid, as_of_date) already in dataset for the given dates and conids."""
        import pyarrow.dataset as ds
        if not dates or not conids:
            return set()
        if not self.base.exists() or not any(self.base.rglob("*.parquet")):
            return set()

        # Build dataset and filter
        dset = ds.dataset(self.base, format="parquet", partitioning="hive", schema=self._schema())

        from pyarrow.dataset import field
        expr = (field("as_of_date").isin(list(dates))) & (field("conid").isin(list(conids)))
        cols = ["conid", "as_of_date"]
        tbl = dset.to_table(filter=expr, columns=cols)
        if tbl.num_rows == 0:
            return set()
        pdf = tbl.to_pandas()
        # Normalize types
        pdf["conid"] = pd.to_numeric(pdf["conid"], errors="coerce").astype("Int64")
        pdf["as_of_date"] = pd.to_datetime(pdf["as_of_date"]).dt.date
        return set(zip(pdf["conid"].astype("Int64"), pdf["as_of_date"]))

    # ---------------- Save ----------------
    def save(self, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            return
        import pyarrow as pa, pyarrow.parquet as pq

        df['as_of_date'] = dt.date.today()

        df = df.copy()
        for col in self.REQUIRED:
            if col not in df.columns:
                df[col] = None

        df["conid"] = pd.to_numeric(df["conid"], errors="coerce").astype("Int64")
        for col in ["symbol", "sec_type", "currency", "primary_exchange"]:
            df[col] = df[col].fillna("").astype(str).str.upper()

        def _to_list_str(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            if isinstance(x, str):
                return [s.strip().upper() for s in x.split(",") if s.strip()]
            if isinstance(x, (list, tuple, set)):
                return [str(s).upper() for s in x]
            return [str(x).upper()]
        df["derivative_sec_types"] = df["derivative_sec_types"].map(_to_list_str)

        def _to_date(v):
            if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
                return v
            if pd.isna(v):
                return dt.date.today()
            return pd.to_datetime(v).date()
        df["as_of_date"] = df["as_of_date"].map(_to_date)

        def _first_char(row):
            base = str(row.get("symbol") or row.get("query") or "").strip().upper()
            return base[0] if base else "#"
        df["first_char"] = df.apply(_first_char, axis=1)

        # Dedup within-batch
        df = df.dropna(subset=["conid", "as_of_date"])
        df["conid"] = df["conid"].astype("Int64")
        df = df.drop_duplicates(subset=["conid", "as_of_date"], keep="last")

        # ---- Append-only: drop rows whose (query, conid, as_of_date) already exist
        u_dates = sorted(set(df["as_of_date"].tolist()))
        u_conids = sorted({int(x) for x in df["conid"].dropna().tolist()})
        exists = self._existing_keys(u_dates, u_conids)
        if exists:
            mask = ~df.apply(lambda r: (int(r["conid"]), r["as_of_date"]) in exists, axis=1)
            df = df.loc[mask]
        if df.empty:
            return

        schema = self._schema().append(pa.field("first_char", pa.string()))
        arrays = {}
        for f in schema:
            col = f.name
            if col == "derivative_sec_types":
                arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.string()))
            elif col == "as_of_date":
                arrays[col] = pa.array(pd.to_datetime(df[col]).values.astype("datetime64[D]"))
            elif col == "conid":
                arrays[col] = pa.array(df[col].astype("Int64").tolist(), type=pa.int64())
            else:
                arrays[col] = pa.array(df[col].astype(object).where(pd.notna(df[col]), None).tolist(), type=f.type)
        table = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))

        pq.write_to_dataset(
            table,
            root_path=str(self.base),
            partition_cols=["as_of_date", "first_char"],
            existing_data_behavior="overwrite_or_ignore",
            compression="snappy",
            use_dictionary=True,
        )

    # ---------------- Read ----------------
    def read(self, columns: Optional[Sequence[str]] = None, **filters) -> pd.DataFrame:
        import pyarrow.dataset as ds
        if not self.base.exists() or not any(self.base.rglob("*.parquet")):
            cols = list(columns) if columns else [f.name for f in self._schema()]
            return pd.DataFrame(columns=cols)

        dset = ds.dataset(self.base, format="parquet", partitioning="hive", schema=self._schema())
        schema = dset.schema
        from pyarrow.dataset import field
        expr = None
        for k, v in filters.items():
            if k not in schema.names:
                continue
            cond = field(k).isin(list(v)) if isinstance(v, (list, tuple, set)) else (field(k) == v)
            expr = cond if expr is None else (expr & cond)
        cols = [c for c in (columns or schema.names) if c in schema.names]
        tbl = dset.to_table(filter=expr, columns=cols)
        return tbl.to_pandas() if tbl.num_rows else pd.DataFrame(columns=cols)
