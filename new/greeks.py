# ibx_repos/greeks.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

class _BaseGreeksRepo:
    def __init__(self, root: Path, separate_subdirs: bool = True):
        self.root = Path(root)
        self.separate_subdirs = separate_subdirs
        if separate_subdirs:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> str:
        return str(self.root)

    @staticmethod
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        float_cols = ['S','opt_price','r','q','T','iv','delta','gamma','vega','theta','rho']
        for c in float_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def _write(self, df: pd.DataFrame):
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=self._path(),
            partition_cols=["underlying","expiry","right","strike"],
            compression="snappy",
            use_dictionary=True
        )

class CalculatedGreeksRepo(_BaseGreeksRepo):
    \"\"Greeks computed from prices via model\"\"
    def __init__(self, root: Path, separate_subdirs: bool = True):
        sub = Path(root) / \"calculated\" if separate_subdirs else Path(root)
        super().__init__(sub, separate_subdirs=False)

    def save(self, df: pd.DataFrame):
        df = self._coerce(df)
        df['source'] = 'calculated'
        self._write(df)

class IbGreeksRepo(_BaseGreeksRepo):
    \"\"Greeks obtained from IB market data\"\"
    def __init__(self, root: Path, separate_subdirs: bool = True):
        sub = Path(root) / \"ib\" if separate_subdirs else Path(root)
        super().__init__(sub, separate_subdirs=False)

    def save(self, df: pd.DataFrame):
        df = self._coerce(df)
        df['source'] = 'ib'
        self._write(df)