from __future__ import annotations

"""
Repair script for option_contracts dataset:
- Finds rows saved under expiry=__HIVE_DEFAULT_PARTITION__ (i.e., missing expiry)
- Derives expiry, right, strike from local_symbol/expiry_raw where possible
- Rewrites corrected rows into proper partitions via OptionContractsRepository

Run: python scripts/repair_option_contracts_partitions.py
"""

import re
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds

from app_settings.settings import get_settings
from common.repositories.option_contracts import OptionContractsRepository


def parse_from_local_symbol(local_symbol: str):
    if not isinstance(local_symbol, str):
        return None, None
    m = re.search(r"(\d{6})([CP])(\d{8})", local_symbol)
    if not m:
        return None, None
    yymmdd = m.group(1)
    right = m.group(2)
    strike = None
    try:
        strike = int(m.group(3)) / 1000.0
    except Exception:
        pass
    try:
        expiry = pd.to_datetime("20" + yymmdd, format="%Y%m%d").date()
    except Exception:
        expiry = None
    return expiry, right, strike


def main():
    settings = get_settings()
    base = settings.paths.data_root
    droot = Path(base) / "option_contracts"
    if not droot.exists():
        print(f"option_contracts dataset not found at {droot}")
        return

    dataset = ds.dataset(str(droot), format="parquet", partitioning="hive")
    tbl = dataset.to_table()
    if tbl.num_rows == 0:
        print("option_contracts is empty")
        return

    df = tbl.to_pandas()
    # Identify missing expiry
    mask = df.get("expiry").isna() if "expiry" in df.columns else pd.Series([], dtype=bool)
    if not mask.any():
        print("No rows with missing expiry found")
        return

    fix = df[mask].copy()
    # Derive from expiry_raw
    def norm_exp(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        try:
            token = s.split()[0]
            if len(token) == 8 and token.isdigit():
                return pd.to_datetime(token, format="%Y%m%d").date()
            return pd.to_datetime(token).date()
        except Exception:
            return None

    if "expiry_raw" in fix.columns:
        fix.loc[:, "expiry"] = fix["expiry_raw"].map(norm_exp)

    # Fallback to local_symbol
    if "local_symbol" in fix.columns:
        for i, row in fix.iterrows():
            if pd.isna(row.get("expiry")):
                e, r, k = parse_from_local_symbol(row.get("local_symbol"))
                if e:
                    fix.at[i, "expiry"] = e
                if (pd.isna(row.get("right")) or str(row.get("right")).strip().lower() in ("", "none", "nan")) and r:
                    fix.at[i, "right"] = r
                if pd.isna(row.get("strike")) and k is not None:
                    fix.at[i, "strike"] = k

    # Drop rows still missing expiry
    fix = fix.dropna(subset=["expiry"])
    if fix.empty:
        print("No fixable rows found (missing expiry in all)")
        return

    # Add timestamps/partition date
    if "created_at" not in fix.columns or fix["created_at"].isna().all():
        fix["created_at"] = pd.Timestamp.utcnow(tz="UTC")
    fix["snapshot"] = pd.to_datetime(fix["created_at"], utc=True).dt.tz_convert("UTC").dt.date

    # Deduplicate by conid + expiry + right
    if "conid" in fix.columns:
        fix = fix.drop_duplicates(subset=[c for c in ["conid", "expiry", "right"] if c in fix.columns])

    repo = OptionContractsRepository(base)
    repo.save(fix)
    print(f"Repaired and saved {len(fix)} rows into corrected partitions")


if __name__ == "__main__":
    main()

