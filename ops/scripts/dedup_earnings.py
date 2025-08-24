#!/usr/bin/env python3
"""
Deduplicate earnings repository partitions in-place.

Scans earnings partitions under the selected base path, merges all files per
`collection_date=YYYY-MM-DD/source=<src>` into a single deterministic
`earnings.parquet` with duplicates removed by key (symbol, earnings_date, source).

Usage:
  # Dedup main repo
  python scripts/dedup_earnings.py

  # Dry-run (report only)
  python scripts/dedup_earnings.py --dry-run

  # Dedup demo repo
  python scripts/dedup_earnings.py --demo

  # Dedup a specific collection date only
  python scripts/dedup_earnings.py --date 2025-08-15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import pandas as pd

# Ensure project root on path so `config` is importable when run from subfolders
sys.path.append(str(Path(__file__).parent.parent))
import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate earnings repository partitions")
    p.add_argument("--demo", action="store_true", help="Use demo base (data/exports/demo)")
    p.add_argument(
        "--date",
        type=str,
        help="Specific collection_date (YYYY-MM-DD) to process; default: all",
    )
    p.add_argument("--dry-run", action="store_true", help="Report actions without writing")
    return p.parse_args()


def list_partitions(base: Path, only_date: str | None) -> List[Tuple[Path, Path]]:
    parts: List[Tuple[Path, Path]] = []
    earnings_dir = base / "earnings"
    if not earnings_dir.exists():
        return []
    for coll_dir in sorted(earnings_dir.glob("collection_date=*")):
        if not coll_dir.is_dir():
            continue
        if only_date and coll_dir.name != f"collection_date={only_date}":
            continue
        for src_dir in sorted(coll_dir.glob("source=*")):
            if src_dir.is_dir():
                parts.append((coll_dir, src_dir))
    return parts


def dedup_partition(src_dir: Path, dry_run: bool) -> Tuple[int, int, int]:
    files = list(src_dir.glob("*.parquet"))
    if not files:
        return 0, 0, 0
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:
            print(f"WARN: could not read {f}: {e}")
    if not frames:
        return 0, 0, 0
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    key_cols = ["symbol", "earnings_date", "source"]
    for k in key_cols:
        if k not in combined.columns:
            # unexpected; keep as-is
            after = before
            break
    else:
        after_df = combined.drop_duplicates(subset=key_cols)
        after = len(after_df)
        combined = after_df

    if not dry_run:
        # Clean old files then write deterministic file
        for f in files:
            try:
                f.unlink()
            except Exception as e:
                print(f"WARN: could not remove {f}: {e}")
        out = src_dir / "earnings.parquet"
        combined.to_parquet(out)
    return len(files), before, after


def main() -> None:
    args = parse_args()
    base = (config.EXPORTS_PATH / "demo") if args.demo else config.DATA_ROOT
    print(f"Scanning earnings under: {base / 'earnings'} {'[DRY-RUN]' if args.dry_run else ''}")

    parts = list_partitions(base, args.date)
    if not parts:
        print("No partitions found.")
        return
    total_files = total_before = total_after = 0
    for coll_dir, src_dir in parts:
        files, before, after = dedup_partition(src_dir, args.dry_run)
        total_files += files
        total_before += before
        total_after += after
        date_str = coll_dir.name.replace("collection_date=", "")
        src_str = src_dir.name.replace("source=", "")
        print(
            f"{date_str} / {src_str}: files={files} rows_before={before} rows_after={after} removed={before - after}"
        )

    print("-" * 60)
    print(
        f"Total: files={total_files} rows_before={total_before} rows_after={total_after} removed={total_before - total_after}"
    )


if __name__ == "__main__":
    main()
