#!/usr/bin/env python3
"""
Earnings Data Repository

Persistent storage for earnings calendar data using PyArrow/Parquet format
with partitioning by date and data lineage tracking.

Features:
- Daily partitioned storage for efficient querying
- Historical data retention with configurable policies
- Data lineage tracking for audit trails
- Deduplication and data quality validation
- Fast columnar queries using PyArrow
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys


sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import configuration first
from config import EARNINGS_CONFIG, EARNINGS_PATH


# Avoid runtime import cycle: only import for type checking
if TYPE_CHECKING:
    from earnings.fetcher import EarningsEvent

from common.instrumentation import get_trading_logger as get_logger
from common.repositories.base import BaseRepository


logger = get_logger(__name__)


@dataclass
class EarningsStorageMetadata:
    """Metadata for earnings data storage."""

    collection_date: date
    source: str
    total_events: int
    date_range_start: date
    date_range_end: date
    unique_symbols: int
    data_quality_score: float
    processing_time_seconds: float


class EarningsCalendarRepository(BaseRepository):
    """Repository for persistent earnings calendar data storage."""

    def __init__(self, data_path: Path = None):
        """
        Initialize earnings repository with configured paths and settings.

        Args:
            data_path: Base data path (uses configured EARNINGS_PATH if None)
        """

        # Use configured path if not provided
        if data_path is None:
            data_path = EARNINGS_PATH.parent
            logger.info(f"📁 Using configured earnings path: {EARNINGS_PATH}")
        else:
            logger.info(f"📁 Using provided data path: {data_path}")

        super().__init__(data_path, "earnings")
        self.data_path = self.dataset_path  # Use the dataset path from BaseRepository
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Partitioning strategy: collection_date=YYYY-MM-DD/source=nasdaq/
        self.partition_cols = ["collection_date", "source"]

        # Use configured retention policy
        self.retention_days = EARNINGS_CONFIG.get("retention_days", 365)

        logger.info("📊 Initialized earnings repository with CONFIGURED settings:")
        logger.info(f"   Data path: {self.data_path}")
        logger.info(f"   Retention: {self.retention_days} days")
        logger.info(f"   Partitioning: {self.partition_cols}")

    # Disable generic BaseRepository.save() for earnings to avoid accidental
    # parquet dataset overwrites. Use store_earnings_batch() which handles
    # partitioning and deduping safely within a collection_date/source.
    def save(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "Use store_earnings_batch() for earnings; generic save() is disabled."
        )

    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for earnings data."""
        return pa.schema(
            [
                pa.field("symbol", pa.string()),
                pa.field("company_name", pa.string()),
                pa.field("earnings_date", pa.string()),  # Stored as ISO string
                pa.field("period_ending", pa.string()),  # Required field for quarterly periods
                pa.field("time", pa.string()),
                pa.field("eps_estimate", pa.float64()),
                pa.field("eps_actual", pa.float64()),
                pa.field("revenue_estimate", pa.float64()),
                pa.field("revenue_actual", pa.float64()),
                pa.field("source", pa.string()),
                pa.field("collection_date", pa.string()),  # Stored as ISO string
                pa.field("days_until_earnings", pa.int32()),
                pa.field("priority_score", pa.float64()),
                pa.field("market_cap", pa.float64()),  # Optional field
            ]
        )

    def _get_partition_columns(self) -> List[str]:
        """Return partition column names."""
        return self.partition_cols

    def _normalize_data(self, data: Any) -> pd.DataFrame:
        """Normalize earnings data to DataFrame format."""
        if isinstance(data, list):
            # List of EarningsEvent objects
            return self._earnings_to_dataframe(data, date.today())
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def store_earnings_batch(
        self,
        earnings: List["EarningsEvent"],
        collection_date: date = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Store a batch of earnings events with automatic partitioning and lineage tracking.

        Args:
            earnings: List of earnings events to store
            collection_date: Date when data was collected (defaults to today)
            metadata: Additional metadata to store with the batch

        Returns:
            Dictionary with storage results and metadata
        """
        if collection_date is None:
            collection_date = date.today()

        if not earnings:
            logger.warning("No earnings events to store")
            return {"events_stored": 0}

        start_time = datetime.now()

        # Convert earnings events to DataFrame
        df = self._earnings_to_dataframe(earnings, collection_date)

        # Calculate storage metadata
        storage_metadata = self._calculate_storage_metadata(
            earnings, collection_date, start_time
        )

        # Group by source for separate storage
        results = {}
        for source in df["source"].unique():
            source_df = df[df["source"] == source].copy()

            # Store data with partitioning
            partition_path = (
                self.data_path
                / f"collection_date={collection_date.isoformat()}"
                / f"source={source}"
            )
            partition_path.mkdir(parents=True, exist_ok=True)

            # Load existing partition files, dedupe, and write a deterministic single file
            frames = [source_df]
            existing_files = list(partition_path.glob("*.parquet"))
            for f in existing_files:
                try:
                    frames.append(pd.read_parquet(f))
                except Exception:
                    logger.warning(f"Could not read existing parquet: {f}")
            combined = pd.concat(frames, ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(
                subset=["symbol", "earnings_date", "source"]
            )  # idempotent key
            after = len(combined)

            # Overwrite partition atomically: remove old files then write one consolidated file
            for f in existing_files:
                try:
                    f.unlink()
                except Exception:
                    logger.warning(f"Could not remove file during overwrite: {f}")

            file_path = partition_path / "earnings.parquet"
            table = pa.Table.from_pandas(combined)
            pq.write_table(table, file_path)

            results[source] = {
                "file_path": str(file_path),
                "events_count": int(len(source_df)),
                "file_size_bytes": file_path.stat().st_size,
                "dedup_removed": int(before - after),
            }

            logger.info(
                f"💾 Stored {len(source_df)} {source} earnings events to {file_path} (dedup removed={before - after})"
            )

        # Store metadata
        self._store_metadata(collection_date, storage_metadata, metadata)

        # Cleanup old data if needed
        self._cleanup_old_data()

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "collection_date": collection_date.isoformat(),
            "total_events_stored": len(earnings),
            "sources": results,
            "processing_time_seconds": processing_time,
            "metadata": asdict(storage_metadata),
        }

    def get_earnings_by_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve earnings data for a specific date range.

        Args:
            start_date: Start of earnings date range
            end_date: End of earnings date range
            symbols: Optional list of symbols to filter
            sources: Optional list of data sources to include

        Returns:
            DataFrame with earnings events
        """
        logger.info(f"🔍 Querying earnings from {start_date} to {end_date}")

        # Build list of partition paths to scan
        partition_filters = []
        current_date = start_date

        while current_date <= end_date:
            date_filter = ("earnings_date", ">=", current_date.isoformat())
            partition_filters.append(date_filter)
            current_date += timedelta(days=1)

        try:
            # Manually iterate partitions to avoid schema merge issues
            frames: List[pd.DataFrame] = []
            for coll_dir in sorted(self.data_path.glob("collection_date=*")):
                if not coll_dir.is_dir():
                    continue
                for src_dir in sorted(coll_dir.glob("source=*")):
                    if not src_dir.is_dir():
                        continue
                    if sources:
                        src_name = src_dir.name.replace("source=", "")
                        if src_name not in set(sources):
                            continue
                    for pqf in src_dir.glob("*.parquet"):
                        try:
                            dfp = pd.read_parquet(pqf)
                            frames.append(dfp)
                        except Exception as read_err:
                            logger.warning(f"Could not read parquet {pqf}: {read_err}")

            if not frames:
                logger.warning(
                    f"No earnings data found for date range {start_date} to {end_date}"
                )
                return pd.DataFrame()

            df = pd.concat(frames, ignore_index=True)

            # Convert date strings back to dates
            df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
            df["collection_date"] = pd.to_datetime(
                df.get("collection_date", None)
            ).dt.date

            # Date filtering
            df = df[
                (df["earnings_date"] >= start_date) & (df["earnings_date"] <= end_date)
            ]

            # Deduplicate across combined frames to avoid repeats from multiple collection days
            if {"symbol", "earnings_date", "source"}.issubset(df.columns):
                df = df.drop_duplicates(
                    subset=["symbol", "earnings_date", "source"], keep="last"
                )

            # Additional filtering
            if symbols:
                df = df[df["symbol"].isin(symbols)]

            if sources:
                df = df[df["source"].isin(sources)]

            logger.info(f"📊 Retrieved {len(df)} earnings events")
            return df

        except Exception as e:
            logger.error(f"Failed to retrieve earnings data: {e}")
            return pd.DataFrame()

    def get_latest_collection(
        self, source: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Get the most recent earnings data collection."""

        try:
            # Find the most recent collection date
            collection_dates = []
            for path in self.data_path.iterdir():
                if path.is_dir() and path.name.startswith("collection_date="):
                    date_str = path.name.replace("collection_date=", "")
                    collection_dates.append(date_str)

            if not collection_dates:
                logger.warning("No earnings collections found")
                return None

            latest_date = max(collection_dates)
            latest_date_obj = datetime.strptime(latest_date, "%Y-%m-%d").date()

            logger.info(f"📊 Loading latest earnings collection from {latest_date}")

            # Load data from latest date
            latest_path = self.data_path / f"collection_date={latest_date}"

            dataframes = []
            for source_path in latest_path.iterdir():
                if source_path.is_dir():
                    if source and not source_path.name.endswith(f"={source}"):
                        continue

                    # Read all parquet files in this source directory
                    for parquet_file in source_path.glob("*.parquet"):
                        df_chunk = pd.read_parquet(parquet_file)
                        dataframes.append(df_chunk)

            if not dataframes:
                logger.warning(f"No data found in latest collection {latest_date}")
                return None

            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)

            # Convert date strings back to dates
            combined_df["earnings_date"] = pd.to_datetime(
                combined_df["earnings_date"]
            ).dt.date
            combined_df["collection_date"] = pd.to_datetime(
                combined_df["collection_date"]
            ).dt.date

            logger.info(
                f"📊 Loaded {len(combined_df)} earnings events from latest collection"
            )
            return combined_df

        except Exception as e:
            logger.error(f"Failed to load latest collection: {e}")
            return None

    def get_historical_symbol_earnings(
        self, symbol: str, days_back: int = 90
    ) -> pd.DataFrame:
        """Get historical earnings for a specific symbol."""

        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        df = self.get_earnings_by_date_range(
            start_date=start_date, end_date=end_date, symbols=[symbol]
        )

        if not df.empty:
            # Sort by earnings date
            df = df.sort_values("earnings_date", ascending=False)
            logger.info(f"📊 Found {len(df)} historical earnings for {symbol}")

        return df

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored earnings data."""

        stats = {
            "total_collections": 0,
            "date_range": {"start": None, "end": None},
            "sources": {},
            "total_events": 0,
            "total_size_mb": 0.0,
            "unique_symbols": 0,
        }

        try:
            collection_dates = []
            all_symbols = set()

            for collection_path in self.data_path.iterdir():
                if not collection_path.is_dir() or not collection_path.name.startswith(
                    "collection_date="
                ):
                    continue

                stats["total_collections"] += 1
                date_str = collection_path.name.replace("collection_date=", "")
                collection_dates.append(date_str)

                # Process each source in this collection
                for source_path in collection_path.iterdir():
                    if not source_path.is_dir():
                        continue

                    source_name = source_path.name.replace("source=", "")
                    if source_name not in stats["sources"]:
                        stats["sources"][source_name] = {
                            "collections": 0,
                            "total_events": 0,
                            "size_mb": 0.0,
                        }

                    stats["sources"][source_name]["collections"] += 1

                    # Read parquet files to get event counts and symbols
                    for parquet_file in source_path.glob("*.parquet"):
                        file_size = parquet_file.stat().st_size
                        stats["total_size_mb"] += file_size / (1024 * 1024)
                        stats["sources"][source_name]["size_mb"] += file_size / (
                            1024 * 1024
                        )

                        # Read parquet metadata to avoid loading full data
                        try:
                            parquet_metadata = pq.read_metadata(parquet_file)
                            num_rows = parquet_metadata.num_rows
                            stats["total_events"] += num_rows
                            stats["sources"][source_name]["total_events"] += num_rows

                            # Sample symbols (read small chunk)
                            sample_df = pd.read_parquet(
                                parquet_file, columns=["symbol"]
                            )
                            all_symbols.update(sample_df["symbol"].unique())

                        except Exception as e:
                            logger.warning(
                                f"Could not read metadata from {parquet_file}: {e}"
                            )

            # Set date range
            if collection_dates:
                stats["date_range"]["start"] = min(collection_dates)
                stats["date_range"]["end"] = max(collection_dates)

            stats["unique_symbols"] = len(all_symbols)
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)

            for source in stats["sources"]:
                stats["sources"][source]["size_mb"] = round(
                    stats["sources"][source]["size_mb"], 2
                )

        except Exception as e:
            logger.error(f"Failed to calculate storage stats: {e}")

        return stats

    def _earnings_to_dataframe(
        self, earnings: List["EarningsEvent"], collection_date: date
    ) -> pd.DataFrame:
        """Convert earnings events to DataFrame for storage."""

        data = []
        for event in earnings:
            record = {
                "symbol": event.symbol,
                "company_name": event.company_name,
                "earnings_date": event.earnings_date.isoformat(),
                "time": event.time,
                "eps_estimate": event.eps_estimate,
                "eps_actual": event.eps_actual,
                "revenue_estimate": event.revenue_estimate,
                "revenue_actual": event.revenue_actual,
                "source": event.source,
                "collection_date": collection_date.isoformat(),
                "days_until_earnings": event.days_until_earnings,
                "priority_score": event.priority_score,
            }

            # Add market cap if available (from NASDAQ)
            if hasattr(event, "market_cap") and event.market_cap is not None:
                record["market_cap"] = event.market_cap

            data.append(record)

        return pd.DataFrame(data)

    def _calculate_storage_metadata(
        self,
        earnings: List["EarningsEvent"],
        collection_date: date,
        start_time: datetime,
    ) -> EarningsStorageMetadata:
        """Calculate metadata for storage batch."""

        processing_time = (datetime.now() - start_time).total_seconds()

        # Calculate date range
        earnings_dates = [e.earnings_date for e in earnings]
        date_range_start = min(earnings_dates) if earnings_dates else collection_date
        date_range_end = max(earnings_dates) if earnings_dates else collection_date

        # Count unique symbols
        unique_symbols = len(set(e.symbol for e in earnings))

        # Calculate data quality score
        quality_factors = []
        for event in earnings:
            if event.eps_estimate is not None:
                quality_factors.append(0.4)
            if event.time and event.time != "unknown":
                quality_factors.append(0.3)
            if hasattr(event, "market_cap") and event.market_cap:
                quality_factors.append(0.3)

        data_quality_score = sum(quality_factors) / len(earnings) if earnings else 0.0

        return EarningsStorageMetadata(
            collection_date=collection_date,
            source=earnings[0].source if earnings else "unknown",
            total_events=len(earnings),
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            unique_symbols=unique_symbols,
            data_quality_score=data_quality_score,
            processing_time_seconds=processing_time,
        )

    def _store_metadata(
        self,
        collection_date: date,
        storage_metadata: EarningsStorageMetadata,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store collection metadata."""

        metadata_path = (
            self.data_path
            / f"collection_date={collection_date.isoformat()}"
            / "_metadata.json"
        )

        metadata = {
            "collection_info": asdict(storage_metadata),
            "stored_at": datetime.now().isoformat(),
        }

        if additional_metadata:
            metadata["additional"] = additional_metadata

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.debug(f"📝 Stored metadata to {metadata_path}")

    def _cleanup_old_data(self):
        """Remove old data based on retention policy."""

        cutoff_date = date.today() - timedelta(days=self.retention_days)

        try:
            for collection_path in self.data_path.iterdir():
                if not collection_path.is_dir() or not collection_path.name.startswith(
                    "collection_date="
                ):
                    continue

                date_str = collection_path.name.replace("collection_date=", "")
                collection_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                if collection_date < cutoff_date:
                    # Remove old collection directory
                    import shutil

                    shutil.rmtree(collection_path)
                    logger.info(f"🗑️ Removed old earnings collection: {collection_path}")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")


def main():
    """Test the earnings repository."""
    from common.utils.logging_setup import setup_logging

    setup_logging()

    print("🗄️ EARNINGS REPOSITORY TEST")
    print("=" * 50)

    # Initialize repository
    repo = EarningsCalendarRepository()

    # Test storage stats
    stats = repo.get_storage_stats()
    print("\n📊 Current storage stats:")
    print(f"   Collections: {stats['total_collections']}")
    print(f"   Total events: {stats['total_events']}")
    print(f"   Storage size: {stats['total_size_mb']} MB")
    print(f"   Unique symbols: {stats['unique_symbols']}")

    # Test latest collection
    latest_data = repo.get_latest_collection()
    if latest_data is not None:
        print(f"\n📊 Latest collection: {len(latest_data)} events")
        print(f"   Sources: {latest_data['source'].unique()}")
        print(
            f"   Date range: {latest_data['earnings_date'].min()} to {latest_data['earnings_date'].max()}"
        )
    else:
        print("\n⚠️ No collections found")

        # Test with some sample data
        from earnings.fetcher import EarningsCalendarFetcher, EarningsSource

        print("\n🔍 Fetching sample data from NASDAQ...")
        fetcher = EarningsCalendarFetcher()
        sample_earnings = fetcher.get_upcoming_earnings(
            days_ahead=7, sources=[EarningsSource.NASDAQ]
        )

        if sample_earnings:
            print(f"   Found {len(sample_earnings)} earnings events")

            # Store the sample data
            result = repo.store_earnings_batch(sample_earnings)
            print(f"   Stored: {result}")
        else:
            print("   No sample data available")


if __name__ == "__main__":
    main()
