"""
Library component for earnings data persistence (repository-only).

This module does not call external APIs. It exposes a small facade over
the EarningsCalendarRepository so callers can persist provided batches and read
local summaries.
"""

from __future__ import annotations

from common.repositories.earnings import EarningsCalendarRepository
from common.utils.events.bus import EventBus, FileEventBus
from config import DATA_ROOT
from datetime import date
from earnings.fetcher import EarningsEvent
from pathlib import Path
from typing import List, Optional

import logging


logger = logging.getLogger(__name__)


class EarningsPersistence:
    """Persist earnings batches and read local summaries (no external calls)."""

    def __init__(
        self, use_config_paths: bool = True, base_data_root: Optional[Path] = None
    ):
        """Initialize storage service with configurable paths.

        Args:
            use_config_paths: Use config.DATA_ROOT when True (ignored if base_data_root set)
            base_data_root: Optional explicit base path (e.g., demo base)
        """
        if base_data_root is not None:
            earnings_path = base_data_root / "earnings"
            logger.info("Using provided base path: %s", base_data_root)
        elif use_config_paths:
            earnings_path = DATA_ROOT / "earnings"
            logger.info("Using configured data path: %s", earnings_path)
        else:
            earnings_path = Path("data/earnings")
            logger.warning("Using fallback data path: %s", earnings_path)

        self.repository = EarningsCalendarRepository(earnings_path.parent)
        logger.info("Earnings persistence initialized (repository-backed)")

    def persist_earnings_batch(
        self,
        earnings: List[EarningsEvent],
        collection_date: Optional[date] = None,
        metadata: Optional[dict] = None,
        *,
        bus: Optional[EventBus] = None,
        dataset_topic: str = "earnings",
    ) -> dict:
        """Persist a provided batch of EarningsEvent objects into the repository."""
        try:
            payload = earnings or []
            if not payload:
                return {"status": "no_data", "events_collected": 0}

            storage_result = self.repository.store_earnings_batch(
                earnings=payload,
                collection_date=collection_date or date.today(),
                metadata=metadata or {},
            )

            result = {
                "status": "success",
                "collection_date": storage_result["collection_date"],
                "events_collected": storage_result["total_events_stored"],
                "storage": storage_result,
            }

            # Publish data-available notification
            try:
                ebus = bus or FileEventBus()
                ebus.notify_data_available(
                    dataset=dataset_topic,
                    collection_date=storage_result["collection_date"],
                    total_events=storage_result.get("total_events_stored", 0),
                    sources=list(storage_result.get("sources", {}).keys()),
                )
            except Exception:
                logger.exception("Failed to publish earnings availability event")

            return result
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Persistence failed")
            return {"status": "error", "error": str(e)}

    def get_storage_summary(self) -> dict:
        """Return a summary of stored data."""
        try:
            stats = self.repository.get_storage_stats()
            return {
                "total_collections": stats.get("total_collections", 0),
                "total_events": stats.get("total_events", 0),
                "storage_size_mb": stats.get("total_size_mb", 0.0),
                "unique_symbols": stats.get("unique_symbols", 0),
                "date_range": stats.get("date_range", {}),
                "sources": stats.get("sources", {}),
            }
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Failed to get storage summary")
            return {"error": str(e)}
