from __future__ import annotations

from common.reliability.logging import get_trading_logger
from common.repositories.earnings import EarningsCalendarRepository
from common.utils.events.bus import FileEventBus
from datetime import date, datetime, timedelta
from earnings.fetcher import EarningsEvent
from pathlib import Path
from typing import List, Optional

import config
import json


class CalendarService:
    """Thin service over the calendar repository.

    - Reads upcoming events from the repo
    - Writes new events (dedup occurs in the repository method)
    - Exposes presence checks and stats if needed by orchestrators
    """

    def __init__(self) -> None:
        self._repo = EarningsCalendarRepository()
        self._logger = get_trading_logger(__name__)

    def get_upcoming_events(
        self, *, days_ahead: int, symbols: Optional[List[str]] = None
    ) -> List[EarningsEvent]:
        start = date.today()
        end = start + timedelta(days=days_ahead)
        df = self._repo.get_earnings_by_date_range(start, end, symbols=symbols)
        if df is None or df.empty:
            return []
        # Convert rows back into EarningsEvent-like objects (dicts also OK for the router schema)
        events: List[EarningsEvent] = []
        for _, row in df.iterrows():
            events.append(
                EarningsEvent(
                    symbol=row.get("symbol"),
                    company_name=row.get("company_name", row.get("symbol")),
                    earnings_date=row.get("earnings_date"),
                    time=row.get("time"),
                    eps_estimate=row.get("eps_estimate"),
                    eps_actual=row.get("eps_actual"),
                    revenue_estimate=row.get("revenue_estimate"),
                    revenue_actual=row.get("revenue_actual"),
                    source=row.get("source", "unknown"),
                )
            )
        return events

    def save_events(self, events: List[EarningsEvent]) -> int:
        if not events:
            self._logger.info("📭 No events to save")
            return 0

        self._logger.info(f"💾 Saving {len(events)} earnings events")
        # Normalize potential Pydantic enum fields (time/source) to strings for storage
        normalized: List[EarningsEvent] = []
        try:
            from enum import Enum

            for e in events:
                time_val = getattr(e, "time", None)
                if isinstance(time_val, Enum):
                    time_val = time_val.value
                source_val = getattr(e, "source", None)
                if isinstance(source_val, Enum):
                    source_val = source_val.value
                normalized.append(
                    EarningsEvent(
                        symbol=e.symbol,
                        company_name=e.company_name,
                        earnings_date=e.earnings_date,
                        time=time_val,
                        eps_estimate=getattr(e, "eps_estimate", None),
                        eps_actual=getattr(e, "eps_actual", None),
                        revenue_estimate=getattr(e, "revenue_estimate", None),
                        revenue_actual=getattr(e, "revenue_actual", None),
                        source=source_val or "unknown",
                    )
                )
        except Exception:
            # Best-effort: if normalization fails, fall back to original list
            normalized = events

        # Pre-filter: check if events already present in repo across all collections
        try:
            if normalized:
                symbols = sorted({e.symbol for e in normalized})
                min_date = min(e.earnings_date for e in normalized)
                max_date = max(e.earnings_date for e in normalized)
                sources = sorted({getattr(e, "source", "unknown") for e in normalized})
                df_existing = self._repo.get_earnings_by_date_range(
                    start_date=min_date,
                    end_date=max_date,
                    symbols=symbols,
                    sources=sources,
                )
                existing_keys = set()
                if df_existing is not None and not df_existing.empty:
                    for _, r in df_existing.iterrows():
                        existing_keys.add(
                            (r.get("symbol"), r.get("earnings_date"), r.get("source"))
                        )
                to_write = [
                    e
                    for e in normalized
                    if (e.symbol, e.earnings_date, getattr(e, "source", "unknown"))
                    not in existing_keys
                ]
            else:
                to_write = []
        except Exception:
            # If presence check fails, fall back to writing all (idempotency handled downstream)
            to_write = normalized

        if not to_write:
            self._logger.info(
                f"✅ All {len(normalized)} events already exist (deduplicated)"
            )
            return 0

        self._logger.info(
            f"📝 Writing {len(to_write)} new events (filtered {len(normalized) - len(to_write)} duplicates)"
        )
        res = self._repo.store_earnings_batch(to_write)

        # Publish a data-available event for downstream consumers
        try:
            ebus = FileEventBus()
            ebus.notify_data_available(
                dataset="earnings",
                collection_date=(
                    res.get("collection_date") or date.today().isoformat()
                ),
                total_events=int(res.get("total_events_stored") or len(to_write)),
                sources=list(
                    {
                        (
                            e.source.value
                            if hasattr(e, "source")
                            and hasattr(getattr(e, "source"), "value")
                            else getattr(e, "source", "unknown")
                        )
                        for e in to_write
                    }
                ),
            )
        except Exception:
            # Best-effort notification; do not fail write on event bus errors
            self._logger.exception("Failed to publish earnings availability event")

        return int(res.get("total_events_stored") or len(to_write))

    def is_fresh(self) -> bool:
        """Return True if the latest collection is within the freshness window.

        Freshness window is defined by `config.EARNINGS_CONFIG['cache_hours']` (default 6h).
        Uses the `_metadata.json` file's `stored_at` timestamp in the most recent
        `collection_date=YYYY-MM-DD` directory. Falls back to False if metadata
        is missing.
        """
        try:
            cache_hours = int(config.EARNINGS_CONFIG.get("cache_hours", 6))
            root = Path(self._repo.data_path)
            # Find latest collection_date directory
            latest_dir: Path | None = None
            latest_date: str | None = None
            for p in root.iterdir():
                if p.is_dir() and p.name.startswith("collection_date="):
                    ds = p.name.split("=", 1)[1]
                    if latest_date is None or ds > latest_date:
                        latest_date = ds
                        latest_dir = p
            if not latest_dir:
                return False
            # If we already have a collection for today, consider it fresh for the rest of the day
            if latest_date == date.today().isoformat():
                return True
            meta_path = latest_dir / "_metadata.json"
            if not meta_path.exists():
                return False
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            stored_at = data.get("stored_at") or data.get("collection_info", {}).get(
                "stored_at"
            )
            if not stored_at:
                return False
            try:
                ts = datetime.fromisoformat(stored_at)
            except Exception:
                return False
            return datetime.now() - ts <= timedelta(hours=cache_hours)
        except Exception:
            return False

    def get_latest_collection_date_str(self) -> Optional[str]:
        try:
            root = Path(self._repo.data_path)
            latest_date: str | None = None
            for p in root.iterdir():
                if p.is_dir() and p.name.startswith("collection_date="):
                    ds = p.name.split("=", 1)[1]
                    if latest_date is None or ds > latest_date:
                        latest_date = ds
            return latest_date
        except Exception:
            return None
