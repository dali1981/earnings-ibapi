"""
Lightweight event bus for local, file-based notifications.

Design goals:
- Keep external consumers decoupled from data ingestion.
- Persist events as JSONL for auditability and simple replay.
- Avoid network services; use filesystem in `config.CACHE_PATH`.

Usage:
    bus = FileEventBus()
    bus.notify_data_available(dataset="equity_bars", symbol="AAPL", bar_size="1 day")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import config
import json
import time


@dataclass
class Event:
    """Structured event payload written to JSONL."""

    type: str
    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type,
                "topic": self.topic,
                "payload": self.payload,
                "ts": self.ts,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )


class EventBus:
    """Abstract event bus interface."""

    def publish(
        self, event_type: str, topic: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        raise NotImplementedError

    def notify_data_available(self, dataset: str, **metadata: Any) -> None:
        """Convenience: publish a standardized data-available event."""
        self.publish("data.available", topic=dataset, payload=metadata)


class FileEventBus(EventBus):
    """File-based event bus that appends JSONL under `cache/events/` and writes a touch file in `cache/ready/`.

    - Events: `data/events/<topic>.jsonl`
    - Ready flags: `data/cache/ready/<topic>.flag`
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        root = root or config.CACHE_PATH
        self.events_dir = Path(root) / "events"
        self.ready_dir = Path(root) / "ready"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)

    def publish(
        self, event_type: str, topic: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        evt = Event(type=event_type, topic=topic, payload=payload or {})
        # Append to JSONL log per topic
        log_path = self.events_dir / f"{topic}.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(evt.to_json() + "\n")
        # Update a ready file that other services can watch via mtime
        flag = self.ready_dir / f"{topic}.flag"
        flag.touch()
