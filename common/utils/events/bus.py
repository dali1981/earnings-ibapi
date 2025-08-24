from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import config


@dataclass
class Event:
    type: str
    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())

    def to_json(self) -> str:
        return json.dumps(
            {"type": self.type, "topic": self.topic, "payload": self.payload, "ts": self.ts},
            separators=(",", ":"),
            ensure_ascii=False,
        )


class EventBus:
    def publish(self, event_type: str, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def notify_data_available(self, dataset: str, **metadata: Any) -> None:
        self.publish("data.available", topic=dataset, payload=metadata)


class FileEventBus(EventBus):
    def __init__(self, root: Optional[Path] = None) -> None:
        root = root or config.CACHE_PATH
        self.events_dir = Path(root) / "events"
        self.ready_dir = Path(root) / "ready"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)

    def publish(self, event_type: str, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        # Enrich payload with correlation_id when available
        enriched: Dict[str, Any] = dict(payload or {})

        # Try to derive correlation_id in the following order:
        # 1) Explicit in payload (correlation_id)
        # 2) Explicit job_id in payload (treat as correlation)
        # 3) reliability.logging context (if configured)
        # 4) Environment JOB_ID
        corr = enriched.get("correlation_id")
        if not corr:
            corr = enriched.get("job_id")
        if not corr:
            try:
                from common.reliability.logging import get_current_context  # type: ignore

                ctx = get_current_context()
                if ctx and getattr(ctx, "correlation_id", None):
                    corr = ctx.correlation_id
            except Exception:
                pass
        if not corr:
            import os

            corr = os.environ.get("JOB_ID")

        if corr and "correlation_id" not in enriched:
            enriched["correlation_id"] = corr

        evt = Event(type=event_type, topic=topic, payload=enriched)
        log_path = self.events_dir / f"{topic}.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(evt.to_json() + "\n")
        (self.ready_dir / f"{topic}.flag").touch()
