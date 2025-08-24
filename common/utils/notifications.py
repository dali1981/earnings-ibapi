"""
Notifications utility: simple, local-first notifiers.

- FileNotifier: appends JSONL notifications under `data/cache/notifications.jsonl`.
- LoggerNotifier: writes to Python logging (for visibility during runs/tests).
- MultiNotifier: fan-out to multiple notifiers.

Network notifiers (e.g., Slack/email) are intentionally omitted by default to
keep the system local-first. They can be added later behind env/config flags.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import config


Severity = str  # one of: "critical", "error", "warning", "info", "success"


@dataclass
class Notification:
    message: str
    severity: Severity = "info"
    category: str = "general"
    context: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())

    def to_json(self) -> str:
        return json.dumps(
            {
                "message": self.message,
                "severity": self.severity,
                "category": self.category,
                "context": self.context,
                "ts": self.ts,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )


class Notifier(Protocol):
    def notify(
        self,
        message: str,
        *,
        severity: Severity = "info",
        category: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:  # pragma: no cover - interface only
        ...


class FileNotifier:
    """Append notifications to a JSONL file under cache.

    Path: `config.CACHE_PATH / "notifications.jsonl"`
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        root = root or config.CACHE_PATH
        self.path = Path(root) / "notifications.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def notify(
        self,
        message: str,
        *,
        severity: Severity = "info",
        category: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        note = Notification(message=message, severity=severity, category=category, context=dict(context or {}))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(note.to_json() + "\n")


class LoggerNotifier:
    """Emit notifications via Python logging."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("notifications")

    def notify(
        self,
        message: str,
        *,
        severity: Severity = "info",
        category: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {"category": category, **(context or {})}
        if severity in ("critical", "error"):
            self.logger.error("%s | %s", message, payload)
        elif severity == "warning":
            self.logger.warning("%s | %s", message, payload)
        elif severity == "success":
            self.logger.info("%s | %s", message, payload)
        else:
            self.logger.info("%s | %s", message, payload)


class MultiNotifier:
    """Fan-out notifications to multiple notifiers."""

    def __init__(self, notifiers: Iterable[Notifier]):
        self._notifiers: List[Notifier] = list(notifiers)

    def notify(
        self,
        message: str,
        *,
        severity: Severity = "info",
        category: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        for n in self._notifiers:
            try:
                n.notify(message, severity=severity, category=category, context=context)
            except Exception:  # pragma: no cover - best-effort fan-out
                logging.getLogger("notifications").exception("Notifier failed: %s", type(n).__name__)


def get_default_notifier(*, include_logger: bool = True) -> Notifier:
    """Construct a sensible default notifier for local usage.

    - Always writes to `cache/notifications.jsonl`
    - Optionally logs to the console/logger
    """
    notifiers: List[Notifier] = [FileNotifier()]
    if include_logger:
        notifiers.append(LoggerNotifier())
    return MultiNotifier(notifiers)


__all__ = [
    "Notification",
    "Notifier",
    "FileNotifier",
    "LoggerNotifier",
    "MultiNotifier",
    "get_default_notifier",
]

