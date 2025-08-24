Notifications
=============

Overview
--------
- Local-first notifications for jobs and services.
- Append JSONL records to `data/cache/notifications.jsonl` for auditability.
- Optional logging output for immediate visibility during runs.

API
---
- `utils.notifications.FileNotifier`: writes JSONL lines to cache.
- `utils.notifications.LoggerNotifier`: emits via Python logging.
- `utils.notifications.MultiNotifier`: fan-out to multiple backends.
- `utils.notifications.get_default_notifier()`: sensible default (file + logger).

Record schema (JSONL)
---------------------
- `message` (str): human message.
- `severity` (str): `critical|error|warning|info|success`.
- `category` (str): logical group (e.g., `jobs`, `events`, `data`).
- `context` (object): extra metadata (symbol, job id, counts).
- `ts` (float): epoch seconds.

Usage
-----
```python
from utils.notifications import get_default_notifier

notify = get_default_notifier()
notify.notify(
    "Equity bars backfill complete",
    severity="success",
    category="jobs",
    context={"job": "equity_bars", "rows": 123, "symbol": "AAPL"},
)
```

Event Bus vs Notifications
--------------------------
- Event Bus: machine-consumable events in `data/cache/events/<topic>.jsonl` with ready flags in `data/cache/ready/`.
- Notifications: human-oriented messages (still JSONL) in `data/cache/notifications.jsonl`.
- They complement each other: jobs write data -> publish event -> optionally send a notification.

Notes
-----
- External channels (Slack/email) are not enabled by default to preserve offline, reproducible workflows. They can be added later behind environment flags.
- Keep messages short and include useful `context`.

