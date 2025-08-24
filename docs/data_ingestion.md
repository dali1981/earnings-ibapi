# Data Ingestion Protocol

This project enforces a local-first workflow: every external read (historical API or live subscription) writes to repositories first; other services then read locally and are notified via file-based events.

## Roles & Responsibilities
- Historical sources: backfill helpers in `ibx_flows/backfill.py` call a `MarketDataSource`, persist to repositories, then publish a `data.available` event.
- Streaming sources: subscription handlers append to repositories in small batches (e.g., 1s/1m); each batch publishes a `data.available` event.
- Consumers: read only from repositories (`repositories/**` or helpers in `utils/local_data.py`). Never consume API responses directly.

## Repositories (Local "DB")
- Format: PyArrow/Parquet, Hive partitions under `data/` (or a provided base path).
- Key datasets: `equity_bars/`, `option_chains/`, `option_bars/`, `earnings/`.
- Access: via repository classes (e.g., `EquityBarRepository.load(symbol=...)`).

## Events (Filesystem Bus)
- Publisher: `utils.events.bus.FileEventBus` writes JSONL under `data/cache/events/<topic>.jsonl` and touches `data/cache/ready/<topic>.flag` to signal availability.
- Topics: `equity_bars`, `option_chains`, `option_bars`, `earnings`. Demo runs use a `.demo` suffix.
- Payload example: `{ "symbol": "AAPL", "bar_size": "1 day", "rows": 390, "start": "2024-01-01", "end": "2024-01-31" }`.

### Correlation IDs (Tracing Jobs)
- Correlation IDs are automatically propagated on publish. The bus injects `payload.correlation_id` if missing using, in order:
  1) `payload.correlation_id` (explicit)
  2) `payload.job_id` (treated as correlation)
  3) `reliability.logging` context `correlation_id` (if a `log_context` is active)
  4) `JOB_ID` environment variable

- Producers should include a `job_id` or run within `reliability.logging.log_context(correlation_id=...)` for consistent tracing.
- Consumers should preserve correlation by processing within the same `log_context` when emitting downstream events.

Example (producer):
```python
from utils.events.bus import FileEventBus
ebus = FileEventBus()
ebus.notify_data_available(dataset="option_bars", symbol="AAPL", start="2024-01-01", end="2024-01-31", job_id="job-1234")
```

Example (consumer propagation):
```python
from reliability.logging import log_context

for evt in cons.read(service="backfill", limit=50):
    payload = evt.get("payload", {})
    corr = payload.get("correlation_id") or payload.get("job_id")
    with log_context(correlation_id=corr):
        # do work, and when publishing, the bus will attach the same correlation_id
        ebus.notify_data_available(dataset="option_bars", symbol=payload["underlying"], start=payload["start"], end=payload["end"]) 
```

## Historical vs. Streaming
- Historical (backfill):
  1) Fetch a bounded window from the source.
  2) Write to the repository (normalized, partitioned).
  3) Publish a `data.available` event with window metadata.
- Streaming (subscription):
  1) Receive ticks/bars via `api/subscriptions.Subscription` callback.
  2) Buffer and flush to repository on a cadence (e.g., each N records or seconds).
  3) Publish a `data.available` event per flush (small payload).

## Safety & Isolation
- Path anchoring: all `data/`, `cache/`, and `logs/` live under the repository root (via `config.PROJECT_ROOT`). Running scripts from subfolders will not create nested `data/` in those subfolders.
- Scripts are guarded from writing to production data without `ALLOW_PROD_WRITES=1`.
- Example/demo scripts write under `data/exports/demo` and use namespaced topics (`*.demo`).

## Minimal Usage
- Backfill equity bars: call `backfill_equity_bars(source, repo, cfg)`. After completion:
  - Files under `data/equity_bars/symbol=<SYM>/...`
  - Event appended to `data/cache/events/equity_bars.jsonl`
- Streaming: register a `Subscription` handler that writes to a repo and calls `FileEventBus().notify_data_available(...)` per flush.

### Event Introspection API
- The monitoring dashboard exposes `GET /events` for recent events:
  - Query params: `topic`, `type`, `hours` (default 24), `aggregate` (truthy for counts by topic/type)
  - Example: `/events?topic=option_bars&hours=6`
  - Aggregate example: `/events?aggregate=1&hours=24`
  - Filter by correlation on the UI via the “Correlation ID” search box

## Idempotency & Deduplication
- Earnings persistence is idempotent per `(collection_date, source)`: re-running the storage on the same day merges with any existing file and drops duplicates by `(symbol, earnings_date, source)`, writing a single deterministic `earnings.parquet` in the partition.
- For other datasets, follow the same pattern where relevant (e.g., consolidated files per partition) to avoid duplicate growth across runs.

## Recovery & Catch-Up
- Durable log + checkpoint: consumers track `<file, offset, last_ts>` under `data/cache/state/<service>/<topic>.json` and resume from there.
- JSONL consumer utility: `utils.events.consumer.JSONLConsumer(topic)` reads new events and advances the checkpoint per service.

Example (consumer loop):
```python
from utils.events.consumer import JSONLConsumer

cons = JSONLConsumer("equity_bars")
while True:
    events = cons.read(service="analytics", limit=100)
    if not events:
        break  # or sleep for streaming
    for evt in events:
        # evt = {"type","topic","payload","ts"}
        payload = evt["payload"]
        # idempotent work based on payload (e.g., symbol+date)
        process(payload)
```

- Rotation: if you rotate files, store the current filename in the checkpoint; on mismatch, resume from the new file and optionally reconcile from `last_ts`.
- Reconciliation: periodically scan repositories (e.g., `present_dates`) and compare to processed keys; enqueue anything missing.

## Option Chain Backfill Consumer
- Script: `jobs/option_chain_backfill_consumer.py`
- Purpose: consume `option_chains` events and backfill historical option bars for the referenced underlying/date window.
- Usage:
  - `python jobs/option_chain_backfill_consumer.py --service backfill-broker --topic option_chains --bar-size "1 day" --once`
  - Demo topic: `--topic option_chains.demo`
- Behavior:
  1) Reads events from JSONL bus with per-service checkpointing.
  2) Runs `backfill_option_bars(IBSource, OptionBarRepository, cfg)` for each event.
  3) Backfill writes to repo and publishes `option_bars` data.available events.
  4) Marks idempotent key `underlying:start:end:bar_size` as processed.
- Make targets:
  - `make consume-option-chains`
  - `make consume-option-chains-demo`

### systemd sample
Create `option-chain-backfill.service`:
```
[Unit]
Description=Option Chain Backfill Consumer
After=network.target

[Service]
WorkingDirectory=/path/to/earnings_ibapi
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/env python jobs/option_chain_backfill_consumer.py --service backfill-broker --topic option_chains --bar-size "1 day"
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
For periodic runs, use a timer unit or cron. The consumer uses checkpoints, so repeated runs catch up and process idempotently.
