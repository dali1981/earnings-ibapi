# Earnings IBAPI — Local-First Data + Event-Driven Consumers

This repo persists market/earnings data locally (Parquet) and notifies services via a simple file-based event bus. Producers always write to repositories first; consumers read locally and react to events with durable checkpoints.

## Quickstart
- Setup: `make dev` (creates `venv`, installs local deps and tools)
- Run unit tests for ingestion components:
  - `pytest -q tests/unit/test_event_bus_consumer.py tests/unit/test_processed_index.py --cov-fail-under=0`
  - `pytest -q tests/unit/test_backfill_*_flow.py --cov-fail-under=0`

## Option Chain → Option Bars Consumer
- One-shot (main topic): `make consume-option-chains`
- One-shot (demo): `make consume-option-chains-demo`
- Behavior:
  - Reads `option_chains` events, backfills option bars via IB historical API, writes to `option_bars/`, then emits `option_bars` events.
  - Uses checkpoints + idempotency (safe to re-run, resumes after restarts).
- Service mode: See `docs/data_ingestion.md` for a systemd unit sample or use cron/timers for periodic runs.

## Docs
- Ingestion protocol, recovery/catch-up, and consumers: `docs/data_ingestion.md`
- Contributor guidelines: `AGENTS.md`

## Notes
- Experimental code lives under `experimental/` (excluded from builds/formatters).
- Vendored code in `ibx-*/` remains for now; consider switching to pip packages to reduce maintenance.
