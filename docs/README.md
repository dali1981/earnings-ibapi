# Documentation Index

This repo implements a local-first data architecture with an event-driven bus, clear separation between examples, scripts, and jobs, and repositories as the single source of truth.

## Core Concepts
- Local-first persistence: All producers write to Parquet repositories under repo-root `data/…` before anything else.
- Event bus: A simple filesystem JSONL bus advertises dataset availability (`data.available` events). See Data Ingestion.
- Consumers: Services read from the JSONL bus with checkpoints and process idempotently; they always read from repositories, never forward API responses.
 - Correlation IDs: Events carry `payload.correlation_id` propagated from job IDs or logging context for end-to-end tracing.

## Doc Map
- Architecture: High-level flows and prerequisites
  - `DATA_FLOW_ARCHITECTURE.md` — dependencies (contracts → chains + equity → option bars), historical vs streaming sequences, deterministic writes.
- Data Ingestion: Local-first policy and event bus
  - `data_ingestion.md` — event topics, JSONL consumer + checkpoints, idempotency index, recovery & catch-up, systemd sample.
- Jobs Overview
  - `jobs.md` — taxonomy (examples vs scripts vs jobs), active jobs, deprecated jobs, usage notes.
- Contributor Guide
  - `../AGENTS.md` — repository guidelines, coding/testing/style, architecture & ingestion pointers.

## Quick Actions (Make)
- View upcoming earnings (local): `make show-earnings`
- Deduplicate earnings partitions: `make dedup-earnings`
- Consume option chain events and backfill bars: `make consume-option-chains`

## Taxonomy (TL;DR)
- examples/: read-only demos (no network, no side effects)
- scripts/: local utilities working on repo data only
- jobs/: operational workflows (may call IB/APIs), scheduled, emit events

## Services (FastAPI)
- Calendar Service (earnings): `http://localhost:9001`
  - `GET /calendar/upcoming?days=14&symbols=AAPL,MSFT`
- Equity History: `http://localhost:9002`
  - `GET /equity/{symbol}/bars?start=YYYY-MM-DD&end=YYYY-MM-DD&bar_size=1%20day`
- Option History: `http://localhost:9003`
  - `GET /options/{underlying}/bars?...` and `GET /options/{underlying}/contracts`
- Option Chains: `http://localhost:9004`
  - `GET /chains/{symbol}/latest`
- Frontend: `http://localhost:9000` (simple UI to call the services)

### Dev Compose
- `docker compose -f docker-compose.dev.yml up --build`
- Open the frontend at `http://localhost:9000`

Note: The legacy Flask monitoring dashboard remains available at `monitoring/dashboard.py` but is not used in the new service layout.

## Directory Layout
- See `docs/SERVICES.md` for the service structure, layering, and run/test guidance.
