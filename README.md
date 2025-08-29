# Earnings IBAPI — Local-First Data + Event-Driven Consumers

This repo persists market/earnings data locally (Parquet) and notifies services via a simple file-based event bus. Producers always write to repositories first; consumers read locally and react to events with durable checkpoints.

## Quickstart
- Setup: `make dev` (creates `venv`, installs local deps and tools)
- Run unit tests for ingestion components:
  - `pytest -q tests/unit/test_event_bus_consumer.py tests/unit/test_processed_index.py --cov-fail-under=0`
  - `pytest -q tests/unit/test_backfill_*_flow.py --cov-fail-under=0`
- View local upcoming earnings: `make show-earnings` (reads from the repo, no external calls)
- Deduplicate past earnings partitions: `make dedup-earnings` (merges all files per partition and removes duplicates)

## Structure
- Common libraries under `common/`: `common/repositories`, `common/utils`, `common/ib`, `common/lineage`, `common/reliability`, `common/monitoring`.
- Services under `services/`: `services/calendar_api`, `services/equity_api`, `services/option_history_api`, `services/option_chain_api`, `services/frontend`.
- Domain packages remain: `earnings/`, `trading/`, vendored `ibx_*` modules, etc.

## Option Chain → Option Bars Consumer
- One-shot (main topic): `make consume-option-chains`
- One-shot (demo): `make consume-option-chains-demo`
- Behavior:
  - Reads `option_chains` events, backfills option bars via IB historical API, writes to `option_bars/`, then emits `option_bars` events.
  - Uses checkpoints + idempotency (safe to re-run, resumes after restarts).
- Service mode: See `docs/data_ingestion.md` for a systemd unit sample or use cron/timers for periodic runs.

## Docs
- Docs index: `docs/README.md`
- Ingestion protocol, recovery/catch-up, and consumers: `docs/data_ingestion.md`
- Architecture & data flow: `docs/DATA_FLOW_ARCHITECTURE.md`
- Jobs overview & taxonomy: `docs/jobs.md`
- Contributor guidelines: `AGENTS.md`

## Docker
- Start dashboard + sqlite db:
  - `docker compose up --build`
  - App: http://localhost:9090
  - SQLite DB persisted in named volume `monitoring_db` (mounted at `/var/db/monitoring.db`).
- App container mounts the repo at `/app` so it can read `data/` and the JSONL events under `data/cache/events`.
- To inspect DB inside the app container: `docker exec -it trading_app sqlite3 /var/db/monitoring.db`.

### Dev mode (live reload)
- Use a dev compose override (enables Flask debug + reloader and mounts the source):
  - `docker compose -f docker-compose.dev.yml up --build`
  - App: http://localhost:9090
  - Code changes under the repo trigger auto-reload inside the container.

## FastAPI Services (Dev)
- Run all services with dev compose:
  - `make services-up` (equivalent to `docker compose -f docker-compose.dev.yml up --build`)
  - Frontend: http://localhost:9000
  - Calendar: http://localhost:9001
  - Equity: http://localhost:9002
  - Options: http://localhost:9003
  - Chains: http://localhost:9004
- Stop services:
  - `make services-down`
- Run individually (local venv, no Docker):
  - `make run-calendar` | `make run-equity` | `make run-options` | `make run-chains` | `make run-frontend`

### Service commands (local)
- Run via services paths, e.g.:
  - `uvicorn services.calendar_api.app:app --reload --port 9001`
  - `uvicorn services.equity_api.app:app --reload --port 9002`
  - `uvicorn services.option_history_api.app:app --reload --port 9003`
  - `uvicorn services.option_chain_api.app:app --reload --port 9004`
  - `uvicorn services.frontend.main:app --reload --port 9000`

## Frontend Service URL Resolution
- Source of truth: `app_settings.Settings` (see `app_settings/settings.py`). The frontend reads service bases from `settings.api.services.*.url` and injects them into the page as `window.__BASES__`.
- Preferred overrides via env (nested):
  - `APP_API__SERVICES__CALENDAR__HOST=svc-calendar`, `APP_API__SERVICES__CALENDAR__PORT=9001`
  - `APP_API__SERVICES__EQUITY__HOST=svc-equity`, `APP_API__SERVICES__EQUITY__PORT=9002`
  - `APP_API__SERVICES__CHAINS__HOST=svc-chains`, `APP_API__SERVICES__CHAINS__PORT=9004`
  - `APP_API__SERVICES__ORCHESTRATOR__HOST=svc-orchestrator`, `APP_API__SERVICES__ORCHESTRATOR__PORT=9005`
- Legacy overrides (still supported):
  - `FRONTEND_CALENDAR_BASE`, `FRONTEND_EQUITY_BASE`, `FRONTEND_CHAINS_BASE`, `FRONTEND_ORCHESTRATOR_BASE`
- Client-side usage:
  - `window.TradingAPI` provides a small API layer used by the frontend controllers; you can also call it from the console.
    - Example: `TradingAPI.calendar.getUpcoming(7, 'AAPL,MSFT').then(({data}) => console.log(data))`
  - UI elements bind to controller actions via `data-action` attributes, wired by `/static/js/bootstrap.js`.

## Notes
- Experimental code lives under `labs/` (excluded from builds/formatters).
- Vendored code in `ibx-*/` remains for now; consider switching to pip packages to reduce maintenance.
- Data paths are anchored to the repo root (see `config.PROJECT_ROOT`). Running from `ops/scripts/` or other subfolders will still write to `<repo>/data` and `<repo>/data/cache`.
- Earnings persistence is idempotent per day/source: re-running storage on the same collection date merges and deduplicates records within the partition, preventing duplicate counts.
