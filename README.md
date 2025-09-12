# Earnings IBAPI — Local-First Data + Event-Driven Consumers

This repo persists market/earnings data locally (Parquet) and notifies services via a simple file-based event bus. Producers always write to repositories first; consumers read locally and react to events with durable checkpoints.

## Quickstart
- Setup: `make dev` (creates `venv`, installs local deps and tools)
- Run core tests: `make test-unit` (comprehensive test suite with 85% coverage target)
- View local upcoming earnings: `make show-earnings` (reads from the repo, no external calls)
- Deduplicate past earnings partitions: `make dedup-earnings` (merges all files per partition and removes duplicates)

## Testing

### Test Categories
The project uses pytest with comprehensive markers for organized testing:

```bash
# Run all unit tests (recommended for development)
make test-unit

# Run specific test categories
pytest -m repository          # Repository/data layer tests
pytest -m event_bus           # Event system tests  
pytest -m integration         # Cross-component tests
pytest -m api                 # API endpoint tests
pytest -m performance         # Performance and load tests
pytest -m reliability         # Error handling and recovery tests
pytest -m mock               # Mock/simulation tests
pytest -m slow               # Long-running tests (excluded by default)
```

### Repository Tests
Test data persistence and querying across all repository types:

```bash
# Test all repositories
pytest tests/unit/repositories/ -v

# Test specific repositories
pytest tests/unit/repositories/test_option_chains_repository.py
pytest tests/unit/repositories/test_option_contracts_repository.py  
pytest tests/unit/repositories/test_earnings_repository.py
pytest tests/unit/repositories/test_equity_bars_repository.py
```

### Event System Tests
Test file-based event bus and consumer infrastructure:

```bash
# Event bus core functionality
pytest tests/unit/events/test_file_event_bus.py -v

# Event consumers and processors
pytest tests/unit/test_event_bus_consumer.py tests/unit/test_processed_index.py

# Backfill workflow tests  
pytest tests/unit/test_backfill_*_flow.py
```

### API and Service Tests
Test FastAPI microservices and endpoints:

```bash
# All service tests
make test-integration

# Specific services
pytest tests/integration/calendar/ -v      # Calendar API tests
pytest tests/integration/frontend/ -v      # Frontend integration tests  
pytest tests/integration/orchestrator/ -v  # Orchestrator workflow tests
```

### Coverage and Quality
```bash
# Run with coverage reporting (85% threshold)
make coverage

# Generate HTML coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Code formatting and linting
make format && make lint
```

### Test Configuration
- **Configuration**: `pytest.ini` with markers and coverage settings
- **Fixtures**: `tests/conftest.py` provides advanced fixtures for repositories, mocks, and data generation
- **Test Data**: Realistic market data generation with Black-Scholes option pricing
- **Mocks**: Comprehensive IB API and event bus simulation
- **Data Structures**: PyArrow/pandas compatibility helpers for repository testing

### Development Testing Workflow
```bash
# Standard development cycle
make dev                    # Setup environment
make test-unit             # Run core tests  
make format && make lint   # Code quality checks
make coverage             # Verify coverage targets

# Focus on specific components during development
pytest tests/unit/repositories/ -v --tb=short  # Repository development
pytest tests/unit/events/ -v --tb=short       # Event system development
pytest -m api -v --tb=short                   # API development
```

### Test Infrastructure  
The testing system includes:
- **880+ test cases** across repository, event, and integration layers
- **Advanced fixtures** with realistic market data generation
- **PyArrow/pandas compatibility** for data structure testing
- **Event bus simulation** with correlation tracking
- **Mock factories** for IB API and external service simulation
- **Performance testing** for large dataset operations
- **Error injection** for reliability testing

**Test Status (After Import/Infrastructure Fixes)**:
- **Tests are now functional** - All import errors resolved ✅
- **90 out of 208 unit tests passing** (43.3% success rate)
- **Advanced test infrastructure** with PyArrow/pandas compatibility
- **Remaining failures** are primarily validation, edge cases, and advanced features

**Test Progress by Component**:
- Event Bus: Core functionality working, some edge cases failing  
- Repository Layer: Basic operations working, advanced features need implementation
- API Layer: Some integration issues remain
- Task System: Requires additional mocking and setup

The test suite now provides a solid foundation for continued development with comprehensive fixtures, realistic data generation, and proper error handling.

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
