# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Create/activate virtual environment and install all dependencies
make dev

# Rebuild environment from scratch
make dev-rebuild

# Activate virtual environment manually
source venv/bin/activate
```

### Testing Commands
```bash
# Run specific test files
pytest tests/unit/test_equity_bars_repository.py::TestEquityBarRepository::test_repository_initialization -v

# Run unit tests for key components
pytest -q tests/unit/test_event_bus_consumer.py tests/unit/test_processed_index.py --cov-fail-under=0
pytest -q tests/unit/test_backfill_*_flow.py --cov-fail-under=0

# Run test suites by category
make test-unit        # Unit tests only
make test-integration # Integration tests
make test-events      # Event bus tests
make test-frontend    # Frontend tests
make test-calendar    # Calendar service tests

# Run all tests with coverage
make coverage
```

### Code Quality
```bash
# Format code (uses Black and isort)
make format

# Lint code
make lint
```

### Service Management
```bash
# Run individual services locally (requires activated venv)
make run-calendar   # Calendar API on :9001
make run-equity     # Equity API on :9002
make run-options    # Option history API on :9003
make run-chains     # Option chain API on :9004
make run-frontend   # Frontend on :9000
make run-orchestrator # Orchestrator API on :9005
make run-ib         # IB API on :9010
make run-flower     # Flower monitoring on :5555

# Docker compose for all services
make services-up    # Start all services
make services-down  # Stop all services
```

### Data Operations
```bash
# View upcoming earnings from local repository
make show-earnings

# Deduplicate earnings partitions
make dedup-earnings

# Consume option chain events (backfill option bars)
make consume-option-chains       # Production topic
make consume-option-chains-demo  # Demo topic

# Bootstrap data for symbols with earnings
make from-earnings      # Execute
make from-earnings-dry  # Dry run

# Start Celery worker for async tasks
make celery-worker

# Enqueue US contracts backfill
make backfill-contracts-us
```

## Architecture Overview

### Core Data Flow Pattern
This is a **local-first, event-driven** system:
1. **Producers** write data to Parquet repositories under `data/`
2. **Events** are published to file-based JSONL bus at `data/cache/events/`
3. **Consumers** read events, process data from repositories, write results
4. **Checkpoints** track processing state at `data/cache/state/`

### Critical Data Dependencies
**MUST follow this order for option data:**
```
Contract Descriptions → Option Chain Snapshots → Equity Bars (spot prices) → Option Bars
```

Key constraints:
- TWS API provides **snapshot-only** option chains (no historical chains)
- Option bars require contracts, chains, and spot prices to exist first
- All data writes are idempotent with deterministic deduplication

### Repository Structure
```
common/
├── repositories/      # Data persistence layer (Parquet)
├── utils/            # Event bus, checkpoints, logging
├── api/              # IB API core functionality
├── reliability/      # Circuit breakers, retry logic
└── monitoring/       # Dashboard and metrics

services/             # FastAPI microservices
├── calendar_api/     # Earnings calendar
├── equity_api/       # Equity bar data
├── option_history_api/ # Option historical data
├── option_chain_api/ # Option chain snapshots
├── orchestrator_api/ # High-level async workflows
└── frontend/         # Web UI

ops/
├── jobs/            # Batch processing jobs
├── scripts/         # Utility scripts
└── tasks/           # Celery async tasks

data/                # All persisted data (Parquet)
├── earnings/        # Earnings announcements
├── equity_bars/     # Stock price history
├── option_bars/     # Option price history
├── option_chains/   # Option chain snapshots
├── contracts/       # Contract descriptions
└── cache/          
    ├── events/      # JSONL event log
    └── state/       # Consumer checkpoints
```

### Event Bus Topics
- `earnings` - Earnings data available
- `equity_bars` - Equity bar data written
- `option_chains` - Option chain snapshots saved
- `option_bars` - Option bar history stored
- Add `.demo` suffix for demo/test data (e.g., `option_chains.demo`)

### Key Python Patterns

#### Repository Usage
```python
from common.repositories.equity_bars import EquityBarRepository

repo = EquityBarRepository("data")
# Check existing data first
present = repo.present_dates("AAPL", "1 day", start_date, end_date)
# Save new data
repo.save(bars_df, symbol="AAPL", bar_size="1 day")
```

#### Event Publishing
```python
from common.utils.events.bus import FileEventBus

bus = FileEventBus()
bus.notify_data_available(
    dataset="equity_bars",
    symbol="AAPL", 
    start="2024-01-01",
    end="2024-01-31",
    job_id="job-123"
)
```

#### Event Consumption
```python
from common.utils.events.consumer import JSONLConsumer
from common.utils.events.processed_index import ProcessedIndex

consumer = JSONLConsumer("option_chains")
index = ProcessedIndex("my-service", "option_chains")

for event in consumer.read(service="my-service", limit=100):
    key = generate_idempotent_key(event["payload"])
    if not index.is_processed(key):
        # Process event
        index.mark_processed(key)
```

### Environment Variables
- `APP_ENV` - Environment (dev/test/prod)
- `APP_LOG_LEVEL` - Logging level (DEBUG/INFO/WARNING/ERROR)
- `APP_CELERY__BROKER_URL` - Celery broker URL (default: amqp://)
- `APP_CELERY__RESULT_BACKEND` - Celery backend (default: rpc://)
- `ALLOW_PROD_WRITES` - Enable production data writes (default: 0)

### Testing Approach
- Tests use `pytest` with markers for categorization
- Coverage requirement: 80% (configured in pytest.ini)
- Test data isolation via fixtures in `conftest.py`
- Mock IB API interactions in `tests/mocks/`

### Common Development Tasks

#### Adding New Data Ingestion
1. Create repository class extending `BaseRepository`
2. Implement backfill job in `ops/jobs/`
3. Add event publishing after data write
4. Create consumer if downstream processing needed
5. Add tests with proper mocking

#### Debugging Event Processing
```bash
# Check event log
tail -f data/cache/events/option_chains.jsonl | jq .

# Check consumer checkpoint
cat data/cache/state/backfill-broker/option_chains.json | jq .

# Monitor correlation IDs for tracing
grep "correlation_id" data/logs/app.log
```

#### Working with Celery Tasks
```python
# Define task with rate limiting
from celery import shared_task

@shared_task(rate_limit="10/m", queue="contracts")
def fetch_contract_details(symbol: str):
    # Implementation
    pass

# Chain tasks for workflows
from celery import chain
workflow = chain(
    search_symbols.s(pattern),
    fetch_contract_descriptions.s()
)
workflow.apply_async()
```

#### Using Orchestrator API
The orchestrator provides high-level async workflows for data operations:

```bash
# Contract setup (search + details)
curl -X POST http://localhost:9005/orchestrator/contracts/search \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "IBM"]}'

curl -X POST http://localhost:9005/orchestrator/contracts/details \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "IBM"]}'

# Historical data fetching
curl -X POST http://localhost:9005/orchestrator/historical/bars \
  -H "Content-Type: application/json" \
  -d '{
    "contracts": [{
      "symbol": "AAPL",
      "secType": "STK", 
      "exchange": "SMART",
      "currency": "USD"
    }],
    "end_date_time": "20240825 23:59:59",
    "duration_str": "1 D",
    "bar_size_setting": "1 hour",
    "repository_type": "equity"
  }'

# Historical data by symbols (auto-resolves contracts)
curl -X POST http://localhost:9005/orchestrator/historical/bars-by-symbols \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "IBM"],
    "end_date_time": "20240825 23:59:59", 
    "duration_str": "5 D",
    "bar_size_setting": "1 day",
    "repository_type": "equity"
  }'
```

All orchestrator endpoints return immediately with tracking IDs while processing happens asynchronously via Celery tasks.

#### Job Monitoring with Flower

Monitor async job status and failures using Flower web UI:

```bash
# Start Flower monitoring
make run-flower

# Or via docker-compose (includes full stack)
docker-compose up flower
```

**Flower Web UI**: http://localhost:5555

Features:
- **Real-time task monitoring** - See running, failed, succeeded tasks
- **Task details** - View arguments, results, tracebacks, execution time
- **Worker monitoring** - Active workers, queues, rate limits
- **Failure debugging** - Exact error messages and stack traces

Browse to `/tasks` to see all job executions, including the historical data and contract search jobs from the orchestrator.

## API Services Architecture

### Service Separation - Write vs Read Operations

The system uses **complementary APIs** with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │────│  Equity API      │────│  Repository     │
│   Analytics     │    │  (Read Data)     │    │  (Parquet)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        ▲
                                                        │
                       ┌──────────────────┐    ┌───────┴─────────┐
                       │  Orchestrator    │────│  IB API         │
                       │  (Write Data)    │    │  (Live Data)    │
                       └──────────────────┘    └─────────────────┘
```

### **Equity API** (Port 9002) - Data Reading Service
**Purpose**: Fast retrieval of existing historical data from repository

#### Endpoints:
- **GET** `/health` - Service health check
- **GET** `/equity/{symbol}/dates?bar_size=1 day` - Get available dates for symbol
- **GET** `/equity/{symbol}/bars?start=2024-01-01&end=2024-12-31&bar_size=1 day` - Get historical bars

#### Schemas:
```python
class EquityDateRange(BaseModel):
    dates: List[date]

class EquityBarsResponse(BaseModel):
    symbol: str
    bar_size: str  
    start: date
    end: date
    rows: int
    sample: List[dict]  # First 50 rows as sample
```

#### Usage Examples:
```bash
# Check available data dates for AAPL
curl http://localhost:9002/equity/AAPL/dates?bar_size=1%20day

# Get historical bars for date range  
curl "http://localhost:9002/equity/AAPL/bars?start=2024-01-01&end=2024-01-31&bar_size=1%20day"
```

### **Orchestrator API** (Port 9005) - Data Ingestion Service
**Purpose**: Asynchronous data collection and workflow orchestration

#### Historical Data Endpoints:
- **POST** `/orchestrator/historical/bars-by-symbols` - Fetch data by ticker symbols (auto-resolves contracts)
- **POST** `/orchestrator/historical/bars` - Fetch data by explicit contract specifications
- **POST** `/contracts/search` - Search and save contract descriptions  
- **POST** `/contracts/details` - Fetch detailed contract information

#### Schemas:
```python
class HistoricalDataBySymbolsRequest(BaseModel):
    symbols: List[str]
    end_date_time: str  # "20240823 23:59:59"
    duration_str: str   # "1 D", "1 M", "1 Y", "2 W"
    bar_size_setting: str  # "1 min", "5 mins", "1 hour", "1 day"
    what_to_show: str = "TRADES"  # "TRADES", "MIDPOINT", "BID", "ASK"
    use_rth: int = 1  # 1=regular hours, 0=all hours
    repository_type: str = "equity"  # "equity" or "option"
    timeout_seconds: float = 20.0

class HistoricalDataRequest(BaseModel):
    contracts: List[Dict[str, Any]]  # IB contract specifications
    # ... same parameters as above

class OrchestratorResult(BaseModel):
    processed: int
    saved: int  
    skipped: int = 0
    details: Optional[dict] = None
    job_id: Optional[str] = None
    monitor_url: Optional[str] = None
```

#### Usage Examples:
```bash
# Fetch historical data by symbols (recommended)
curl -X POST http://localhost:9005/orchestrator/historical/bars-by-symbols \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "IBM"],
    "end_date_time": "20240825 23:59:59",
    "duration_str": "5 D", 
    "bar_size_setting": "1 day",
    "repository_type": "equity"
  }'

# Fetch by explicit contracts
curl -X POST http://localhost:9005/orchestrator/historical/bars \
  -H "Content-Type: application/json" \
  -d '{
    "contracts": [{
      "symbol": "AAPL",
      "secType": "STK",
      "exchange": "SMART", 
      "currency": "USD"
    }],
    "end_date_time": "20240825 23:59:59",
    "duration_str": "1 D",
    "bar_size_setting": "1 hour"
  }'
```

### **Recommended Workflow**:

1. **Data Collection**: Use `Orchestrator` to fetch new data from IB API
   - Async processing via Celery tasks
   - Rate limited (2 requests/sec for historical data)
   - Monitor progress via Flower UI

2. **Data Consumption**: Use `Equity API` for fast queries of existing data
   - Synchronous responses for immediate results
   - Optimized for frontend/analytics use cases

### **No Functional Overlap**: 
- **Orchestrator**: Write-heavy async operations (data ingestion)
- **Equity API**: Read-heavy sync operations (data retrieval)

### Important Configuration Files
- `app_settings/settings.py` - Central configuration using Pydantic
- `pytest.ini` - Test configuration and markers
- `pyproject.toml` - Black/isort formatting config
- `docker-compose.dev.yml` - Local development services
- `sitecustomize.py` - Global logging setup

### Safety and Best Practices
- Production paths protected by default (see `common/utils/safety.py`)
- Use `--demo` flag or set `ALLOW_PROD_WRITES=1` for production writes
- All repository writes are idempotent with deduplication
- Correlation IDs preserved across event chains for tracing
- Circuit breakers on external API calls (IB, web APIs)

### Logging and Monitoring
- Structured logging with correlation IDs
- Pretty console formatting with timestamps and colors
- Dashboard available at http://localhost:9090 (when running)
- Events UI with correlation filtering for debugging