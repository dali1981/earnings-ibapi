# Celery Tasks Structure

This directory contains Celery tasks organized by functionality:

## Task Files

### `contracts.py`
- **Purpose**: Legacy contract backfill tasks
- **Queue**: `contracts`
- **Rate Limit**: 10/m
- **Tasks**:
  - `contracts.backfill_fetch`: Fetch contracts for backfill

### `search.py` 
- **Purpose**: Symbol search and contract descriptions
- **Queue**: `symbol-search`
- **Tasks**:
  - `search.fetch_matching_symbols`: Search for matching symbols via IB API
  - `search.save_contract_descriptions`: Save search results to repository

### `contract_details.py`
- **Purpose**: Contract details fetching and storage
- **Queue**: `contracts`
- **Rate Limit**: 10/m
- **Tasks**:
  - `contracts.check_existing_details`: Check which symbols already have details in repo
  - `contracts.fetch_details_from_ib`: Fetch missing details from IB API
  - `contracts.save_details_to_repo`: Save details to contracts repository

### `historical_data.py`
- **Purpose**: Historical bars data fetching and storage
- **Queue**: `historical`
- **Rate Limit**: 2/s (IB API historical data limit)
- **Tasks**:
  - `historical.fetch_bars_from_ib`: Fetch historical bars from IB API
  - `historical.save_bars_to_repo`: Save bars to equity_bars or option_bars repository
  - `historical.resolve_contracts_for_symbols`: Resolve contract details for symbols

## Task Flow Patterns

### Contract Setup Flow
1. **Search**: `search.fetch_matching_symbols` → `search.save_contract_descriptions`
2. **Details**: `contracts.check_existing_details` → `contracts.fetch_details_from_ib` → `contracts.save_details_to_repo`

### Historical Data Flow
1. **By Contracts**: `historical.fetch_bars_from_ib` → `historical.save_bars_to_repo`
2. **By Symbols**: `historical.resolve_contracts_for_symbols` → (fetch/save for each resolved contract)

## Queue Configuration

```toml
[celery.contracts]
fetch_rate = "10/m"
queue = "contracts"

[celery.search]
queue = "symbol-search"

[celery.historical]
fetch_rate = "2/s"
queue = "historical"
```

## Usage Examples

### Start Celery Worker
```bash
# All queues
celery -A ibx_celery.app worker --loglevel=INFO

# Specific queues
celery -A ibx_celery.app worker --loglevel=INFO -Q contracts,historical,symbol-search
```

### API Usage
See orchestrator API endpoints:
- `/orchestrator/contracts/search` - Search and save contract descriptions
- `/orchestrator/contracts/details` - Fetch and save contract details
- `/orchestrator/historical/bars` - Fetch historical bars by contracts
- `/orchestrator/historical/bars-by-symbols` - Fetch historical bars by symbols

## Rate Limiting

Tasks are rate limited to comply with IB API restrictions:
- **Contract details**: 10 requests/minute
- **Symbol search**: 10 requests/second
- **Historical data**: 2 requests/second

## Data Flow

All tasks follow the event-driven pattern:
1. **Fetch** data from IB API
2. **Process** and normalize data
3. **Save** to appropriate repository (Parquet files)
4. **Optionally publish** events for downstream consumers