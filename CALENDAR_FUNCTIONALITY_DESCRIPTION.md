# Calendar API Functionality Description

## Overview

The Calendar API service is a FastAPI-based microservice that provides comprehensive earnings calendar functionality for the trading system. It serves as the central hub for earnings data collection, storage, retrieval, and management with support for multiple data sources and real-time fetching capabilities.

## Architecture

### Service Structure
```
services/calendar_api/
├── app.py                    # FastAPI application initialization
├── main.py                   # Entry point
├── routers/
│   └── calendar.py           # REST API endpoints
├── schemas/
│   └── calendar.py           # Pydantic models for API
├── service/
│   ├── calendar_service.py   # Core business logic
│   ├── earnings_service.py   # Legacy earnings service
│   └── earnings_fetcher_service.py  # External data fetching
└── examples/
    └── calendar_ui.html      # Testing interface
```

### Data Flow
```
External Sources → Fetcher Services → Calendar Service → Repository → Parquet Storage
                                          ↓
                              Event Bus (earnings.jsonl)
                                          ↓
                              Downstream Consumers
```

## Core Components

### 1. Calendar Service (`calendar_service.py`)
**Primary business logic layer** that orchestrates earnings data operations:

- **Data Retrieval**: Queries upcoming earnings from the repository with filtering support
- **Event Storage**: Handles batch saving with deduplication and presence checking
- **Freshness Management**: Tracks data collection timestamps and cache validity
- **Event Publishing**: Publishes data availability events via FileEventBus

**Key Methods**:
- `get_upcoming_events(days_ahead, symbols)` - Retrieves filtered earnings from repository
- `save_events(events)` - Saves earnings with deduplication and event publishing
- `is_fresh()` - Checks if data is within configured cache window
- `get_latest_collection_date_str()` - Returns most recent collection date

### 2. Earnings Fetcher Service (`earnings_fetcher_service.py`)
**External data acquisition layer** that fetches earnings from multiple sources:

- **Multi-Source Support**: NASDAQ, Yahoo Finance, and "all" source combinations
- **Async Operations**: Uses AsyncEarningsCalendarFetcher for concurrent fetching
- **Enhanced Logging**: Detailed logging for fetch operations and error handling
- **Symbol Validation**: Reports found/missing symbols when specific symbols requested

**Key Features**:
- Source-specific handling (Yahoo requires symbols parameter)
- Concurrent fetching from multiple sources
- Error handling with detailed logging
- Symbol availability reporting

### 3. Repository Layer (`common/repositories/earnings.py`)
**Persistent storage layer** using PyArrow/Parquet with advanced features:

**Storage Strategy**:
- **Partitioning**: `collection_date=YYYY-MM-DD/source=nasdaq/earnings.parquet`
- **Deduplication**: By `(symbol, earnings_date, source)` key
- **Metadata Tracking**: Storage metadata with quality scores and lineage
- **Retention Policy**: Configurable data retention (default 365 days)

**Key Features**:
- Date-range querying with partition optimization
- Historical symbol earnings lookup
- Storage statistics and health monitoring
- Atomic writes with deduplication

### 4. Data Models (`earnings/models.py` & `schemas/calendar.py`)
**Type-safe data structures** for earnings events:

**Core Models**:
- `EarningsEvent`: Core earnings event with symbol, date, estimates/actuals
- `EarningsEventModel`: API response model with validation
- `EarningsQueryMetadata`: Query metadata for enhanced UX
- `EarningsResponseWithMetadata`: Enhanced response with suggestions

**Validation Features**:
- NaN/Infinity handling for JSON compatibility
- Enum normalization for storage
- Float field validation and sanitization

## API Endpoints

### Read Operations
- `GET /health` - Service health check
- `GET /calendar/upcoming` - Read earnings from repository with metadata
- `GET /calendar/get-upcoming` - Simple repository read (legacy compatibility)
- `GET /calendar/present` - Check if specific event exists
- `GET /calendar/symbol/{symbol}` - Historical earnings for symbol
- `GET /calendar/stats` - Repository storage statistics
- `GET /calendar/data-availability` - Data availability analysis
- `GET /calendar/suggest-symbols` - Symbol suggestions for upcoming earnings

### Write Operations
- `POST /calendar/fetch-upcoming` - Fetch from external sources with source selection
- `POST /calendar/fetch-upcoming/yahoo` - Yahoo-specific fetch (requires symbols)
- `POST /calendar/events` - Bulk add events with deduplication

### Query Parameters
- `days`: Look-ahead window (0-180 days, default 14)
- `symbols`: Comma-separated symbol list (optional)
- `source`: Data source selection (nasdaq/yahoo/all, default nasdaq)
- `days_back`: Historical lookback (for symbol queries, default 365)

## Data Sources Integration

### NASDAQ Source
- **Default source** for general calendar queries
- **No symbol requirement** - can fetch all upcoming earnings
- **High data quality** with comprehensive coverage
- **Rate limited** to prevent API abuse

### Yahoo Finance Source
- **Symbol-specific queries** - requires symbols parameter
- **Realistic headers required** for API calls
- **Per-symbol lookup supported**
- **Bulk calendar queries** for date ranges

### Multi-Source ("all")
- **Concurrent fetching** from multiple sources
- **Automatic deduplication** with quality preference
- **Source-specific error handling**
- **Consolidated results** with source attribution

## Storage and Caching

### Parquet Storage Structure
```
data/earnings/
├── collection_date=2025-01-15/
│   ├── source=nasdaq/
│   │   └── earnings.parquet
│   ├── source=yahoo/
│   │   └── earnings.parquet
│   └── _metadata.json
└── collection_date=2025-01-16/
    └── source=nasdaq/
        └── earnings.parquet
```

### Data Deduplication
- **Primary Key**: `(symbol, earnings_date, source)`
- **Merge Strategy**: Read existing partitions, deduplicate, atomic rewrite
- **Quality Preference**: Prioritizes events with more complete data

### Freshness Management
- **Cache Window**: Configurable hours (default 6 hours)
- **Collection Metadata**: Tracks storage timestamps
- **Freshness Check**: Based on `_metadata.json` timestamps
- **Daily Collections**: Same-day collections considered fresh

## Event Bus Integration

### Event Publishing
- **Topic**: `earnings` (with `.demo` suffix for test data)
- **Trigger**: Successful data storage operations
- **Payload**: Collection date, event count, sources, symbols
- **Consumer Support**: JSONL-based event log with checkpoints

### Event Schema
```json
{
  "dataset": "earnings",
  "collection_date": "2025-01-15",
  "total_events": 150,
  "sources": ["nasdaq", "yahoo"],
  "symbols": ["AAPL", "MSFT", "..."],
  "job_id": "earnings-fetch-123",
  "correlation_id": "cal-456"
}
```

## Configuration and Environment

### Configuration Sources
- `config.EARNINGS_CONFIG` - Earnings-specific settings
- `config.EARNINGS_PATH` - Storage path configuration
- Environment variables for service settings

### Key Settings
- `cache_hours`: Data freshness window (default 6)
- `retention_days`: Data retention policy (default 365)
- `APP_LOG_LEVEL`: Logging verbosity
- `APP_ENV`: Environment (dev/test/prod)

## Error Handling and Logging

### Comprehensive Logging
- **Structured logging** with correlation IDs
- **Performance metrics** (fetch times, storage durations)
- **Data quality metrics** (events found/missing, deduplication counts)
- **Error details** with stack traces for debugging

### Error Recovery
- **Graceful degradation** when external sources fail
- **Partial success handling** for multi-source operations
- **Data validation** with detailed error messages
- **Persistence error isolation** (fetch succeeds even if storage fails)

## Testing and Validation

### Test Coverage
- **Unit tests**: Service logic, repository operations, API endpoints
- **Integration tests**: End-to-end calendar workflows
- **Cache tests**: Freshness and metadata handling
- **Router tests**: API endpoint validation and error handling

### Data Quality Validation
- **Schema validation** using Pydantic models
- **Date range validation** for earnings dates
- **Symbol format validation** and normalization
- **Estimate/actual value validation** with NaN handling

## Performance Characteristics

### Query Performance
- **Partition pruning** for date-range queries
- **Columnar storage** with PyArrow for efficient filtering
- **Metadata caching** for quick availability checks
- **Symbol indexing** for fast symbol-specific queries

### Storage Efficiency
- **Compression**: Parquet format with optimal compression
- **Deduplication**: Reduces storage overhead
- **Partition management**: Efficient date-based partitioning
- **Cleanup automation**: Retention policy enforcement

## Frontend Integration

### API Client Support
- **Consistent response formats** with metadata
- **CORS enabled** for local development and testing
- **Response headers** indicating data source and collection date
- **Error handling** with detailed error messages

### UI Components
- **Calendar visualization** in Next.js frontend
- **Symbol search and suggestions**
- **Date range filtering**
- **Source selection interface**

## Dependencies and External Services

### Core Dependencies
- **FastAPI**: Web framework with automatic API documentation
- **PyArrow/Parquet**: High-performance columnar storage
- **Pandas**: Data manipulation and analysis
- **Pydantic**: Data validation and serialization
- **asyncio**: Asynchronous programming support

### External APIs
- **NASDAQ API**: Primary earnings data source
- **Yahoo Finance API**: Secondary earnings data source
- **Rate limiting**: Respectful API usage patterns

## Operational Considerations

### Monitoring
- **Health checks** with dependency validation
- **Storage statistics** for capacity planning
- **Fetch success rates** and error tracking
- **Query performance metrics**

### Deployment
- **Container ready** with proper logging configuration
- **Environment-specific** configuration support
- **Graceful shutdown** handling
- **Resource usage optimization**

### Maintenance
- **Automated cleanup** of old data
- **Data quality monitoring**
- **Source availability tracking**
- **Performance baseline monitoring**

This Calendar API service provides a robust, scalable foundation for earnings data management in the trading system, with comprehensive features for data acquisition, storage, retrieval, and real-time processing.