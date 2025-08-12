# System Design - Earnings-Driven Options Trading

## Architecture Overview

The earnings-driven options trading system follows a modular architecture designed for scalability, reliability, and maintainability.

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────┐│
│  │     FMP     │  │   Finnhub   │  │   NASDAQ    │  │Yahoo ││
│  │ (Premium)   │  │ (Secondary) │  │ (Fallback)  │  │ (ER) ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────┘│
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                   EARNINGS DISCOVERY LAYER                  │
├─────────────────────────────────────────────────────────────┤
│  Market Scanner → Filters → Strategy Scoring → Prioritization│
│       │             │            │               │          │
│   ALL Events    Market Cap   Calendar/Straddle  HIGH/MED/LOW│
│   (1,122+)      Volume       Strangle Scoring      Groups   │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                   DATA COLLECTION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  Priority Queue → IB TWS API → Data Storage → Lineage Track │
│       │              │             │              │         │
│   Opportunity    Options Chains  PyArrow/Parquet  Metadata  │
│    Driven        Equity Bars     Repositories     Tracking  │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                   RELIABILITY LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  Circuit Breakers → Retry Logic → Rate Limiting → Monitoring│
│       │                │             │              │       │
│   Auto Recovery    Exponential    API Protection   Logging   │
│   Mechanisms       Backoff        Mechanisms       Alerts    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Earnings Discovery Engine
**Location**: `earnings_trading/discovery.py`

```python
class EarningsDiscoveryEngine:
    """Market-wide earnings opportunity discovery"""
    
    def discover_earnings_opportunities(self, days_ahead: int) -> List[EarningsCandidate]:
        # Step 1: Get ALL market earnings
        all_earnings = self.earnings_fetcher.get_upcoming_earnings(symbols=None)
        
        # Step 2: Apply basic filters
        candidates = self._filter_for_options_trading(all_earnings)
        
        # Step 3: Score for strategies
        self._score_options_strategies(candidates)
        
        # Step 4: Prioritize and return
        return self._prioritize_candidates(candidates)
```

**Key Features**:
- Market-wide scanning (no fixed portfolios)
- Multi-strategy scoring (calendar, straddle, strangle)
- Intelligent filtering based on options trading requirements
- Priority-based output for data collection

### 2. Multi-Source Data Fetcher
**Location**: `earnings/fetcher.py`

```python
class EarningsCalendarFetcher:
    """Robust multi-source earnings data acquisition"""
    
    def get_upcoming_earnings(self, sources: List[EarningsSource]) -> List[EarningsEvent]:
        for source in sources:
            try:
                earnings = self._fetch_from_source(source)
                if earnings:
                    return self._deduplicate_earnings(earnings)
            except Exception as e:
                logger.warning(f"Source {source} failed: {e}")
                continue
```

**Fallback Hierarchy**:
1. **FMP** (Primary) - Premium features, detailed data
2. **Finnhub** (Secondary) - Good coverage, rate limited
3. **NASDAQ** (Active) - Free, reliable, market-wide coverage
4. **Yahoo** (Emergency) - Scraping fallback

### 3. Dynamic Data Pipeline
**Location**: `earnings_trading/data_pipeline.py`

The pipeline adapts data collection based on discovered opportunities rather than fixed symbol lists.

```python
def run_earnings_driven_collection(self):
    # Step 1: Discover current opportunities
    opportunities = self.discovery_engine.discover_earnings_opportunities()
    
    # Step 2: Prioritize symbols based on earnings proximity and strategy scores
    priority_groups = self._group_by_priority(opportunities)
    
    # Step 3: Collect data for highest priority symbols first
    for priority, symbols in priority_groups.items():
        self._collect_data_for_symbols(symbols, priority)
```

## Data Flow Architecture

### Input Phase
1. **Market Scanning**: NASDAQ API provides comprehensive earnings calendar
2. **Data Normalization**: Convert various API formats to standardized `EarningsEvent`
3. **Deduplication**: Remove duplicate events from multiple sources

### Processing Phase
1. **Basic Filtering**: 
   - Market cap > $1B
   - Symbol length < 5 characters
   - No special characters (warrants, rights)
   - Reasonable timing windows (1-60 days)

2. **Strategy Scoring**:
   ```python
   def _score_options_strategies(self, candidate):
       base_score = self._calculate_timing_score(candidate)
       candidate.calendar_score = base_score * self._calendar_multiplier(candidate)
       candidate.straddle_score = base_score * self._straddle_multiplier(candidate)  
       candidate.strangle_score = base_score * self._strangle_multiplier(candidate)
   ```

3. **Priority Classification**:
   - **CRITICAL**: Score 90+, earnings ≤3 days
   - **HIGH**: Score 75+, earnings ≤7 days
   - **MEDIUM**: Score 60+, earnings ≤14 days  
   - **LOW**: Score 50+, earnings >14 days

### Output Phase
1. **CSV Export**: Human-readable opportunity list with action items
2. **Priority Lists**: Structured data for automated collection
3. **Lineage Tracking**: Metadata about data sources and processing steps

## Reliability & Error Handling

### Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
```

**States**:
- **CLOSED**: Normal operation
- **OPEN**: Failing fast, not attempting calls  
- **HALF_OPEN**: Testing recovery

### Retry Logic
```python
@retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0
)
def fetch_earnings_data():
    # Network call with automatic retry
```

### Rate Limiting
- **NASDAQ**: 0.3 second delays between requests
- **FMP**: 250 requests/day limit tracking
- **Finnhub**: 60 requests/minute limit tracking

## Storage Architecture

### Repository Pattern
Each data type has a dedicated repository with standardized interfaces:

```python
class EquityBarsRepository(BaseRepository):
    def store(self, symbol: str, bars: List[Bar], **metadata):
        # Store with automatic lineage tracking
        
class OptionChainsRepository(BaseRepository):  
    def store(self, underlying: str, chains: List[OptionChain], **metadata):
        # Store with partitioning by expiry date
```

### Partitioning Strategy
- **Equity Bars**: `symbol=AAPL/trade_date=2025-08-12/`
- **Option Chains**: `underlying=AAPL/snapshot_date=2025-08-12/`
- **Option Bars**: `underlying=AAPL/expiry=20250815/trade_date=2025-08-12/`

## Configuration Management

### Environment-Based Configuration
```python
@dataclass
class EarningsConfig:
    min_market_cap: float = 1e9
    min_avg_volume: float = 1e6
    min_price: float = 10.0
    max_price: float = 1000.0
    min_days_ahead: int = 1
    max_days_ahead: int = 60
```

### API Configuration
```python
api_configs = {
    EarningsSource.NASDAQ: {
        "base_url": "https://api.nasdaq.com/api/calendar/earnings",
        "rate_limit": None,  # Free tier
        "headers": {"User-Agent": "Trading-System/1.0"}
    }
}
```

## Performance Characteristics

### Scalability Metrics
- **Processing Speed**: 1,122 earnings events in <30 seconds
- **Memory Usage**: ~150MB for full market scan
- **Storage Efficiency**: Parquet compression ~10:1 ratio
- **Throughput**: 40+ symbols/minute data collection

### Bottleneck Analysis
1. **Network I/O**: API calls are primary bottleneck
2. **Data Processing**: Filtering and scoring are CPU-bound but fast
3. **Storage I/O**: PyArrow/Parquet writes are optimized

## Security Considerations

### API Key Management
- Environment variable storage
- No hardcoded credentials
- Graceful fallback when keys unavailable

### Data Privacy
- No personally identifiable information stored
- Public market data only
- Compliance with data source terms of service

### Network Security
- HTTPS for all API calls
- User-Agent headers for identification
- Respectful rate limiting

## Monitoring & Observability

### Logging Strategy
```python
logger.info(f"📅 Found {len(all_earnings)} total earnings events")
logger.info(f"📊 {len(candidates)} candidates passed basic filters")  
logger.info(f"🎯 {len(scored_candidates)} high-quality candidates found")
```

### Key Metrics
- **Data Freshness**: Last successful API call timestamp
- **Success Rates**: API call success percentages by source
- **Processing Time**: End-to-end pipeline duration
- **Opportunity Count**: Number of trading candidates found

### Alerting
- Circuit breaker state changes
- API quota approaching limits  
- Unusual data patterns (very few/many opportunities)
- System errors requiring intervention

This architecture provides a robust foundation for earnings-driven options trading with built-in reliability, scalability, and maintainability.