# Earnings Fetcher API Reference

## Class: EarningsCalendarFetcher

Multi-source earnings data fetcher with automatic fallback mechanisms.

### Constructor

```python
EarningsCalendarFetcher(cache_dir: Path = Path("cache/earnings"))
```

**Parameters:**
- `cache_dir`: Directory for caching earnings data

### Methods

#### get_upcoming_earnings

```python
get_upcoming_earnings(
    symbols: Optional[List[str]] = None,
    days_ahead: int = 30,
    sources: List[EarningsSource] = None
) -> List[EarningsEvent]
```

Fetch upcoming earnings events with multi-source fallback.

**Parameters:**
- `symbols`: List of stock symbols to filter (None = all symbols)
- `days_ahead`: Number of days ahead to fetch
- `sources`: List of data sources to try (defaults to FMP → Finnhub → NASDAQ)

**Returns:**
List of `EarningsEvent` objects sorted by priority

**Example:**
```python
fetcher = EarningsCalendarFetcher()

# Get all earnings for next 14 days
all_earnings = fetcher.get_upcoming_earnings(days_ahead=14)

# Get specific symbols only
tech_earnings = fetcher.get_upcoming_earnings(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    days_ahead=21
)

# Use only NASDAQ source
nasdaq_earnings = fetcher.get_upcoming_earnings(
    sources=[EarningsSource.NASDAQ],
    days_ahead=30
)
```

#### save_to_cache / load_from_cache

```python
save_to_cache(earnings: List[EarningsEvent], cache_key: str = None)
load_from_cache(cache_key: str = None, max_age_hours: int = 6) -> Optional[List[EarningsEvent]]
```

Cache management for earnings data.

## Class: EarningsEvent

Single earnings event data structure.

### Properties

```python
@dataclass
class EarningsEvent:
    symbol: str
    company_name: str
    earnings_date: date
    time: Optional[str] = None  # "bmo", "amc", "dmt"
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    source: str = "unknown"
```

#### Computed Properties

**days_until_earnings**
```python
@property
def days_until_earnings(self) -> int
```
Returns number of days until earnings (negative if in past).

**priority_score**
```python
@property
def priority_score(self) -> float
```
Returns priority score for scheduling (higher = more urgent).

Priority scoring:
- Today or past: 100.0
- Next 3 days: 90.0
- Next 7 days: 80.0
- Next 14 days: 60.0
- Next 30 days: 40.0
- Future: 20.0

## Enums

### EarningsSource

```python
class EarningsSource(Enum):
    FMP = "financial_modeling_prep"
    FINNHUB = "finnhub"
    NASDAQ = "nasdaq"
    YAHOO = "yahoo"
```

### Timing Values

- `"bmo"`: Before market open
- `"amc"`: After market close  
- `"dmt"`: During market hours
- `"unknown"`: Timing not specified

## Error Handling

The fetcher implements robust error handling:

1. **Source Fallback**: If primary source fails, automatically tries secondary sources
2. **Network Timeouts**: 30-second timeout on all requests
3. **Rate Limiting**: Respectful delays between API calls
4. **Data Validation**: Validates and cleans all incoming data

## Configuration

### API Keys (Optional)

Set environment variables for premium features:

```bash
export FMP_API_KEY=your_financial_modeling_prep_key
export FINNHUB_API_KEY=your_finnhub_key
```

**Note:** System works without API keys using NASDAQ fallback.

### Custom Configuration

```python
fetcher = EarningsCalendarFetcher()

# Modify API configurations
fetcher.api_configs[EarningsSource.FMP]['api_key'] = 'your_key'

# Add custom headers
fetcher.session.headers.update({
    'Custom-Header': 'value'
})
```

## NASDAQ API Details

The NASDAQ source provides comprehensive free coverage:

### Endpoint
```
https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD
```

### Response Format
```json
{
  "data": {
    "rows": [
      {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "marketCap": "$2,500,000,000,000",
        "time": "time-after-hours",
        "epsForecast": "$1.35",
        "noOfEsts": "12"
      }
    ]
  }
}
```

### Market Cap Parsing

The fetcher automatically parses market cap strings:
- `"$2.5T"` → `2.5e12`
- `"$150B"` → `1.5e11`
- `"$500M"` → `5e8`

## Performance Characteristics

### Throughput
- **NASDAQ**: ~300 earnings/second
- **Network bound**: Primary bottleneck is API response time
- **Memory**: ~1MB per 1000 earnings events

### Caching
- **Default TTL**: 6 hours
- **Storage**: JSON format in `cache/earnings/`
- **Automatic cleanup**: Old cache files removed

### Rate Limits
- **NASDAQ**: No explicit limit (respectful 0.3s delays)
- **FMP Free**: 250 requests/day
- **Finnhub Free**: 60 requests/minute

## Advanced Usage

### Custom Date Ranges
```python
from datetime import date, timedelta

# Fetch earnings for specific date range
start_date = date(2025, 8, 15)
end_date = date(2025, 8, 20)

earnings = []
current_date = start_date
while current_date <= end_date:
    daily_earnings = fetcher._fetch_nasdaq_single_day(current_date, None)
    earnings.extend(daily_earnings)
    current_date += timedelta(days=1)
```

### Market Cap Filtering
```python
# Filter for large cap stocks only
large_cap_earnings = []
for event in all_earnings:
    if hasattr(event, 'market_cap') and event.market_cap:
        if event.market_cap > 10e9:  # $10B+
            large_cap_earnings.append(event)
```

### Data Quality Scoring
```python
def calculate_data_quality(event: EarningsEvent) -> float:
    """Calculate data quality score for an earnings event."""
    score = 0.0
    
    # Has EPS estimate
    if event.eps_estimate is not None:
        score += 0.3
    
    # Has timing information
    if event.time and event.time != 'unknown':
        score += 0.2
    
    # Has market cap (NASDAQ only)
    if hasattr(event, 'market_cap') and event.market_cap:
        score += 0.2
    
    # Recent/upcoming (more reliable)
    days_until = abs(event.days_until_earnings)
    if days_until <= 7:
        score += 0.3
    elif days_until <= 30:
        score += 0.2
    
    return score
```

## Integration Examples

### With Discovery Engine
```python
from earnings_trading.discovery import EarningsDiscoveryEngine

# Use custom fetcher in discovery engine
engine = EarningsDiscoveryEngine()
engine.earnings_fetcher = EarningsCalendarFetcher(cache_dir=Path("custom_cache"))

opportunities = engine.discover_earnings_opportunities(days_ahead=14)
```

### With Data Pipeline
```python
from earnings_trading.data_pipeline import EarningsDataPipeline

# Priority-based data collection
pipeline = EarningsDataPipeline()
results = pipeline.run_earnings_driven_collection(
    days_ahead=21,
    min_opportunity_score=60.0
)
```