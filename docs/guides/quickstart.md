# Quick Start Guide

Get up and running with the earnings-driven options trading system in minutes.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Internet connection (for earnings data)

### Install Dependencies
```bash
cd earnings_ibapi
pip install -r requirements.txt
```

### Optional: Set API Keys
```bash
# Optional - system works without these using NASDAQ fallback
export FMP_API_KEY=your_financial_modeling_prep_key
export FINNHUB_API_KEY=your_finnhub_key
```

## ⚡ Quick Start - 3 Commands

### 1. Discover Earnings (Pure Discovery)
```bash
python -m earnings.discovery
```

Expected output highlights:
- Raw events stored under `data/earnings/collection_date=YYYY-MM-DD/source=<provider>/earnings_*.parquet`
- Earning-date CSV exported to `data/exports/earnings_discoveries_YYYY-MM-DD.csv`

### 2. Test Earnings Data Fetcher
```bash
python -m earnings.fetcher
```

### 3. Compare Data Sources (optional)
```bash
python ops/scripts/compare_earnings_sources.py
```

## 📊 Understanding the Results (pure collection)

- CSV columns: `symbol`, `company_name`, `earnings_date`, `time`, `days_until`, `market_cap`, `eps_estimate`, `quality_score`, `discovery_date`, `source`.
- Sorting: by `days_until` asc and `market_cap` desc.
- No filtering: everything collected is stored (parquet) and exported (CSV).
- Recent data: use `EarningsDiscoveryEngine.get_recent_discoveries(max_age_days=1)`.

Time field
- Normalization: Recognized values are normalized to `bmo` (before market open) or `amc` (after market close).
- Preservation: Otherwise the provider label is preserved (e.g., `dmt`, `during market`); if missing, `unknown`.

### (removed)
- **100 pts**: Excellent opportunity, earnings within 7 days
- **80 pts**: Very good opportunity, good timing
- **60 pts**: Good opportunity, reasonable timing
- **50 pts**: Consider opportunity, adequate timing

### (removed)
- **strangle**: High volatility expected, suitable for directional moves
- **calendar_spread**: Time decay play, good for range-bound movement
- **straddle**: Maximum volatility play, best for binary events

### (removed)
- **HIGH**: Earnings within 7 days, score 75+
- **MEDIUM**: Earnings within 14 days, score 60+
- **LOW**: Earnings >14 days, score 50+

## 🎯 Basic Usage Examples

### Programmatic Discovery
```python
from earnings.discovery import EarningsDiscoveryEngine

# Initialize engine
engine = EarningsDiscoveryEngine()

# Discover and store earnings (pure discovery)
discoveries = engine.discover_and_store_earnings(days_ahead=21, days_back=0)

# Load recent discoveries as a DataFrame
df = engine.get_recent_discoveries(max_age_days=1)
print(df.head())
```

### Custom Filtering
```python
# Filter for specific market cap range
engine.min_market_cap = 5e9  # $5B minimum
engine.discovery_max_days = 14   # Only next 2 weeks

discoveries = engine.discover_and_store_earnings(days_ahead=14)
df = engine.get_recent_discoveries(max_age_days=1)
```

### Export to CSV
Discovery automatically exports a daily CSV to `data/exports/earnings_discoveries_YYYY-MM-DD.csv`.

## 🔍 Exploring Results

### View Exported CSV
```bash
# Open exported CSV
open data/exports/earnings_discoveries_2025-08-12.csv

# Or view in terminal
head -10 data/exports/earnings_discoveries_2025-08-12.csv
```

CSV sample columns:
```
symbol,company_name,earnings_date,time,days_until,market_cap,eps_estimate,quality_score,discovery_date,source
AAPL,Apple Inc.,2025-08-20,amc,8,3.2e12,2.50,80.0,2025-08-12,nasdaq
```

## 🛠️ Configuration

### Basic Settings (earnings/discovery.py)
```python
# Market filters
self.min_market_cap = 1e9      # $1B minimum
self.min_avg_volume = 1e6      # 1M shares volume
self.min_price = 10.0          # $10 minimum stock price
self.max_price = 1000.0        # $1000 maximum stock price

# Timing windows  
self.discovery_min_days = 1        # At least 1 day ahead
self.discovery_max_days = 60       # Within 2 months
```

## 🔧 Bootstrap From Earnings

### What is a Dry Run?
A "dry run" is a no-op preview that plans and prints what would happen without making any external requests or writing data:
- Reads latest earning-date CSV to list target symbols
- Prints planned steps per symbol (contracts → chains → equity → options)
- Shows key parameters (lookback window, bar sizes, symbol list)
- Makes NO IB Gateway/TWS calls
- Writes NO data to filesystem

### Bootstrap Data Collection
```bash
# Preview what will be collected (dry run)
make from-earnings-dry

# Actually run the bootstrap
make from-earnings

# Custom bootstrap with specific parameters
python ops/jobs/run_from_earnings.py --max-symbols 10 --lookback-days 365 --option-bar-size "1 min"

# Bootstrap specific symbols
python ops/jobs/run_from_earnings.py --symbols AAPL,MSFT,NVDA --lookback-days 180
```

### Bootstrap Process
For each symbol, the bootstrap will:
1. **Contracts**: Targeted backfill if missing
2. **Chain snapshot**: Refresh if >24h old  
3. **Equity bars**: Backfill 1-day bars over lookback window
4. **Option bars**: Backfill 1-minute bars (default) over lookback window

### Data Storage Locations
- Contracts: `data/contracts/`
- Option chains: `data/option_chains/`
- Equity bars: `data/equity_bars/`
- Option bars: `data/option_bars/`

## 📈 Next Steps (optional)

### 1. Data Collection Setup
Once you have identified opportunities, set up data collection:
```bash
# Run comprehensive data pipeline
# optional later when trading is enabled
# python -m jobs.run_earnings_pipeline
```

### 2. Options Chain Analysis
For detailed options analysis:
```bash
# Get option chains for top opportunities
# optional later when trading is enabled
# python ops/examples/complete_workflow_example.py
```

### 3. Historical Analysis
Compare current opportunities with historical patterns:
```bash
# requires enabling extra sources via EARNINGS_ALLOWED_SOURCES
python ops/scripts/compare_earnings_sources.py --compare-historical 30
```

## 🚨 Troubleshooting

### Common Issues

**No earnings data found**
```bash
# Check your internet connection
ping api.nasdaq.com

# Verify the system can access NASDAQ API
curl "https://api.nasdaq.com/api/calendar/earnings?date=2025-08-12"
```

**Very few opportunities found**
- Lower the `min_score` threshold to 40.0
- Increase `days_ahead` to 30 or 60 days
- Check if it's a weekend (markets closed, less earnings data)

**Memory issues with large datasets**
- Reduce `days_ahead` parameter
- Add `max_symbols` limit in discovery engine
- Process in smaller chunks

### Debugging Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run discovery with detailed logging
engine = EarningsDiscoveryEngine()
opportunities = engine.discover_earnings_opportunities(days_ahead=7)
```

### Performance Tuning
```python
# For faster processing with fewer results
opportunities = engine.discover_earnings_opportunities(
    days_ahead=7,      # Shorter time window
    min_score=75.0     # Higher quality threshold
)
```

## ✅ Validation

### Verify Installation
```bash
# Test all major components
python -c "from earnings_trading.discovery import EarningsDiscoveryEngine; print('✅ Discovery engine loaded')"
python -c "from earnings.fetcher import EarningsCalendarFetcher; print('✅ Earnings fetcher loaded')" 
python -c "import pandas as pd; print('✅ Pandas available')"
```

### Test Data Pipeline
```bash
# Run a quick test with minimal data
python -c "
from earnings_trading.discovery import EarningsDiscoveryEngine
engine = EarningsDiscoveryEngine()
opps = engine.discover_earnings_opportunities(days_ahead=3, min_score=40.0)
print(f'✅ Found {len(opps)} opportunities in next 3 days')
"
```

You're now ready to discover earnings-driven options trading opportunities! 🚀

**Next recommended reading**: [Discovery Engine Guide](discovery-engine.md)
