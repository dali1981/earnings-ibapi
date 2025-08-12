# Earnings-Driven Options Trading System

🎯 **Automated market-wide earnings discovery and options trading opportunities identification system**

This system scans **ALL** upcoming earnings across the market, analyzes options strategies (calendar spreads, strangles, straddles), and prioritizes data collection for the most promising opportunities.

## 🚀 Quick Start

```bash
# Run pure earnings discovery (NASDAQ-only by default)
python -m earnings.discovery

# Optional: compare sources (if enabled via env)
python scripts/compare_earnings_sources.py
```

## 📊 Key Features

- **Market-Wide Earnings Discovery**: Scans market using the free NASDAQ API by default
- **Options Strategy Analysis**: Optional; trading layer is decoupled and can be run later
- **Dynamic Symbol Selection**: Opportunity-driven selection (when using trading layer)
- **Configurable Sources (Opt-in)**: Additional sources can be enabled via `EARNINGS_ALLOWED_SOURCES`
- **Priority-Based Collection**: Available in optional trading layer
- **Real-Time Filtering**: Basic market filters (market cap, timing, symbol hygiene)

## 🎯 System Overview

```
Market Data → Earnings Discovery → Strategy Scoring → Data Collection → Trading Opportunities
    ↓              ↓                    ↓               ↓              ↓
 NASDAQ API   →  ALL Earnings    →   Calendar/     →  Options    →   Prioritized
 (Free)          (1,122+)           Straddle/        Chains          Symbols
                                   Strangle
```

## 📁 Project Structure

```
earnings_ibapi/
├── README.md                    # This file
├── docs/                       # Documentation
│   ├── architecture/           # System design docs
│   ├── guides/                # User guides  
│   └── api/                   # API references
├── earnings/                  # Earnings data acquisition (pure discovery)
│   ├── fetcher.py            # NASDAQ-only by default; opt-in other sources
│   └── scheduler.py          # Priority-based scheduling
├── earnings_trading/         # Deprecated (shim for legacy imports)
│   ├── discovery.py          # Compatibility shim
│   └── data_pipeline.py      # Legacy pipeline (not used in current flow)
├── scripts/                  # Utility scripts
│   └── compare_earnings_sources.py  # Data source comparison
├── examples/                 # Usage examples
├── repositories/             # Data storage layer
├── reliability/              # Circuit breakers, retry logic
├── lineage/                  # Data lineage tracking
└── tests/                    # Test suite
```

## 🔍 Core Components

### 1. Earnings Discovery (`earnings/discovery.py`)
- **Market-wide scanning**: Collects earnings dates (no fixed portfolios)
- **No filtering at this stage**: All collected events are stored and exported
- **Exports**: Daily CSV of earning-date records for downstream consumers

### 2. Configurable Sources (Opt-in) Fetcher (`earnings/fetcher.py`)
- **Default**: NASDAQ-only (no API keys required)
- **Optional**: Enable additional sources via `EARNINGS_ALLOWED_SOURCES` (e.g., `nasdaq,fmp,finnhub`) and set API keys
- 
- 

### 3. Trading Layer (optional)
- **Analyzer**: `trading/strategy_analyzer.py` (not required for discovery)
- **Pipelines**: Deferred until trading is enabled

## 🎯 Recent Earnings-Date Results

```
🔍 EARNINGS DISCOVERY SYSTEM TEST
============================================================

✅ Collected 2,800+ raw events (NASDAQ)
📊 1,000+ earning-date records after basic market filters
💾 Exported to data/exports/earnings_discoveries_YYYY-MM-DD.csv
```

## 📦 Earnings-Date Data Lifecycle

- **Raw Storage**: Parquet under `data/earnings/collection_date=YYYY-MM-DD/source=nasdaq/earnings_HHMMSS.parquet`.
- **Deduplication only**: De-dup by `symbol+earnings_date`; no market filtering at this layer.
- **Daily Export**: CSV at `data/exports/earnings_discoveries_YYYY-MM-DD.csv` sorted by `days_until` (asc) and `market_cap` (desc).
- **Programmatic Access**: `EarningsDiscoveryEngine.get_recent_discoveries(max_age_days=1)` returns the latest as a DataFrame.
- **Downstream Use**: Other systems can consume the daily CSV or query raw parquet directly; any filtering/analysis happens later.

## 📚 Documentation

| Category | Documents |
|----------|-----------|
| **Getting Started** | [Quick Start Guide](docs/guides/quickstart.md) |
| **Architecture** | [System Design](docs/architecture/system-design.md) |
| **Data Sources** | [Earnings APIs](docs/architecture/data-sources.md) |
| **User Guides** | [Discovery Engine](docs/guides/discovery-engine.md) |
| **API Reference** | [Earnings Fetcher](docs/api/earnings-fetcher.md) |
| **Operations** | [Daily Pipeline](docs/guides/daily-pipeline.md) |

## 🛠️ Configuration

### Environment Variables
```bash
# Default: NASDAQ-only (no keys required)
export EARNINGS_ALLOWED_SOURCES=nasdaq

# Optional: enable extra sources (set keys if you opt in)
export FMP_API_KEY=your_fmp_key
export FINNHUB_API_KEY=your_finnhub_key

# Data paths
export DATA_PATH=/path/to/trading/data
export CACHE_PATH=/path/to/cache
```

### Key Settings
- **Min Market Cap**: $1B (configurable in `earnings/discovery.py`)
- **Time Window**: 1-60 days ahead (configurable)
- **Strategy Thresholds**: 50+ score minimum (adjustable)

## 🎯 Usage Examples

### Basic Earnings Discovery (pure)
```python
from earnings.discovery import EarningsDiscoveryEngine

engine = EarningsDiscoveryEngine()
discoveries = engine.discover_and_store_earnings(days_ahead=21, days_back=0)

df = engine.get_recent_discoveries(max_age_days=1)
print(df.head())
```

### Data Source Comparison
```bash
# Compare NASDAQ vs Yahoo earnings data
python scripts/compare_earnings_sources.py --days-ahead 14 --compare-historical 7
```

## 🚀 Recent Major Updates

### NASDAQ API Integration (Latest)
- ✅ **Free market-wide coverage**: 1,122+ earnings events without API keys
- ✅ **Enhanced multi-day fetching**: Covers full date ranges efficiently  
- ✅ **Market cap parsing**: Automatic filtering for options-tradeable symbols
- ✅ **Production-ready**: Handles weekends, rate limiting, error recovery

### Earnings-Driven Architecture 
- ✅ **Replaced fixed portfolios** with dynamic opportunity discovery
- ✅ **Market-wide scanning** instead of predefined symbol lists
- ✅ **Strategy-specific scoring** for calendar spreads, strangles, straddles
- ✅ **Priority-based data collection** focusing on highest-value opportunities

## 🧪 Testing & Validation

```bash
# Run discovery system test (pure)
python -m earnings.discovery

# Test earnings data fetcher
python -m earnings.fetcher

# Optional: run data source comparison (if extra sources enabled)
python scripts/compare_earnings_sources.py

# Full test suite
pytest tests/
```

## 🔧 CI & Troubleshooting

### CI
- Smoke tests: GitHub Actions runs a fast smoke (`python tests/test_runner.py --smoke`).
- Guard rails: CI fails if new code imports the deprecated `earnings_trading` package.

### Common Issues

1. **No earnings data found**
   - Check internet connection
   - NASDAQ API might be temporarily down
   - Verify date ranges (weekends have no data)

2. **API rate limits**
   - System automatically falls back to NASDAQ (free)
   - Add delays between requests if needed
   - Consider upgrading API plans for higher volume

3. **Memory usage with large datasets**
   - Adjust `max_symbols` parameter in discovery
   - Use chunked processing for large date ranges

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow existing code patterns and add tests
4. Update relevant documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASDAQ**: For providing free earnings calendar API
- **Interactive Brokers**: For comprehensive options data via TWS API
- **Financial data providers**: FMP, Finnhub for premium earnings feeds

---

## 📈 Performance Metrics (Latest Run)

| Metric | Value |
|--------|-------|
| **Total Earnings Events** | 1,122 |
| **Filtered Opportunities** | 229 |
| **Processing Time** | <30 seconds |
| **Memory Usage** | ~150MB |
| **Success Rate** | 99.9% |
| **Data Sources Active** | 1 (NASDAQ) |

**Last Updated**: August 12, 2025
