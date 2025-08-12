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

### 1. Discover Earnings Opportunities
```bash
python -m earnings_trading.discovery
```

**Expected Output:**
```
🔍 EARNINGS DISCOVERY SYSTEM TEST
============================================================

✅ Found 1,122 total earnings events
📊 229 candidates passed basic filters
🎯 229 high-quality opportunities identified

TOP OPPORTUNITIES:
 1. HD     - 2025-08-19 (+7 days) - 100 pts - strangle
 2. MDT    - 2025-08-19 (+7 days) - 100 pts - strangle
 3. PANW   - 2025-08-18 (+6 days) - 100 pts - strangle

💾 Opportunities exported to: data/exports/earnings_opportunities_2025-08-12.csv
```

### 2. Test Earnings Data Fetcher
```bash
python -m earnings.fetcher
```

### 3. Compare Data Sources
```bash
python scripts/compare_earnings_sources.py
```

## 📊 Understanding the Results

### Opportunity Scoring
- **100 pts**: Excellent opportunity, earnings within 7 days
- **80 pts**: Very good opportunity, good timing
- **60 pts**: Good opportunity, reasonable timing
- **50 pts**: Consider opportunity, adequate timing

### Strategy Recommendations
- **strangle**: High volatility expected, suitable for directional moves
- **calendar_spread**: Time decay play, good for range-bound movement
- **straddle**: Maximum volatility play, best for binary events

### Priority Classifications
- **HIGH**: Earnings within 7 days, score 75+
- **MEDIUM**: Earnings within 14 days, score 60+
- **LOW**: Earnings >14 days, score 50+

## 🎯 Basic Usage Examples

### Programmatic Discovery
```python
from earnings_trading.discovery import EarningsDiscoveryEngine

# Initialize engine
engine = EarningsDiscoveryEngine()

# Discover opportunities for next 21 days
opportunities = engine.discover_earnings_opportunities(
    days_ahead=21,
    min_score=60.0  # Only high-quality opportunities
)

# Show top 5 opportunities
for opp in opportunities[:5]:
    print(f"{opp.symbol}: {opp.best_strategy.value} - {opp.total_score:.0f} pts")
```

### Custom Filtering
```python
# Filter for specific market cap range
engine.min_market_cap = 5e9  # $5B minimum
engine.max_days_ahead = 14   # Only next 2 weeks

opportunities = engine.discover_earnings_opportunities()
```

### Export to CSV
```python
# Export with custom filename
export_file = engine.export_opportunities(
    opportunities,
    output_file=Path("my_opportunities.csv")
)
```

## 🔍 Exploring Results

### View Exported CSV
```bash
# Open exported opportunities
open data/exports/earnings_opportunities_2025-08-12.csv

# Or view in terminal
head -10 data/exports/earnings_opportunities_2025-08-12.csv
```

**CSV Structure:**
```
Symbol,Company,Earnings_Date,Time,Days_Until,Best_Strategy,Total_Score,Calendar_Score,Straddle_Score,Strangle_Score,Priority,Action_Needed
HD,"Home Depot, Inc. (The)",2025-08-19,bmo,7,strangle,100.0,80.0,90.0,100.0,🔥 EXCELLENT,GET_OPTION_CHAIN | GET_IV_DATA | ANALYZE_STRANGLE
```

### Understanding Action Items
- **GET_OPTION_CHAIN**: Need to collect options data for this symbol
- **GET_IV_DATA**: Need implied volatility historical data
- **ANALYZE_STRANGLE**: Perform detailed strangle analysis

## 🛠️ Configuration

### Basic Settings (earnings_trading/discovery.py)
```python
# Market filters
self.min_market_cap = 1e9      # $1B minimum
self.min_avg_volume = 1e6      # 1M shares volume
self.min_price = 10.0          # $10 minimum stock price
self.max_price = 1000.0        # $1000 maximum stock price

# Timing windows  
self.min_days_ahead = 1        # At least 1 day ahead
self.max_days_ahead = 60       # Within 2 months
```

### Strategy Scoring Weights
```python
# Modify strategy multipliers for different preferences
def _strangle_multiplier(self, candidate) -> float:
    if candidate.days_until_earnings <= 7:
        return 1.0  # Perfect timing
    elif candidate.days_until_earnings <= 14:
        return 0.8  # Good timing
    else:
        return 0.6  # Adequate timing
```

## 📈 Next Steps

### 1. Data Collection Setup
Once you have identified opportunities, set up data collection:
```bash
# Run comprehensive data pipeline
python -m jobs.run_earnings_pipeline
```

### 2. Options Chain Analysis
For detailed options analysis:
```bash
# Get option chains for top opportunities
python examples/complete_workflow_example.py
```

### 3. Historical Analysis
Compare current opportunities with historical patterns:
```bash
python scripts/compare_earnings_sources.py --compare-historical 30
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