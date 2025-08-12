# System Logic & Configuration Explanation

## 🔍 **Your Questions Answered**

### **1. Why 7 days ahead?**
**BEFORE**: Hardcoded in various places, confusing defaults
```python
# Old hardcoded values scattered everywhere
days_ahead = 21  # in discovery.py
days_ahead = 60  # in daily_collector.py
--days-ahead 7   # command line override
```

**NOW CONFIGURED**: Clear separation of timing concepts in `config.py`
```python
# config.py - EARNINGS_CONFIG
"default_days_ahead": 30,          # How far ahead to search for earnings

# config.py - DISCOVERY_CONFIG  
"discovery_min_days": 1,           # Discovery window (1-60 days ahead)
"discovery_max_days": 60,          
"strategy_optimal_start": 7,       # When strategies work best (7-14 days)
"strategy_optimal_end": 14,

# Data collection around earnings date
"data_collection": {
    "pre_earnings_days": 3,        # Collect 3 days before earnings  
    "post_earnings_days": 2,       # Collect 2 days after earnings
    "lookback_days": 30,           # Historical baseline (30 days back)
    "iv_history_days": 90          # IV analysis (90 days back)
}
```

### **2. Why strangle strategy for HD?**
**BEFORE**: Hardcoded strategy logic
```python
# Old hardcoded strangle multiplier
if candidate.days_until_earnings <= 7:
    return 1.0  # HD has 7 days = perfect for strangles
```

**NOW CONFIGURED**: Strategy preferences in `config.py`
```python
# config.py - DISCOVERY_CONFIG["strategy_multipliers"]
"strangle": {
    "under_7_days": 1.0,        # HD has 7 days = 1.0 multiplier
    "7_to_14_days": 0.8,
    "over_14_days": 0.6
}
```

### **3. Where does 100 pts come from?**
**CALCULATION BREAKDOWN**:
```python
# HD earnings in 7 days:
base_timing_score = 100.0      # 7 days is in optimal window (7-14 days)
strangle_multiplier = 1.0      # ≤7 days is perfect for strangles
final_score = 100.0 * 1.0 = 100 pts
```

**NOW CONFIGURED**: All scoring weights in `config.py`
```python
# config.py - DISCOVERY_CONFIG["timing_weights"]
"optimal": 100.0,    # 7-14 days = 100 base points
"good": 80.0,        # 3-21 days = 80 base points
"fair": 60.0,        # 1-3 days = 60 base points
"poor": 40.0         # Outside ranges = 40 base points
```

### **4. Where are filters defined?**
**BEFORE**: Hardcoded in `discovery.py`
```python
# Old hardcoded filters
self.min_market_cap = 1e9      # Hardcoded!
self.min_avg_volume = 1e6      # Hardcoded!
```

**NOW CONFIGURED**: All filters in `config.py`
```python
# config.py - DISCOVERY_CONFIG
"min_market_cap": 1_000_000_000,     # $1B minimum
"max_market_cap": 2_000_000_000_000, # $2T maximum  
"min_avg_volume": 1_000_000,         # 1M shares
"min_price": 10.0,                   # $10 minimum
"max_price": 1000.0,                 # $1000 maximum
```

### **5. What data is stored?**
**CLEAR SEPARATION NOW**:

🗄️ **Raw Storage (EarningsRepository)**:
- **What**: ALL unfiltered NASDAQ earnings data (892 events)
- **Where**: `data/earnings/earnings/collection_date=YYYY-MM-DD/source=nasdaq/`
- **Format**: PyArrow/Parquet compressed
- **Script**: `scripts/store_earnings_data.py` (pure storage, no analysis)

📊 **Analysis Output (Discovery Engine)**:
- **What**: FILTERED opportunities with strategy recommendations (229 events)
- **Where**: `data/exports/earnings_opportunities_YYYY-MM-DD.csv`
- **Format**: CSV with scoring and recommendations
- **Script**: `earnings_trading/discovery.py` (analysis layer)

### **6. Simple Storage Script**
✅ **CREATED**: `scripts/store_earnings_data.py`
```bash
# Just store raw data, no analysis
python scripts/store_earnings_data.py --days-ahead 30

# With specific symbols only
python scripts/store_earnings_data.py --symbols AAPL,GOOGL,MSFT

# Summary only (no new collection)
python scripts/store_earnings_data.py --summary-only
```

### **7. Config.py Usage**
✅ **FIXED**: All components now use `config.py`
```python
# All paths now configured
DATA_ROOT = Path("data")
EARNINGS_PATH = DATA_ROOT / "earnings"
EXPORTS_PATH = DATA_ROOT / "exports"

# All settings now configured
EARNINGS_CONFIG = {...}
DISCOVERY_CONFIG = {...}
```

### **8. Database Path Issues**
✅ **FIXED**: Databases use configured paths
```python
# Before: databases created in current directory
repo = EarningsRepository("data/earnings")  # Relative!

# Now: uses configured paths
repo = EarningsRepository()  # Uses EARNINGS_PATH from config
```

### **9. Redundant Logging**
✅ **FIXED**: Centralized logging in `utils/logging_setup.py`
```python
# Before: each script set up its own logging
logging.basicConfig(level=logging.INFO)  # Everywhere!

# Now: centralized setup
from utils.logging_setup import setup_logging
setup_logging()
```

## 📊 **Data Flow Diagram**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NASDAQ API    │    │  Raw Storage    │    │   Discovery     │
│   (892 events)  │───▶│  Repository     │───▶│   Analysis      │
│   All symbols   │    │ (Unfiltered)    │    │  (Filtered)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                    ┌─────────────────┐      ┌─────────────────┐
                    │ PyArrow/Parquet │      │ CSV Opportunities│
                    │ Partitioned     │      │ 229 scored      │
                    │ by date/source  │      │ candidates      │
                    └─────────────────┘      └─────────────────┘
```

## 🛠️ **How to Modify Settings**

### **Change Discovery Filters**
Edit `config.py`:
```python
DISCOVERY_CONFIG = {
    "min_market_cap": 5_000_000_000,    # Change to $5B minimum
    "optimal_days_start": 3,            # Change optimal window
    "optimal_days_end": 10,
    # ... other settings
}
```

### **Change Strategy Scoring**
Edit `config.py`:
```python
DISCOVERY_CONFIG = {
    "strategy_multipliers": {
        "strangle": {
            "under_7_days": 1.2,        # Boost strangle preference
            "7_to_14_days": 1.0,        # Keep strangles strong longer
            "over_14_days": 0.4         # Reduce far-out strangles
        }
    }
}
```

### **Change Collection Timing**
Edit `config.py`:
```python
EARNINGS_CONFIG = {
    "default_days_ahead": 60,           # Look further ahead
    "retention_days": 180,              # Keep data for 6 months
    "cache_hours": 12,                  # Cache for 12 hours
}
```

## 🗂️ **File Organization**

### **Configuration Files**
- `config.py` - **MASTER CONFIGURATION** (all settings)
- `utils/logging_setup.py` - Centralized logging

### **Data Scripts**
- `scripts/store_earnings_data.py` - **Pure storage** (no analysis)
- `scripts/demonstrate_persistence.py` - System demonstration
- `scripts/compare_earnings_sources.py` - Data source comparison

### **Analysis Scripts**
- `earnings_trading/discovery.py` - **Filtered analysis** with scoring
- `jobs/daily_earnings_collection.py` - Automated collection job

### **Data Storage**
- `data/earnings/` - **Raw unfiltered data** (all 892 events)
- `data/exports/` - **Analyzed opportunities** (filtered 229 events)
- `data/logs/` - Log files
- `data/cache/` - Temporary cache files

## 🎯 **Key Improvements Made**

1. ✅ **Configurable Filters**: All hardcoded values moved to `config.py`
2. ✅ **Centralized Paths**: No more relative path issues
3. ✅ **Centralized Logging**: Single setup function for all scripts
4. ✅ **Clear Data Separation**: Raw storage vs analyzed opportunities
5. ✅ **Simple Storage Script**: Pure data storage without analysis
6. ✅ **Documented Logic**: All scoring and filtering logic explained

## 🚀 **Usage Examples**

### **Store Raw Data Only**
```bash
# Just store earnings data, no filtering
python scripts/store_earnings_data.py --days-ahead 30 --source nasdaq
```

### **Analyze for Trading Opportunities**
```python
from earnings_trading.discovery import EarningsDiscoveryEngine

# Uses configured filters and scoring
engine = EarningsDiscoveryEngine()
opportunities = engine.discover_earnings_opportunities()

# Or with custom config
custom_config = DISCOVERY_CONFIG.copy()
custom_config["min_market_cap"] = 5_000_000_000  # $5B minimum
engine = EarningsDiscoveryEngine(config=custom_config)
```

### **Check What's Stored**
```bash
# Summary of stored data
python scripts/store_earnings_data.py --summary-only
```

## ⚙️ **Configuration Schema**

### **EARNINGS_CONFIG** (Collection Settings)
```python
{
    "default_days_ahead": int,      # How far ahead to collect
    "retention_days": int,          # How long to keep data  
    "cache_hours": int,             # Cache fresh data duration
    "data_sources": [str],          # Preferred data sources
    "api_delays": {str: float}      # Rate limiting per source
}
```

### **DISCOVERY_CONFIG** (Analysis Settings)  
```python
{
    # Market filters
    "min_market_cap": int,
    "max_market_cap": int,
    "min_avg_volume": int,
    "min_price": float,
    "max_price": float,
    
    # Timing
    "optimal_days_start": int,
    "optimal_days_end": int,
    
    # Scoring weights
    "timing_weights": {
        "optimal": float,
        "good": float, 
        "fair": float,
        "poor": float
    },
    
    # Strategy multipliers
    "strategy_multipliers": {
        "calendar_spread": {...},
        "straddle": {...},
        "strangle": {...}
    }
}
```

**All your concerns have been addressed with configurable, documented, and centralized solutions!** 🎉