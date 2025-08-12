# Trading Data System - Operational Guide

## Overview

This guide provides comprehensive instructions for operating the trading data system correctly, ensuring data dependencies are respected and the persistence strategy is followed.

## ⚠️ **Critical Rules - MUST FOLLOW**

### 1. **Data Dependency Order** (Cannot be violated)
```
Contracts → Option Chains → Equity Data → Option Bars
```

### 2. **TWS API Constraints** (Technical limitations)
- Option chains are **snapshot-only** (not historical)
- Contract details requests are **throttled** 
- Historical data requires **market data permissions**

### 3. **Persistence Strategy** (Never waste API calls)
- Check existing data before any API request
- Store everything that is requested
- Use incremental updates only

## 🚀 **Getting Started - Cold Start Process**

### Step 1: Initialize a New Symbol

```python
from jobs.orchestrator import DataPipelineOrchestrator
from pathlib import Path

# Initialize orchestrator
orchestrator = DataPipelineOrchestrator(
    base_path=Path("data"),
    enable_lineage=True
)

# Setup new symbol (follows dependency order automatically)
results = orchestrator.setup_new_symbol("AAPL", lookback_days=365)

# Check results
for result in results:
    print(f"{result.job_type.value}: {'SUCCESS' if result.success else 'FAILED'}")
    if not result.success:
        print(f"  Error: {result.error_message}")
```

### Step 2: Validate Setup
```python
from jobs.orchestrator import DataValidation

validator = DataValidation(Path("data"))

try:
    # Validate all prerequisites for option trading
    validator.validate_option_backfill_prerequisites("AAPL")
    print("✅ All prerequisites met - ready for option data")
except PrerequisiteError as e:
    print(f"❌ Prerequisites missing: {e}")
```

## 📅 **Daily Operations**

### Morning Data Update
```python
# List of symbols to update
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

# Run daily updates (incremental only)
results = orchestrator.daily_update(symbols)

# Check for any failures
for symbol, symbol_results in results.items():
    failures = [r for r in symbol_results if not r.success]
    if failures:
        print(f"⚠️ {symbol} had {len(failures)} failures")
        for failure in failures:
            print(f"  - {failure.job_type.value}: {failure.error_message}")
```

### Check System Health
```python
# Validate data integrity
for symbol in symbols:
    try:
        validator.validate_data_integrity(symbol)
        print(f"✅ {symbol}: Data integrity OK")
    except Exception as e:
        print(f"❌ {symbol}: {e}")

# Check API circuit breakers
status = orchestrator.generate_status_report()
print(f"Circuit breaker status: {status['circuit_breaker_status']}")
```

## 📊 **Data Analysis & Research**

### Historical Analysis
```python
from repositories import EquityBarRepository, OptionChainSnapshotRepository
from datetime import date

# Load historical data for analysis
equity_repo = EquityBarRepository("data/equity_bars")
chain_repo = OptionChainSnapshotRepository("data/option_chains")

# Get equity data for date range
equity_data = equity_repo.load_symbol_data("AAPL", 
                                          start_date=date(2024, 1, 1),
                                          end_date=date(2024, 12, 31))

# Get option chain snapshot for analysis
chain_data = chain_repo.load_chain_snapshot("AAPL", date(2024, 6, 15))

print(f"Loaded {len(equity_data)} equity bars")
print(f"Loaded {len(chain_data)} option contracts")
```

### Lineage Analysis  
```python
from lineage.query import LineageQueryEngine
from lineage.visualizer import LineageVisualizer

# Analyze data lineage
query_engine = LineageQueryEngine(tracker)
visualizer = LineageVisualizer(tracker)

# Find data sources
sources = query_engine.find_data_sources(source_type="repository")
print(f"Found {len(sources)} data sources")

# Generate lineage report
report = visualizer.generate_summary_report()
print(report)
```

## ⚡ **Live Trading Operations**

### Real-time Data Setup
```python
from streaming import LiveDataManager

# Initialize live data streams
live_manager = LiveDataManager()

# Start streaming for active symbols
symbols = ["AAPL", "GOOGL"] 
live_manager.start_streaming(symbols)

# Monitor data quality
live_manager.monitor_stream_health()
```

### Position Monitoring
```python
from trading import PositionManager

position_manager = PositionManager()

# Get current positions
positions = position_manager.get_active_positions()

for position in positions:
    # Get real-time Greeks
    greeks = position_manager.calculate_position_greeks(position)
    print(f"{position.symbol}: Delta={greeks.delta:.3f}, Gamma={greeks.gamma:.3f}")
```

## 🔧 **Troubleshooting**

### Common Issues & Solutions

#### ❌ **Problem**: Option backfill fails with "No contract descriptions"
```python
# Solution: Run contract backfill first
from jobs.backfill_contracts_us_rt import backfill_us_equity_contracts

# This must complete before option data
backfill_us_equity_contracts(symbol="AAPL")
```

#### ❌ **Problem**: Option chain data is stale
```python
# Solution: Refresh option chain snapshot
from jobs.contracts_snapshot import refresh_option_chain

# Option chains are snapshot-only, must be refreshed periodically
refresh_option_chain("AAPL")
```

#### ❌ **Problem**: Missing equity data for option analysis
```python
# Solution: Ensure equity data exists for the period
equity_repo = EquityBarRepository("data/equity_bars")

# Check what dates we have
present = equity_repo.present_dates("AAPL", "1 day", start_date, end_date)
missing_dates = set(pd.bdate_range(start_date, end_date).date) - present

if missing_dates:
    print(f"Missing equity data for: {sorted(missing_dates)}")
    # Run equity backfill for missing dates
```

#### ❌ **Problem**: API rate limiting / throttling
```python
# Solution: Use circuit breaker and retry logic
from reliability import CircuitBreaker, RetryConfig

with CircuitBreaker(failure_threshold=5, recovery_timeout=300):
    # API calls are automatically protected
    data = api_client.get_equity_bars(symbol, start_date, end_date)
```

### Data Validation Commands

```python
# Comprehensive validation script
def validate_all_data():
    validator = DataValidation(Path("data"))
    
    symbols = get_all_active_symbols()
    
    validation_report = {
        "contracts_ok": [],
        "chains_ok": [],
        "equity_ok": [],
        "option_ok": [],
        "integrity_ok": [],
        "failures": []
    }
    
    for symbol in symbols:
        try:
            # Check contracts
            validator.validate_symbol_contracts_exist(symbol)
            validation_report["contracts_ok"].append(symbol)
            
            # Check chains  
            validator.validate_option_chain_current(symbol)
            validation_report["chains_ok"].append(symbol)
            
            # Check equity
            validator.validate_equity_data_available(symbol)
            validation_report["equity_ok"].append(symbol)
            
            # Check option prerequisites
            validator.validate_option_backfill_prerequisites(symbol)
            validation_report["option_ok"].append(symbol)
            
            # Check integrity
            validator.validate_data_integrity(symbol)
            validation_report["integrity_ok"].append(symbol)
            
        except Exception as e:
            validation_report["failures"].append({
                "symbol": symbol,
                "error": str(e)
            })
    
    return validation_report

# Run validation
report = validate_all_data()
print(f"Validation complete: {len(report['integrity_ok'])} symbols OK, {len(report['failures'])} failures")
```

## 📋 **Maintenance Tasks**

### Weekly Maintenance
```python
def weekly_maintenance():
    """Run weekly maintenance tasks."""
    
    # 1. Refresh all contract descriptions
    symbols = get_all_active_symbols()
    for symbol in symbols:
        refresh_contract_descriptions(symbol)
    
    # 2. Clean up old temporary data
    cleanup_temp_files()
    
    # 3. Validate data integrity
    integrity_report = validate_all_data()
    send_maintenance_report(integrity_report)
    
    # 4. Update system metrics
    update_system_metrics()
```

### Monthly Tasks
```python
def monthly_maintenance():
    """Run monthly maintenance tasks."""
    
    # 1. Archive old data
    archive_old_data(months_back=12)
    
    # 2. Optimize repository storage
    optimize_parquet_files()
    
    # 3. Update lineage metadata
    cleanup_old_lineage_data()
    
    # 4. Performance analysis
    generate_performance_report()
```

## 🎯 **Best Practices**

### 1. **Always Check Before Requesting**
```python
# ❌ BAD: Request without checking
data = api_client.get_equity_bars("AAPL", "2024-01-01", "2024-12-31")

# ✅ GOOD: Check existing data first  
repo = EquityBarRepository("data/equity_bars")
present = repo.present_dates("AAPL", "1 day", start_date, end_date)

if len(present) < expected_count:
    # Only request missing data
    missing_windows = calculate_missing_windows(present, start_date, end_date)
    for window_start, window_end in missing_windows:
        data = api_client.get_equity_bars("AAPL", window_start, window_end)
        repo.save(data, symbol="AAPL", bar_size="1 day")
```

### 2. **Validate Prerequisites**
```python
# ❌ BAD: Request option data without validation
option_data = api_client.get_option_bars(contracts, start_date, end_date)

# ✅ GOOD: Validate prerequisites first
validator = DataValidation(Path("data"))
try:
    validator.validate_option_backfill_prerequisites("AAPL")
    option_data = api_client.get_option_bars(contracts, start_date, end_date)
except PrerequisiteError as e:
    print(f"Cannot request option data: {e}")
    # Fix prerequisites first
```

### 3. **Use Dependency-Aware Orchestration**
```python
# ❌ BAD: Manual job execution without dependency checks
run_option_backfill("AAPL")  # May fail if contracts missing

# ✅ GOOD: Use orchestrator for dependency management  
orchestrator = DataPipelineOrchestrator(Path("data"))
results = orchestrator.setup_new_symbol("AAPL")  # Handles dependencies automatically
```

### 4. **Monitor Data Quality**
```python
# Set up continuous monitoring
from monitoring import DataQualityMonitor

monitor = DataQualityMonitor()

# Check data quality every hour
@schedule.every(1).hours
def check_data_quality():
    quality_report = monitor.generate_quality_report()
    
    if quality_report.has_critical_issues():
        send_alert("Critical data quality issues detected", quality_report)
    
    # Log quality metrics
    log_quality_metrics(quality_report)
```

## 📈 **Performance Optimization**

### Repository Performance
```python
# Optimize query performance
def optimize_queries():
    # Use proper filtering to reduce data scanned
    equity_repo = EquityBarRepository("data/equity_bars")
    
    # ✅ GOOD: Filter by partitioning columns
    data = equity_repo.load(symbol="AAPL", trade_date="2024-06-15")
    
    # ❌ AVOID: Loading all data then filtering
    all_data = equity_repo.load()
    filtered = all_data[all_data['symbol'] == 'AAPL']
```

### API Optimization
```python
# Batch API requests efficiently
def optimize_api_calls():
    # ✅ GOOD: Batch contract requests
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    # Request all contracts in single batch
    contracts = api_client.get_contract_descriptions(symbols)
    
    # ❌ AVOID: Individual requests
    for symbol in symbols:
        contract = api_client.get_contract_description(symbol)
```

## 🚨 **Alerts & Monitoring**

### Critical Alerts
```python
# Set up critical monitoring
CRITICAL_ALERTS = [
    "prerequisite_violation",  # Option data requested without dependencies
    "data_integrity_failure",   # Data consistency violations
    "api_circuit_breaker_open", # API protection triggered
    "stale_option_chains",     # Option chains older than 24 hours
    "missing_equity_data",     # Equity data gaps
]

def setup_alerts():
    for alert_type in CRITICAL_ALERTS:
        monitor.add_alert_rule(
            alert_type=alert_type,
            severity="critical",
            notification_channel="slack_trading_ops"
        )
```

This operational guide ensures the trading data system is used correctly with proper dependency management and efficient data usage.