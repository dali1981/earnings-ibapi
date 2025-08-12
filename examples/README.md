# IB API Examples

This directory contains examples demonstrating Interactive Brokers API integration.

## Files

### `ib_fundamental_data.py`
**Pure IB-only example** - Uses only Interactive Brokers API
- Fetches fundamental data from IB only
- Gets earnings calendar from IB only  
- Retrieves analyst estimates from IB only
- No external API dependencies
- Clean, focused demonstration of IB capabilities

**Usage:**
```bash
python examples/ib_fundamental_data.py
```

## Prerequisites

- IB Gateway or TWS running
- `ibapi` package installed: `pip install ibapi`
- Valid IB account with market data subscriptions

## Features

- **Company Fundamentals**: P/E ratios, market cap, financial ratios
- **Earnings Calendar**: Earnings dates and timing from IB
- **Analyst Estimates**: EPS and revenue estimates 
- **Financial Health Scoring**: Automated fundamental analysis
- **CSV Export**: Export all data for further analysis