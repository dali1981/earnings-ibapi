"""
Minimal pytest configuration for basic testing.
"""
import os
import tempfile
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
import threading
import queue

import pytest
import pandas as pd
import pyarrow as pa


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_equity_data():
    """Sample equity bar data for testing."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "open": [150.0, 151.0, 152.0, 153.0, 154.0],
        "high": [152.0, 153.0, 154.0, 155.0, 156.0],
        "low": [149.0, 150.0, 151.0, 152.0, 153.0],
        "close": [151.0, 152.0, 153.0, 154.0, 155.0],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        "wap": [150.5, 151.5, 152.5, 153.5, 154.5],
    })


@pytest.fixture
def sample_option_data():
    """Sample option bar data for testing."""
    times = pd.date_range("2024-01-01 09:30", periods=10, freq="1min")
    return pd.DataFrame({
        "time": times,
        "open": [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9],
        "high": [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0],
        "low": [4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8],
        "close": [5.05, 5.15, 5.25, 5.35, 5.45, 5.55, 5.65, 5.75, 5.85, 5.95],
        "volume": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    })


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    size = 1000  # Reduced size for testing
    dates = pd.date_range("2024-01-01", periods=size, freq="1min")
    
    return pd.DataFrame({
        "time": dates,
        "open": 150 + (pd.Series(range(size)) % 100) * 0.1,
        "high": 150 + (pd.Series(range(size)) % 100) * 0.1 + 0.5,
        "low": 150 + (pd.Series(range(size)) % 100) * 0.1 - 0.3,
        "close": 150 + (pd.Series(range(size)) % 100) * 0.1 + 0.2,
        "volume": 1000 + pd.Series(range(size)) * 10,
    })


class TestDataHelper:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_bars(symbol: str, start_date: date, num_days: int, 
                   bar_size: str = "1 day") -> pd.DataFrame:
        """Create sample bar data for testing."""
        if bar_size == "1 day":
            dates = pd.date_range(start_date, periods=num_days, freq="D")
            date_col = "date"
            date_values = dates.strftime("%Y%m%d")
        else:
            dates = pd.date_range(start_date, periods=num_days * 100, freq="1min")  # Reduced
            date_col = "time"
            date_values = dates
        
        base_price = 150.0
        return pd.DataFrame({
            date_col: date_values,
            "open": [base_price + i * 0.1 for i in range(len(dates))],
            "high": [base_price + i * 0.1 + 1 for i in range(len(dates))],
            "low": [base_price + i * 0.1 - 0.5 for i in range(len(dates))],
            "close": [base_price + i * 0.1 + 0.5 for i in range(len(dates))],
            "volume": [1000000 + i * 1000 for i in range(len(dates))],
        })


@pytest.fixture
def test_data_helper():
    """Test data helper fixture."""
    return TestDataHelper()


@pytest.fixture
def option_chain_data():
    """Generate sample option chain data."""
    strikes = [140, 145, 150, 155, 160]
    chain_data = []
    
    for strike in strikes:
        for right in ['C', 'P']:
            chain_data.append({
                'underlying': 'AAPL',
                'expiry': date(2024, 3, 15),
                'strike': strike,
                'right': right,
                'bid': strike * 0.05 if right == 'C' else strike * 0.03,
                'ask': strike * 0.06 if right == 'C' else strike * 0.04,
                'last': strike * 0.055 if right == 'C' else strike * 0.035,
                'volume': 100 + strike,
                'openInterest': 1000 + strike * 10,
                'impliedVolatility': 0.25 + (strike - 150) * 0.001,
                'delta': 0.5 if right == 'C' else -0.5,
                'gamma': 0.02,
                'theta': -0.05 if right == 'C' else -0.03,
                'vega': 0.1,
            })
    
    return pd.DataFrame(chain_data)


# ============================================================================
# Data Repository Fixtures
# ============================================================================

@pytest.fixture
def equity_repository(temp_data_dir):
    """Equity bar repository fixture."""
    from ibx_repos.equity_bars import EquityBarRepository
    return EquityBarRepository(temp_data_dir / "equity_bars")


@pytest.fixture
def option_repository(temp_data_dir):
    """Option bar repository fixture."""
    from ibx_repos.option_bars import OptionBarRepository
    return OptionBarRepository(temp_data_dir / "option_bars")


@pytest.fixture
def chain_repository(temp_data_dir):
    """Option chain repository fixture."""
    from ibx_repos.chains import OptionChainSnapshotRepository
    return OptionChainSnapshotRepository(temp_data_dir / "option_chains")


@pytest.fixture
def contract_repository(temp_data_dir):
    """Contract descriptions repository fixture."""
    from ibx_repos.contract_descriptions import ContractDescriptionsRepository
    return ContractDescriptionsRepository(temp_data_dir / "contract_descriptions")


@pytest.fixture
def populated_equity_repo(equity_repository, sample_equity_data):
    """Equity repository with sample data."""
    equity_repository.save(sample_equity_data, "AAPL", "1 day", "TRADES")
    return equity_repository