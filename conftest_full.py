"""
Pytest configuration and shared fixtures for the trading project.
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
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from ibapi.client import EClient

# Import project modules
from config import IB_HOST, IB_PORT, IB_CLIENT_IDS, LOGGING_CONFIG
from ibx_repos.equity_bars import EquityBarRepository
from ibx_repos.option_bars import OptionBarRepository
from ibx_repos.chains import OptionChainSnapshotRepository
# from api.ib_core import IBRuntime, RequestIdSequencer, FutureRegistry, IBWrapperBridge
from streamer import EarningsStreamer, Config
from utils import RateLimiter


# ============================================================================
# Test Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Test configuration dictionary."""
    return {
        "ib": {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 999,  # Test client ID
            "timeout": 5.0,
        },
        "data": {
            "test_symbols": ["AAPL", "MSFT", "TSLA", "GOOGL"],
            "test_dates": [
                date(2024, 1, 15),
                date(2024, 2, 15),
                date(2024, 3, 15),
            ],
        },
        "test_settings": {
            "fast_mode": True,
            "use_mocks": True,
        }
    }


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_earnings_config(temp_data_dir, test_config):
    """Test configuration for EarningsStreamer."""
    return Config(
        symbol="AAPL",
        date=date(2024, 3, 15),
        host=test_config["ib"]["host"],
        port=test_config["ib"]["port"],
        client_id=test_config["ib"]["client_id"],
        days_before=5,
        days_after=2,
        out_dir=temp_data_dir
    )


# ============================================================================
# IB API Mocking Fixtures
# ============================================================================

class MockBar:
    """Mock IB API Bar object."""
    def __init__(self, date_str: str, open_price: float, high: float, 
                 low: float, close: float, volume: int, wap: float = None, 
                 barCount: int = None):
        self.date = date_str
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.wap = wap or (high + low) / 2
        self.barCount = barCount or 1


class MockContractDetails:
    """Mock IB API ContractDetails object."""
    def __init__(self, symbol: str, conId: int = None):
        self.contract = Mock()
        self.contract.symbol = symbol
        self.contract.conId = conId or hash(symbol) % 100000
        self.contract.secType = "STK"
        self.contract.exchange = "SMART"
        self.contract.currency = "USD"


class MockIBClient:
    """Mock IB API Client for testing."""
    
    def __init__(self):
        self.connected = False
        self.requests = {}
        self.request_counter = 1000
        self.wrapper = None
        
    def connect(self, host: str, port: int, clientId: int):
        self.connected = True
        if self.wrapper:
            # Simulate nextValidId callback
            threading.Timer(0.1, lambda: self.wrapper.nextValidId(self.request_counter)).start()
    
    def disconnect(self):
        self.connected = False
    
    def isConnected(self):
        return self.connected
    
    def reqContractDetails(self, reqId: int, contract: Contract):
        self.requests[reqId] = {"type": "contractDetails", "contract": contract}
        # Simulate async response
        details = MockContractDetails(contract.symbol)
        threading.Timer(0.1, lambda: self._send_contract_details(reqId, details)).start()
    
    def reqHistoricalData(self, reqId: int, contract: Contract, endDateTime: str,
                         durationStr: str, barSizeSetting: str, whatToShow: str,
                         useRTH: int, formatDate: int, keepUpToDate: bool, chartOptions: List):
        self.requests[reqId] = {
            "type": "historicalData", 
            "contract": contract,
            "endDateTime": endDateTime,
            "duration": durationStr,
            "barSize": barSizeSetting
        }
        # Simulate historical data response
        threading.Timer(0.1, lambda: self._send_historical_data(reqId, contract)).start()
    
    def reqMktData(self, reqId: int, contract: Contract, genericTicks: str,
                   snapshot: bool, regulatorySnapshot: bool, mktDataOptions: List):
        self.requests[reqId] = {"type": "marketData", "contract": contract}
        # Simulate market data tick
        threading.Timer(0.1, lambda: self._send_market_data(reqId)).start()
    
    def reqSecDefOptParams(self, reqId: int, underlyingSymbol: str, exchange: str,
                          underlyingSecType: str, underlyingConId: int):
        self.requests[reqId] = {
            "type": "secDefOptParams",
            "symbol": underlyingSymbol,
            "conId": underlyingConId
        }
        threading.Timer(0.1, lambda: self._send_option_params(reqId)).start()
    
    def _send_contract_details(self, reqId: int, details):
        if self.wrapper:
            self.wrapper.contractDetails(reqId, details)
            self.wrapper.contractDetailsEnd(reqId)
    
    def _send_historical_data(self, reqId: int, contract: Contract):
        if self.wrapper:
            # Generate sample bars
            base_date = datetime.now().date() - timedelta(days=5)
            base_price = 150.0
            
            for i in range(5):
                bar_date = base_date + timedelta(days=i)
                open_price = base_price + i
                high_price = open_price + 2
                low_price = open_price - 1
                close_price = open_price + 1
                volume = 1000000 + i * 10000
                
                bar = MockBar(
                    bar_date.strftime("%Y%m%d"),
                    open_price, high_price, low_price, close_price, volume
                )
                self.wrapper.historicalData(reqId, bar)
            
            self.wrapper.historicalDataEnd(reqId, "", "")
    
    def _send_market_data(self, reqId: int):
        if self.wrapper:
            # Simulate bid/ask ticks
            self.wrapper.tickPrice(reqId, 1, 150.25, Mock())  # Bid
            self.wrapper.tickPrice(reqId, 2, 150.27, Mock())  # Ask
            self.wrapper.tickSize(reqId, 0, 100)  # Bid size
            self.wrapper.tickSize(reqId, 3, 200)  # Ask size
    
    def _send_option_params(self, reqId: int):
        if self.wrapper:
            # Sample option parameters
            self.wrapper.securityDefinitionOptionParameter(
                reqId, "SMART", 12345, "AAPL", 100,
                ["20240315", "20240415", "20240515"],
                [140.0, 145.0, 150.0, 155.0, 160.0]
            )
            self.wrapper.securityDefinitionOptionParameterEnd(reqId)


@pytest.fixture
def mock_ib_client():
    """Mock IB Client fixture."""
    return MockIBClient()


@pytest.fixture
def mock_ib_runtime(mock_ib_client):
    """Mock IB Runtime with mocked client."""
    # Import here to avoid circular import
    from api.ib_core import IBRuntime
    with patch('api.ib_core.EClient', return_value=mock_ib_client):
        runtime = IBRuntime(host="127.0.0.1", port=4002, client_id=999)
        runtime.client = mock_ib_client
        mock_ib_client.wrapper = runtime.wrapper
        yield runtime


@pytest.fixture
def mock_wrapper():
    """Mock EWrapper for testing."""
    wrapper = Mock(spec=EWrapper)
    return wrapper


# ============================================================================
# Data Repository Fixtures
# ============================================================================

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
def equity_repository(temp_data_dir):
    """Equity bar repository fixture."""
    return EquityBarRepository(temp_data_dir / "equity_bars")


@pytest.fixture
def option_repository(temp_data_dir):
    """Option bar repository fixture."""
    return OptionBarRepository(temp_data_dir / "option_bars")


@pytest.fixture
def chain_repository(temp_data_dir):
    """Option chain repository fixture."""
    return OptionChainSnapshotRepository(temp_data_dir / "option_chains")


@pytest.fixture
def populated_equity_repo(equity_repository, sample_equity_data):
    """Equity repository with sample data."""
    equity_repository.save(sample_equity_data, "AAPL", "1 day", "TRADES")
    return equity_repository


# ============================================================================
# Contract and Utility Fixtures
# ============================================================================

@pytest.fixture
def sample_stock_contract():
    """Sample stock contract for testing."""
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


@pytest.fixture
def sample_option_contract():
    """Sample option contract for testing."""
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "OPT"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = "20240315"
    contract.strike = 150.0
    contract.right = "C"
    contract.multiplier = "100"
    return contract


@pytest.fixture
def rate_limiter():
    """Rate limiter fixture for testing."""
    return RateLimiter(calls_per_second=10)


@pytest.fixture
def request_sequencer():
    """Request ID sequencer fixture."""
    from api.ib_core import RequestIdSequencer
    seq = RequestIdSequencer()
    seq.set_base(1000)  # Set base for testing
    return seq


# ============================================================================
# Streaming and Integration Fixtures
# ============================================================================

@pytest.fixture
def mock_earnings_streamer(test_earnings_config, mock_ib_runtime):
    """Mock earnings streamer for testing."""
    with patch('streamer.IBClient') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        streamer = EarningsStreamer(test_earnings_config)
        streamer._client = mock_client
        yield streamer


# ============================================================================
# Test Data Generators
# ============================================================================

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


@pytest.fixture
def market_data_ticks():
    """Generate sample market data ticks."""
    return [
        {'type': 'tickPrice', 'tick': 'BID', 'price': 150.25},
        {'type': 'tickPrice', 'tick': 'ASK', 'price': 150.27},
        {'type': 'tickSize', 'tick': 'BID_SIZE', 'size': 100},
        {'type': 'tickSize', 'tick': 'ASK_SIZE', 'size': 200},
        {'type': 'tickString', 'tick': 'LAST_TIMESTAMP', 'value': '1640995200'},
        {'type': 'tickGeneric', 'tick': 'SHORTABLE', 'value': 3.0},
    ]


# ============================================================================
# Performance and Load Testing Fixtures
# ============================================================================

@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    size = 10000
    dates = pd.date_range("2020-01-01", periods=size, freq="1min")
    
    return pd.DataFrame({
        "time": dates,
        "open": 150 + (pd.Series(range(size)) % 100) * 0.1,
        "high": 150 + (pd.Series(range(size)) % 100) * 0.1 + 0.5,
        "low": 150 + (pd.Series(range(size)) % 100) * 0.1 - 0.3,
        "close": 150 + (pd.Series(range(size)) % 100) * 0.1 + 0.2,
        "volume": 1000 + pd.Series(range(size)) * 10,
    })


# ============================================================================
# Test Utilities
# ============================================================================

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
            dates = pd.date_range(start_date, periods=num_days * 390, freq="1min")
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


# ============================================================================
# Cleanup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Cleanup temporary files after each test."""
    yield
    # Cleanup code can go here if needed


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    for item in items:
        # Add markers based on test location or name
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)