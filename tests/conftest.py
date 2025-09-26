"""
Enhanced pytest configuration for comprehensive testing.
Provides fixtures for repositories, mocks, data generation, and testing utilities.
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import queue
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import pandas as pd
import pyarrow as pa
import pytest
import numpy as np

# Set test environment
os.environ['APP_ENV'] = 'testing'
os.environ['ALLOW_PROD_WRITES'] = '1'

# ============================================================================
# Core Testing Infrastructure
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data with proper cleanup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        'data_dir': 'test_data',
        'cache_ttl': 300,
        'rate_limit': 2,
        'timeout': 10,
        'batch_size': 100,
        'max_retries': 3
    }

# ============================================================================
# Market Data Generation
# ============================================================================

class MarketDataGenerator:
    """Advanced market data generator for realistic testing."""
    
    @staticmethod
    def generate_equity_bars(
        symbol: str, 
        days: int, 
        bar_size: str = "1 day",
        start_date: Optional[date] = None,
        base_price: float = 150.0,
        volatility: float = 0.02
    ) -> pd.DataFrame:
        """Generate realistic equity bar data with proper price relationships."""
        if start_date is None:
            start_date = date.today() - timedelta(days=days)
            
        if bar_size == "1 day":
            dates = pd.date_range(start_date, periods=days, freq="D")
            date_col = "date"
            date_values = dates.strftime("%Y%m%d")
        else:
            # Convert bar_size to minutes for intraday data
            freq_map = {"1 min": "1min", "5 mins": "5min", "1 hour": "1H"}
            freq = freq_map.get(bar_size, "1min")
            dates = pd.date_range(
                f"{start_date} 09:30", 
                periods=days * 390 if bar_size == "1 min" else days * 78,  # Market hours
                freq=freq
            )
            date_col = "time"
            date_values = dates
        
        # Generate realistic price movement using random walk
        np.random.seed(42)  # For reproducible tests
        returns = np.random.normal(0, volatility, len(dates))
        prices = [base_price]
        
        for i in range(1, len(dates)):
            price_change = prices[-1] * returns[i]
            new_price = max(prices[-1] + price_change, 0.01)  # Prevent negative prices
            prices.append(new_price)
        
        # Generate OHLC data with realistic relationships
        data = []
        for i, price in enumerate(prices):
            # High/Low spread based on volatility
            spread = price * volatility * np.random.uniform(0.5, 2.0)
            high = price + spread * np.random.uniform(0.3, 0.7)
            low = price - spread * np.random.uniform(0.3, 0.7)
            
            # Open/Close within High/Low range
            open_price = np.random.uniform(low, high)
            close_price = np.random.uniform(low, high)
            
            # Volume with realistic patterns
            base_volume = 1000000
            volume = int(base_volume * np.random.lognormal(0, 0.5))
            
            # WAP (Volume Weighted Average Price)
            wap = (high + low + close_price) / 3
            
            data.append({
                date_col: date_values[i],
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": volume,
                "wap": round(wap, 2)
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_option_chain(
        underlying: str, 
        expiries: List[date],
        spot_price: float = 150.0,
        risk_free_rate: float = 0.05
    ) -> pd.DataFrame:
        """Generate realistic option chain data with Greeks calculations."""
        chain_data = []
        
        for expiry in expiries:
            days_to_expiry = (expiry - date.today()).days
            time_to_expiry = max(days_to_expiry / 365.0, 0.001)  # Avoid division by zero
            
            # Generate strikes around the money
            strikes = [
                round(spot_price * mult, 0) 
                for mult in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
            ]
            
            for strike in strikes:
                for right in ["C", "P"]:
                    # Simplified Black-Scholes approximation for testing
                    moneyness = spot_price / strike
                    base_iv = 0.25
                    
                    # IV smile: higher IV for OTM options
                    if right == "C":
                        iv_adj = 0.05 * max(0, 1 - moneyness)  # OTM calls
                    else:
                        iv_adj = 0.05 * max(0, moneyness - 1)  # OTM puts
                    
                    implied_vol = base_iv + iv_adj
                    
                    # Approximate option pricing
                    if right == "C":
                        intrinsic = max(spot_price - strike, 0)
                        delta = 0.5 + 0.3 * (moneyness - 1)
                    else:
                        intrinsic = max(strike - spot_price, 0)
                        delta = -0.5 - 0.3 * (moneyness - 1)
                    
                    # Time value decreases with time to expiry
                    time_value = strike * implied_vol * np.sqrt(time_to_expiry) * 0.4
                    theo_price = intrinsic + time_value
                    
                    # Greeks approximations
                    gamma = 0.02 * np.exp(-(moneyness - 1)**2 * 4)
                    theta = -theo_price * 0.1 / (time_to_expiry * 365)
                    vega = strike * 0.01 * np.sqrt(time_to_expiry)
                    
                    # Bid/Ask spread
                    spread = max(theo_price * 0.05, 0.05)
                    bid = max(theo_price - spread/2, 0.01)
                    ask = theo_price + spread/2
                    
                    # Volume and OI patterns
                    volume = int(np.random.exponential(100))
                    open_interest = int(np.random.exponential(500))
                    
                    chain_data.append({
                        "underlying": underlying,
                        "expiry": expiry,
                        "strike": strike,
                        "right": right,
                        "bid": round(bid, 2),
                        "ask": round(ask, 2),
                        "last": round((bid + ask) / 2, 2),
                        "volume": volume,
                        "openInterest": open_interest,
                        "impliedVolatility": round(implied_vol, 4),
                        "delta": round(delta, 4),
                        "gamma": round(gamma, 4),
                        "theta": round(theta, 4),
                        "vega": round(vega, 4)
                    })
        
        return pd.DataFrame(chain_data)
    
    @staticmethod
    def generate_earnings_calendar(
        symbols: List[str],
        days_ahead: int = 30
    ) -> pd.DataFrame:
        """Generate realistic earnings calendar data."""
        calendar_data = []
        start_date = date.today()
        
        for i, symbol in enumerate(symbols):
            # Spread earnings across the period
            earnings_date = start_date + timedelta(days=i * (days_ahead // len(symbols)))
            
            # Generate realistic earnings data
            revenue_base = np.random.uniform(1e9, 50e9)  # $1B to $50B
            eps_base = np.random.uniform(0.5, 5.0)
            
            calendar_data.append({
                "symbol": symbol,
                "company_name": f"{symbol} Corporation",
                "earnings_date": earnings_date,
                "period_ending": earnings_date - timedelta(days=90),  # Quarterly
                "when_time": np.random.choice(["BMO", "AMC", "DMT"]),  # Before/After/During market
                "revenue_estimate": round(revenue_base, 2),
                "eps_estimate": round(eps_base, 2),
                "confirmed": np.random.choice([True, False], p=[0.8, 0.2])
            })
        
        return pd.DataFrame(calendar_data)

@pytest.fixture
def market_data_generator():
    """Market data generator fixture."""
    return MarketDataGenerator()

# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_equity_data():
    """Sample equity bar data for testing."""
    generator = MarketDataGenerator()
    return generator.generate_equity_bars("AAPL", 5, "1 day")

@pytest.fixture
def sample_option_data():
    """Sample option bar data for testing."""
    times = pd.date_range("2024-01-01 09:30", periods=10, freq="1min")
    return pd.DataFrame({
        "time": times,
        "open": np.random.uniform(5.0, 6.0, 10),
        "high": np.random.uniform(5.5, 6.5, 10),
        "low": np.random.uniform(4.5, 5.5, 10),
        "close": np.random.uniform(4.8, 6.2, 10),
        "volume": np.random.randint(50, 500, 10)
    })

@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    size = 10000  # Larger dataset for performance tests
    generator = MarketDataGenerator()
    return generator.generate_equity_bars("TEST", size//100, "1 min")

@pytest.fixture
def option_chain_data():
    """Generate sample option chain data."""
    generator = MarketDataGenerator()
    expiries = [date.today() + timedelta(days=30), date.today() + timedelta(days=60)]
    return generator.generate_option_chain("AAPL", expiries)

@pytest.fixture
def earnings_calendar_data():
    """Generate sample earnings calendar data."""
    generator = MarketDataGenerator()
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    return generator.generate_earnings_calendar(symbols)

# ============================================================================
# Repository Fixtures
# ============================================================================

@pytest.fixture
def equity_repository(temp_data_dir):
    """Equity bar repository fixture."""
    from common.repositories.equity_bars import EquityBarRepository
    repo = EquityBarRepository(temp_data_dir / "equity_bars")
    repo.base_path.mkdir(parents=True, exist_ok=True)
    return repo

@pytest.fixture
def option_repository(temp_data_dir):
    """Option bar repository fixture."""
    from common.repositories.option_bars import OptionBarRepository
    repo = OptionBarRepository(temp_data_dir / "option_bars") 
    repo.base_path.mkdir(parents=True, exist_ok=True)
    return repo

@pytest.fixture
def option_chains_repository(temp_data_dir):
    """Option chains repository fixture."""
    from common.repositories.option_chains import OptionChainSnapshotRepository
    repo = OptionChainSnapshotRepository(temp_data_dir / "option_chains")
    repo.base_path.mkdir(parents=True, exist_ok=True)
    return repo

@pytest.fixture
def option_contracts_repository(temp_data_dir):
    """Option contracts repository fixture."""
    from common.repositories.option_contracts import OptionContractsRepository
    repo = OptionContractsRepository(temp_data_dir / "option_contracts")
    repo.base_path.mkdir(parents=True, exist_ok=True)
    return repo

@pytest.fixture
def contract_descriptions_repository(temp_data_dir):
    """Contract descriptions repository fixture."""
    from common.repositories.contract_descriptions import ContractDescriptionsRepository
    repo = ContractDescriptionsRepository(temp_data_dir / "contract_descriptions")
    repo.base_path.mkdir(parents=True, exist_ok=True)
    return repo

@pytest.fixture
def earnings_repository(temp_data_dir):
    """Earnings repository fixture."""
    from common.repositories.earnings import EarningsCalendarRepository
    from datetime import date
    
    repo = EarningsCalendarRepository(temp_data_dir / "earnings")
    repo.base_path.mkdir(parents=True, exist_ok=True)
    
    # Add a helper method for testing that accepts DataFrames
    def save_dataframe_for_testing(df, collection_date=None, source="test"):
        """Helper method to save DataFrame data for testing."""
        if collection_date is None:
            collection_date = date.today()
        
        # Convert DataFrame rows to simple dict format and save directly to parquet
        import pandas as pd
        
        # Add required metadata columns
        df = df.copy()
        if 'collection_date' not in df.columns:
            df['collection_date'] = collection_date
        if 'source' not in df.columns:
            df['source'] = source
        
        # Create the partition directory structure
        partition_path = repo.data_path / f"collection_date={collection_date}" / f"source={source}"
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Save directly as parquet
        parquet_file = partition_path / "data.parquet"
        df.to_parquet(parquet_file, index=False)
        
        return True
    
    # Add the helper method to the repository
    repo.save_dataframe_for_testing = save_dataframe_for_testing
    return repo

@pytest.fixture
def populated_equity_repo(equity_repository, sample_equity_data):
    """Equity repository with sample data."""
    equity_repository.save(sample_equity_data, symbol="AAPL", bar_size="1 day", data_type="TRADES")
    return equity_repository

@pytest.fixture
def populated_option_chains_repo(option_chains_repository, option_chain_data):
    """Option chains repository with sample data."""
    # Convert option_chain_data to the proper format for option chains
    from datetime import date
    aggregated_data = pd.DataFrame({
        "underlying": ["AAPL"],
        "underlying_conid": [123456],
        "exchange": ["SMART"],
        "trading_class": ["AAPL"],
        "multiplier": ["100"],
        "expirations": [["20240315", "20240415"]],
        "strikes": [[140.0, 145.0, 150.0, 155.0, 160.0]],
        "as_of": [date.today()]
    })
    option_chains_repository.save_chain_snapshot(aggregated_data, "AAPL", date.today())
    return option_chains_repository

# ============================================================================
# Event Bus & Messaging Fixtures
# ============================================================================

@pytest.fixture
def event_bus_test_environment(temp_data_dir):
    """Complete event bus testing environment with cleanup."""
    from common.utils.events.bus import FileEventBus
    from common.utils.events.consumer import JSONLConsumer
    from common.utils.events.processed_index import ProcessedIndex
    
    # Set up event bus with temp directory - pass cache directory as root
    cache_dir = temp_data_dir / "cache"
    state_dir = cache_dir / "state"
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    
    bus = FileEventBus(cache_dir)
    
    return {
        'bus': bus,
        'events_dir': bus.events_dir,  # Use the actual events_dir from the bus
        'state_dir': state_dir,
        'consumer_factory': lambda topic: JSONLConsumer(topic, bus.events_dir),
        'index_factory': lambda service, topic: ProcessedIndex(service, topic, state_dir)
    }

@pytest.fixture
def mock_event_consumer():
    """Mock event consumer for testing."""
    consumer = Mock()
    consumer.read.return_value = iter([
        {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'data_available',
            'payload': {
                'dataset': 'equity_bars',
                'symbol': 'AAPL',
                'start': '2024-01-01',
                'end': '2024-01-31'
            }
        }
    ])
    return consumer

# ============================================================================
# Mock Factories
# ============================================================================

class EnhancedIBClientMock:
    """Sophisticated IB API mock with realistic market data generation."""
    
    def __init__(self):
        self.connected = False
        self.next_order_id = 1
        self.positions = {}
        self.account_values = {}
        self.contract_details = {}
        self.historical_data = {}
        
    def connect(self, host="127.0.0.1", port=7497, clientId=1):
        self.connected = True
        return True
        
    def disconnect(self):
        self.connected = False
        
    def isConnected(self):
        return self.connected
        
    def reqContractDetails(self, reqId, contract):
        # Simulate contract details response
        details = Mock()
        details.contract = contract
        details.marketName = "SMART"
        details.minTick = 0.01
        details.priceMagnifier = 1
        return details
        
    def reqHistoricalData(self, reqId, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate=False, chartOptions=[]):
        # Generate realistic historical data
        symbol = contract.symbol
        generator = MarketDataGenerator()
        
        # Parse duration to get number of bars
        duration_map = {"1 D": 1, "1 W": 7, "1 M": 30, "1 Y": 365}
        days = duration_map.get(durationStr, 30)
        
        bars_df = generator.generate_equity_bars(symbol, days, barSizeSetting)
        
        # Convert to IB bar format
        bars = []
        for _, row in bars_df.iterrows():
            bar = Mock()
            bar.date = row['date'] if 'date' in row else row['time']
            bar.open = row['open']
            bar.high = row['high']
            bar.low = row['low']
            bar.close = row['close']
            bar.volume = row['volume']
            bar.wap = row.get('wap', (row['high'] + row['low']) / 2)
            bar.count = 100
            bars.append(bar)
            
        self.historical_data[reqId] = bars
        return bars

@pytest.fixture
def mock_ib_client_factory():
    """Advanced IB API mock factory."""
    def create_mock(connected=True):
        mock = EnhancedIBClientMock()
        if connected:
            mock.connect()
        return mock
    return create_mock

@pytest.fixture
def mock_celery_task():
    """Mock Celery task for async testing."""
    task = Mock()
    task.delay.return_value = Mock(id="test-task-id", status="SUCCESS")
    task.apply_async.return_value = Mock(id="test-task-id", status="SUCCESS")
    return task

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db = Mock()
    db.execute.return_value = Mock(fetchall=Mock(return_value=[]), fetchone=Mock(return_value=None))
    db.commit.return_value = None
    db.rollback.return_value = None
    return db

# ============================================================================
# Performance Testing Utilities
# ============================================================================

@pytest.fixture
def performance_test_dataset():
    """Large-scale test data for performance validation."""
    generator = MarketDataGenerator()
    
    # Generate multiple symbols with substantial data
    datasets = {}
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    for symbol in symbols:
        datasets[symbol] = {
            'daily_bars': generator.generate_equity_bars(symbol, 365, "1 day"),  # 1 year daily
            'minute_bars': generator.generate_equity_bars(symbol, 5, "1 min"),   # 5 days minute
            'option_chains': generator.generate_option_chain(symbol, [
                date.today() + timedelta(days=30),
                date.today() + timedelta(days=60),
                date.today() + timedelta(days=90)
            ])
        }
    
    return datasets

@pytest.fixture
def benchmark_timer():
    """Timer utility for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
            
        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self):
            if self.end_time is None:
                return time.perf_counter() - self.start_time
            return self.end_time - self.start_time
    
    return Timer

# ============================================================================
# Test Data Helpers
# ============================================================================

class TestDataHelper:
    """Helper class for generating various test data scenarios."""
    
    @staticmethod
    def create_bars(symbol: str, start_date: date, num_days: int, bar_size: str = "1 day") -> pd.DataFrame:
        """Create sample bar data for testing."""
        generator = MarketDataGenerator()
        return generator.generate_equity_bars(symbol, num_days, bar_size, start_date)
    
    @staticmethod
    def create_error_scenario_data():
        """Create data that triggers various error conditions."""
        return pd.DataFrame({
            'date': ['invalid_date', '20240101', '20240102'],
            'open': [None, -100, 'invalid'],  # None, negative, wrong type
            'high': [150, 'abc', 155],
            'low': [148, 149, None],
            'close': [149.5, 150.5, 151.5],
            'volume': [-1000, 1100000, 1200000]  # Negative volume
        })
    
    @staticmethod
    def create_boundary_test_data():
        """Create data for boundary condition testing."""
        return pd.DataFrame({
            'date': ['20240101'],
            'open': [1e-10],     # Very small price
            'high': [1e10],      # Very large price
            'low': [0],          # Zero price
            'close': [float('inf')],  # Infinite price
            'volume': [0]        # Zero volume
        })

@pytest.fixture
def test_data_helper():
    """Test data helper fixture."""
    return TestDataHelper()

# ============================================================================
# Async Testing Support
# ============================================================================

@pytest.fixture
async def async_test_client():
    """Async test client for API testing."""
    from httpx import AsyncClient
    
    async with AsyncClient() as client:
        yield client

@pytest.fixture
def mock_async_service():
    """Mock async service for testing."""
    service = AsyncMock()
    service.process_data = AsyncMock(return_value={"status": "success"})
    service.fetch_data = AsyncMock(return_value=[{"id": 1, "value": "test"}])
    return service

# ============================================================================
# Logging & Debugging Fixtures
# ============================================================================

@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    # Cleanup
    root_logger.removeHandler(handler)
    root_logger.setLevel(original_level)

@pytest.fixture
def debug_mode():
    """Enable debug mode for testing."""
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.DEBUG)
    
    yield True
    
    logging.getLogger().setLevel(original_level)

# ============================================================================
# Cleanup & Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Auto cleanup after each test."""
    yield
    
    # Clean up any test-specific environment variables
    test_env_vars = [k for k in os.environ.keys() if k.startswith('TEST_')]
    for var in test_env_vars:
        os.environ.pop(var, None)