"""
Unit tests for market data feed.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional

# These imports will fail initially (Red phase)
from common.trading.market_data import MarketDataFeed, MarketData, MarketDataManager
from common.schemas.trading import EquityContract, OptionContract
from common.constants import SecType, OptionRight


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.market_data
class TestMarketData:
    """Test MarketData data model."""
    
    def test_market_data_initialization(self):
        """Test MarketData initialization with required fields."""
        data = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02"),
            volume=1000,
            timestamp=datetime.utcnow()
        )
        
        assert data.symbol == "AAPL"
        assert data.bid == Decimal("150.00")
        assert data.ask == Decimal("150.05")
        assert data.last == Decimal("150.02")
        assert data.volume == 1000
        assert isinstance(data.timestamp, datetime)
    
    def test_market_data_spread_calculation(self):
        """Test bid-ask spread calculation."""
        data = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02"),
            volume=1000,
            timestamp=datetime.utcnow()
        )
        
        spread = data.spread
        assert spread == Decimal("0.05")
    
    def test_market_data_mid_price_calculation(self):
        """Test mid price calculation."""
        data = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02"),
            volume=1000,
            timestamp=datetime.utcnow()
        )
        
        mid_price = data.mid_price
        assert mid_price == Decimal("150.025")


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.market_data
class TestMarketDataFeed:
    """Test MarketDataFeed functionality."""
    
    @pytest.fixture
    def mock_ib_client(self):
        """Create mock IB client."""
        client = Mock()
        client.is_connected = Mock(return_value=True)
        client.request_market_data = AsyncMock()
        client.cancel_market_data = AsyncMock()
        return client
    
    @pytest.fixture
    def market_data_feed(self, mock_ib_client):
        """Create market data feed for testing."""
        return MarketDataFeed(ib_client=mock_ib_client)
    
    @pytest.fixture
    def equity_contract(self):
        """Create test equity contract."""
        return EquityContract(symbol="AAPL", sec_type=SecType.STOCK)
    
    @pytest.fixture
    def option_contract(self):
        """Create test option contract."""
        return OptionContract(
            symbol="AAPL",
            sec_type=SecType.OPTION,
            strike=Decimal("150.00"),
            right="CALL",
            expiry="2024-03-15",
            multiplier="100"
        )
    
    def test_market_data_feed_initialization(self, market_data_feed, mock_ib_client):
        """Test MarketDataFeed initialization."""
        assert market_data_feed.ib_client == mock_ib_client
        assert isinstance(market_data_feed._subscriptions, dict)
        assert isinstance(market_data_feed._data_cache, dict)
        assert isinstance(market_data_feed._subscription_callbacks, dict)
    
    @pytest.mark.asyncio
    async def test_subscribe_real_time_data(self, market_data_feed, equity_contract, mock_ib_client):
        """Test real-time data subscription."""
        # Subscribe to real-time data
        await market_data_feed.subscribe_real_time(equity_contract)
        
        # Verify subscription was set up
        assert equity_contract.symbol in market_data_feed._subscriptions
        
        # Verify IB client was called
        mock_ib_client.request_market_data.assert_called_once()
        call_args = mock_ib_client.request_market_data.call_args[1]
        assert call_args["contract"] == equity_contract
    
    @pytest.mark.asyncio
    async def test_unsubscribe_real_time_data(self, market_data_feed, equity_contract, mock_ib_client):
        """Test unsubscribing from real-time data."""
        # First subscribe
        await market_data_feed.subscribe_real_time(equity_contract)
        
        # Then unsubscribe
        await market_data_feed.unsubscribe_real_time(equity_contract)
        
        # Verify subscription was removed
        assert equity_contract.symbol not in market_data_feed._subscriptions
        
        # Verify cancel was called
        mock_ib_client.cancel_market_data.assert_called_once()
    
    def test_get_market_data_returns_cached_data(self, market_data_feed, equity_contract):
        """Test cached market data retrieval."""
        # Cache some test data
        test_data = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02"),
            volume=1000,
            timestamp=datetime.utcnow()
        )
        market_data_feed._data_cache["AAPL"] = test_data
        
        # Retrieve cached data
        result = market_data_feed.get_market_data(equity_contract)
        
        assert result.symbol == "AAPL"
        assert result.last == Decimal("150.02")
        assert result.bid == Decimal("150.00")
        assert result.ask == Decimal("150.05")
    
    def test_get_market_data_returns_none_for_no_cache(self, market_data_feed, equity_contract):
        """Test market data retrieval when no cached data exists."""
        result = market_data_feed.get_market_data(equity_contract)
        assert result is None
    
    def test_update_market_data_caches_data(self, market_data_feed):
        """Test market data update caching."""
        market_data = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02"),
            volume=1000,
            timestamp=datetime.utcnow()
        )
        
        # Update market data
        market_data_feed.update_market_data(market_data)
        
        # Verify data was cached
        cached_data = market_data_feed._data_cache["AAPL"]
        assert cached_data == market_data
    
    def test_get_subscribed_symbols(self, market_data_feed, equity_contract):
        """Test retrieving list of subscribed symbols."""
        # Initially no subscriptions
        symbols = market_data_feed.get_subscribed_symbols()
        assert len(symbols) == 0
        
        # Add subscription manually for testing
        market_data_feed._subscriptions["AAPL"] = 12345
        
        symbols = market_data_feed.get_subscribed_symbols()
        assert "AAPL" in symbols
        assert len(symbols) == 1
    
    def test_is_subscribed(self, market_data_feed, equity_contract):
        """Test checking if symbol is subscribed."""
        # Initially not subscribed
        assert not market_data_feed.is_subscribed(equity_contract)
        
        # Add subscription manually for testing
        market_data_feed._subscriptions["AAPL"] = 12345
        
        # Now should be subscribed
        assert market_data_feed.is_subscribed(equity_contract)
    
    def test_get_all_market_data(self, market_data_feed):
        """Test retrieving all cached market data."""
        # Cache some test data
        data1 = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02"),
            volume=1000,
            timestamp=datetime.utcnow()
        )
        data2 = MarketData(
            symbol="MSFT",
            bid=Decimal("300.00"),
            ask=Decimal("300.10"),
            last=Decimal("300.05"),
            volume=2000,
            timestamp=datetime.utcnow()
        )
        
        market_data_feed._data_cache["AAPL"] = data1
        market_data_feed._data_cache["MSFT"] = data2
        
        # Get all market data
        all_data = market_data_feed.get_all_market_data()
        
        assert len(all_data) == 2
        assert "AAPL" in all_data
        assert "MSFT" in all_data
        assert all_data["AAPL"] == data1
        assert all_data["MSFT"] == data2


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.market_data
class TestMarketDataManager:
    """Test MarketDataManager functionality."""
    
    @pytest.fixture
    def mock_market_data_feed(self):
        """Create mock market data feed."""
        return Mock(spec=MarketDataFeed)
    
    @pytest.fixture
    def mock_position_manager(self):
        """Create mock position manager."""
        return Mock()
    
    @pytest.fixture
    def market_data_manager(self, mock_market_data_feed, mock_position_manager):
        """Create market data manager for testing."""
        return MarketDataManager(
            market_data_feed=mock_market_data_feed,
            position_manager=mock_position_manager
        )
    
    def test_market_data_manager_initialization(self, market_data_manager, mock_market_data_feed, mock_position_manager):
        """Test MarketDataManager initialization."""
        assert market_data_manager.market_data_feed == mock_market_data_feed
        assert market_data_manager.position_manager == mock_position_manager
    
    @pytest.mark.asyncio
    async def test_subscribe_portfolio_positions(self, market_data_manager, mock_market_data_feed, mock_position_manager):
        """Test subscribing to market data for all portfolio positions."""
        # Mock portfolio positions
        positions = [
            Mock(contract=EquityContract(symbol="AAPL", sec_type=SecType.STOCK)),
            Mock(contract=EquityContract(symbol="MSFT", sec_type=SecType.STOCK))
        ]
        mock_position_manager.get_positions_by_account.return_value = positions
        
        # Subscribe to portfolio positions
        await market_data_manager.subscribe_portfolio_positions(account="DU123456")
        
        # Verify subscriptions were made
        assert mock_market_data_feed.subscribe_real_time.call_count == 2
        
        # Check specific calls
        call_args_list = mock_market_data_feed.subscribe_real_time.call_args_list
        called_symbols = [call[0][0].symbol for call in call_args_list]
        assert "AAPL" in called_symbols
        assert "MSFT" in called_symbols
    
    @pytest.mark.asyncio
    async def test_update_position_prices(self, market_data_manager, mock_market_data_feed, mock_position_manager):
        """Test updating position prices from market data."""
        # Mock market data
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("150.00"),
                ask=Decimal("150.05"),
                last=Decimal("150.02"),
                volume=1000,
                timestamp=datetime.utcnow()
            ),
            "MSFT": MarketData(
                symbol="MSFT",
                bid=Decimal("300.00"),
                ask=Decimal("300.10"),
                last=Decimal("300.05"),
                volume=2000,
                timestamp=datetime.utcnow()
            )
        }
        mock_market_data_feed.get_all_market_data.return_value = market_data
        
        # Update position prices
        await market_data_manager.update_position_prices(account="DU123456")
        
        # Verify position manager was called with market prices
        mock_position_manager.update_market_prices.assert_called_once()
        call_args = mock_position_manager.update_market_prices.call_args[0]
        price_dict = call_args[0]
        
        assert "AAPL" in price_dict
        assert "MSFT" in price_dict
        assert price_dict["AAPL"] == Decimal("150.02")
        assert price_dict["MSFT"] == Decimal("300.05")
    
    def test_get_current_prices(self, market_data_manager, mock_market_data_feed):
        """Test getting current prices for symbols."""
        # Mock market data
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("150.00"),
                ask=Decimal("150.05"),
                last=Decimal("150.02"),
                volume=1000,
                timestamp=datetime.utcnow()
            )
        }
        mock_market_data_feed.get_all_market_data.return_value = market_data
        
        # Get current prices
        prices = market_data_manager.get_current_prices(symbols=["AAPL"])
        
        assert "AAPL" in prices
        assert prices["AAPL"] == Decimal("150.02")
    
    def test_get_current_prices_empty_for_no_data(self, market_data_manager, mock_market_data_feed):
        """Test getting current prices when no data available."""
        mock_market_data_feed.get_all_market_data.return_value = {}
        
        prices = market_data_manager.get_current_prices(symbols=["AAPL"])
        
        assert len(prices) == 0
    
    def test_error_handling_invalid_contract(self):
        """Test error handling for invalid contract."""
        # Create a real market data feed for validation testing
        mock_ib_client = Mock()
        feed = MarketDataFeed(ib_client=mock_ib_client)
        
        # Test with None contract
        with pytest.raises(ValueError, match="Contract cannot be None"):
            feed.get_market_data(None)