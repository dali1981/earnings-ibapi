"""
Market data feed and management for trading operations.
Provides real-time market data integration with IB API.
"""

from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from common.schemas.trading import EquityContract, OptionContract
from common.schemas.base import BaseAPIModel


class MarketData(BaseAPIModel):
    """Real-time market data for a symbol."""
    
    symbol: str = Field(..., description="Symbol identifier")
    bid: Optional[Decimal] = Field(None, description="Best bid price")
    ask: Optional[Decimal] = Field(None, description="Best ask price")  
    last: Optional[Decimal] = Field(None, description="Last trade price")
    volume: Optional[int] = Field(None, description="Volume traded")
    high: Optional[Decimal] = Field(None, description="Day high price")
    low: Optional[Decimal] = Field(None, description="Day low price")
    open_price: Optional[Decimal] = Field(None, description="Opening price")
    close_price: Optional[Decimal] = Field(None, description="Previous close price")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data timestamp")
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price between bid and ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / Decimal("2")
        return None


class MarketDataError(Exception):
    """Market data related errors."""
    pass


class MarketDataFeed:
    """Real-time market data feed using IB API."""
    
    def __init__(self, ib_client):
        """Initialize market data feed."""
        self.ib_client = ib_client
        self._subscriptions: Dict[str, int] = {}  # symbol -> request_id
        self._data_cache: Dict[str, MarketData] = {}  # symbol -> MarketData
        self._subscription_callbacks: Dict[int, Callable] = {}  # request_id -> callback
        self._next_request_id = 1000
    
    async def subscribe_real_time(self, contract: Union[EquityContract, OptionContract]) -> None:
        """Subscribe to real-time market data for a contract."""
        if contract is None:
            raise ValueError("Contract cannot be None")
        
        if not self.ib_client.is_connected():
            raise MarketDataError("IB client is not connected")
        
        symbol = contract.symbol
        if symbol in self._subscriptions:
            # Already subscribed
            return
        
        # Get next request ID
        request_id = self._next_request_id
        self._next_request_id += 1
        
        # Store subscription
        self._subscriptions[symbol] = request_id
        
        # Set up callback for this subscription
        self._subscription_callbacks[request_id] = self._handle_market_data_update
        
        # Request market data from IB
        await self.ib_client.request_market_data(
            contract=contract,
            request_id=request_id,
            snapshot=False,
            regulatory_snapshot=False
        )
    
    async def unsubscribe_real_time(self, contract: Union[EquityContract, OptionContract]) -> None:
        """Unsubscribe from real-time market data."""
        if contract is None:
            raise ValueError("Contract cannot be None")
        
        symbol = contract.symbol
        if symbol not in self._subscriptions:
            return
        
        request_id = self._subscriptions[symbol]
        
        # Cancel market data from IB
        await self.ib_client.cancel_market_data(request_id=request_id)
        
        # Clean up subscription tracking
        del self._subscriptions[symbol]
        if request_id in self._subscription_callbacks:
            del self._subscription_callbacks[request_id]
        
        # Optionally keep cached data or remove it
        # For now, we'll keep cached data for historical reference
    
    def get_market_data(self, contract: Union[EquityContract, OptionContract]) -> Optional[MarketData]:
        """Get cached market data for a contract."""
        if contract is None:
            raise ValueError("Contract cannot be None")
        
        return self._data_cache.get(contract.symbol)
    
    def update_market_data(self, market_data: MarketData) -> None:
        """Update cached market data."""
        if market_data is None:
            raise ValueError("Market data cannot be None")
        
        self._data_cache[market_data.symbol] = market_data
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self._subscriptions.keys())
    
    def is_subscribed(self, contract: Union[EquityContract, OptionContract]) -> bool:
        """Check if contract is currently subscribed."""
        if contract is None:
            return False
        return contract.symbol in self._subscriptions
    
    def get_all_market_data(self) -> Dict[str, MarketData]:
        """Get all cached market data."""
        return self._data_cache.copy()
    
    def _handle_market_data_update(self, request_id: int, field: int, value: float, symbol: str) -> None:
        """Handle market data update from IB API."""
        # This would be called by IB API callbacks
        # For now, we'll provide a basic implementation
        
        if symbol not in self._data_cache:
            self._data_cache[symbol] = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow()
            )
        
        # Update specific field based on IB field codes
        market_data = self._data_cache[symbol]
        
        # IB field codes (simplified mapping)
        if field == 1:  # Bid
            market_data.bid = Decimal(str(value))
        elif field == 2:  # Ask  
            market_data.ask = Decimal(str(value))
        elif field == 4:  # Last
            market_data.last = Decimal(str(value))
        elif field == 7:  # Volume
            market_data.volume = int(value)
        elif field == 6:  # High
            market_data.high = Decimal(str(value))
        elif field == 7:  # Low
            market_data.low = Decimal(str(value))
        elif field == 14:  # Open
            market_data.open_price = Decimal(str(value))
        elif field == 9:  # Close
            market_data.close_price = Decimal(str(value))
        
        # Update timestamp
        market_data.timestamp = datetime.utcnow()


class MarketDataManager:
    """High-level market data management for trading operations."""
    
    def __init__(self, market_data_feed: MarketDataFeed, position_manager=None):
        """Initialize market data manager."""
        self.market_data_feed = market_data_feed
        self.position_manager = position_manager
    
    async def subscribe_portfolio_positions(self, account: str) -> None:
        """Subscribe to market data for all positions in portfolio."""
        if not self.position_manager:
            raise MarketDataError("Position manager not configured")
        
        # Get current positions
        positions = self.position_manager.get_positions_by_account(account)
        
        # Subscribe to market data for each position
        for position in positions:
            await self.market_data_feed.subscribe_real_time(position.contract)
    
    async def update_position_prices(self, account: str) -> None:
        """Update position prices from current market data."""
        if not self.position_manager:
            raise MarketDataError("Position manager not configured")
        
        # Get all current market data
        all_market_data = self.market_data_feed.get_all_market_data()
        
        # Create price dictionary for position manager
        price_dict = {}
        for symbol, market_data in all_market_data.items():
            if market_data.last is not None:
                price_dict[symbol] = market_data.last
        
        # Update position prices
        if price_dict:
            self.position_manager.update_market_prices(price_dict, account=account)
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        """Get current prices for specified symbols."""
        all_data = self.market_data_feed.get_all_market_data()
        
        prices = {}
        for symbol in symbols:
            if symbol in all_data and all_data[symbol].last is not None:
                prices[symbol] = all_data[symbol].last
        
        return prices
    
    def get_market_data_summary(self) -> Dict[str, Any]:
        """Get summary of current market data state."""
        all_data = self.market_data_feed.get_all_market_data()
        subscriptions = self.market_data_feed.get_subscribed_symbols()
        
        return {
            "total_cached_symbols": len(all_data),
            "active_subscriptions": len(subscriptions),
            "subscribed_symbols": subscriptions,
            "cached_symbols": list(all_data.keys()),
            "last_update": max(
                [data.timestamp for data in all_data.values()],
                default=None
            )
        }


# Factory function for creating market data components
class MarketDataFactory:
    """Factory for creating market data components."""
    
    @staticmethod
    def create_market_data_feed(ib_client) -> MarketDataFeed:
        """Create market data feed with IB client."""
        return MarketDataFeed(ib_client=ib_client)
    
    @staticmethod
    def create_market_data_manager(ib_client, position_manager=None) -> MarketDataManager:
        """Create market data manager with dependencies."""
        feed = MarketDataFactory.create_market_data_feed(ib_client)
        return MarketDataManager(
            market_data_feed=feed,
            position_manager=position_manager
        )