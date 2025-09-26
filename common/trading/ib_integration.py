"""
Interactive Brokers API integration wrapper.
Provides a clean interface to IB API for order management.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, Union
from datetime import datetime

from common.schemas.trading import (
    Order, OrderAction, OrderType, OrderStatus, TimeInForce,
    EquityContract, OptionContract
)
from common.constants import SecType, OptionRight


class IBConnectionError(Exception):
    """IB connection related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class IBOrderError(Exception):
    """IB order related errors."""
    
    def __init__(self, message: str, order_id: Optional[int] = None):
        super().__init__(message)
        self.order_id = order_id


class IBContract:
    """IB Contract wrapper for compatibility."""
    
    def __init__(self):
        self.symbol = ""
        self.secType = ""
        self.exchange = ""
        self.currency = ""
        self.strike = 0.0
        self.right = ""
        self.lastTradeDateOrContractMonth = ""
        self.multiplier = ""


class IBOrder:
    """IB Order wrapper for compatibility."""
    
    def __init__(self):
        self.action = ""
        self.totalQuantity = 0
        self.orderType = ""
        self.lmtPrice = 0.0
        self.auxPrice = 0.0
        self.tif = ""
        self.clientId = 0
        self.orderId = None


class IBClient:
    """Interactive Brokers API client wrapper."""
    
    def __init__(self, ib_wrapper=None, client_id: int = 1):
        """Initialize IB client with wrapper and client ID."""
        self.ib_wrapper = ib_wrapper
        self.client_id = client_id
        self.next_order_id = 1001  # Start from a reasonable order ID
        self._active_orders: Dict[int, Order] = {}
        self._connected = False
    
    async def connect(self, host: str = "127.0.0.1", port: int = 7497) -> bool:
        """Connect to IB TWS/Gateway."""
        try:
            if self.ib_wrapper:
                result = await self.ib_wrapper.connect(host, port, self.client_id)
                self._connected = result
                return result
            else:
                # Mock connection for testing
                self._connected = True
                return True
        except Exception as e:
            raise IBConnectionError(f"Connection failed: {e}", {
                "host": host, 
                "port": port, 
                "client_id": self.client_id
            })
    
    async def disconnect(self) -> None:
        """Disconnect from IB TWS/Gateway."""
        if self.ib_wrapper:
            await self.ib_wrapper.disconnect()
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        if self.ib_wrapper and hasattr(self.ib_wrapper, 'isConnected'):
            return self.ib_wrapper.isConnected()
        return self._connected
    
    async def place_order(self, order: Order) -> int:
        """Place order with IB and return order ID."""
        if not self.is_connected():
            raise IBConnectionError("Not connected to IB")
        
        try:
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Convert to IB contracts and orders
            ib_contract = self._convert_contract_to_ib(order.contract)
            ib_order = self._convert_order_to_ib(order)
            ib_order.orderId = order_id
            
            # Store order for tracking
            self._active_orders[order_id] = order
            
            # Place order via IB wrapper
            if self.ib_wrapper:
                self.ib_wrapper.placeOrder(order_id, ib_contract, ib_order)
            
            return order_id
            
        except Exception as e:
            raise IBOrderError(f"Failed to place order: {e}")
    
    async def cancel_order(self, order_id: int) -> None:
        """Cancel order by ID."""
        if not self.is_connected():
            raise IBConnectionError("Not connected to IB")
        
        try:
            if self.ib_wrapper:
                self.ib_wrapper.cancelOrder(order_id)
                
            # Remove from active orders
            self._active_orders.pop(order_id, None)
            
        except Exception as e:
            raise IBOrderError(f"Failed to cancel order: {e}", order_id=order_id)
    
    async def modify_order(self, order_id: int, **modifications) -> Order:
        """Modify existing order."""
        if order_id not in self._active_orders:
            raise IBOrderError(f"Order {order_id} not found", order_id=order_id)
        
        try:
            # Get original order
            original_order = self._active_orders[order_id]
            
            # Create modified order
            order_dict = original_order.dict()
            order_dict.update(modifications)
            
            # Handle special cases for proper types
            if 'limit_price' in modifications and modifications['limit_price'] is not None:
                order_dict['limit_price'] = Decimal(str(modifications['limit_price']))
            if 'stop_price' in modifications and modifications['stop_price'] is not None:
                order_dict['stop_price'] = Decimal(str(modifications['stop_price']))
            
            modified_order = Order(**order_dict)
            
            # Cancel original and place new order (IB modification pattern)
            await self.cancel_order(order_id)
            
            # Place modified order with same ID
            ib_contract = self._convert_contract_to_ib(modified_order.contract)
            ib_order = self._convert_order_to_ib(modified_order)
            ib_order.orderId = order_id
            
            if self.ib_wrapper:
                self.ib_wrapper.placeOrder(order_id, ib_contract, ib_order)
            
            # Update stored order
            self._active_orders[order_id] = modified_order
            
            return modified_order
            
        except Exception as e:
            raise IBOrderError(f"Failed to modify order: {e}", order_id=order_id)
    
    def _convert_contract_to_ib(self, contract: Union[EquityContract, OptionContract]) -> IBContract:
        """Convert domain contract to IB contract."""
        ib_contract = IBContract()
        
        ib_contract.symbol = contract.symbol
        ib_contract.exchange = getattr(contract, 'exchange', 'SMART')
        ib_contract.currency = getattr(contract, 'currency', 'USD')
        
        if isinstance(contract, EquityContract):
            ib_contract.secType = "STK"
        elif isinstance(contract, OptionContract):
            ib_contract.secType = "OPT"
            ib_contract.strike = float(contract.strike)
            ib_contract.right = "C" if contract.right == OptionRight.CALL else "P"
            # Convert date to IB format YYYYMMDD
            if hasattr(contract.expiry, 'strftime'):
                ib_contract.lastTradeDateOrContractMonth = contract.expiry.strftime('%Y%m%d')
            else:
                ib_contract.lastTradeDateOrContractMonth = str(contract.expiry)
            ib_contract.multiplier = str(getattr(contract, 'multiplier', 100))
        
        return ib_contract
    
    def _convert_order_to_ib(self, order: Order) -> IBOrder:
        """Convert domain order to IB order."""
        ib_order = IBOrder()
        
        # Basic order fields
        ib_order.action = order.action.value
        ib_order.totalQuantity = order.quantity
        ib_order.clientId = self.client_id
        
        # Order type conversion
        if order.order_type == OrderType.MARKET:
            ib_order.orderType = "MKT"
        elif order.order_type == OrderType.LIMIT:
            ib_order.orderType = "LMT"
            if order.limit_price:
                ib_order.lmtPrice = float(order.limit_price)
        elif order.order_type == OrderType.STOP:
            ib_order.orderType = "STP"
            if order.stop_price:
                ib_order.auxPrice = float(order.stop_price)
        elif order.order_type == OrderType.STOP_LIMIT:
            ib_order.orderType = "STP LMT"
            if order.limit_price:
                ib_order.lmtPrice = float(order.limit_price)
            if order.stop_price:
                ib_order.auxPrice = float(order.stop_price)
        
        # Time in force
        if order.time_in_force:
            if order.time_in_force == TimeInForce.DAY:
                ib_order.tif = "DAY"
            elif order.time_in_force == TimeInForce.GTC:
                ib_order.tif = "GTC"
            elif order.time_in_force == TimeInForce.IOC:
                ib_order.tif = "IOC"
            elif order.time_in_force == TimeInForce.FOK:
                ib_order.tif = "FOK"
        
        return ib_order
    
    def _generate_order_id(self) -> int:
        """Generate next order ID."""
        order_id = self.next_order_id
        self.next_order_id += 1
        return order_id


class IBClientFactory:
    """Factory for creating IB clients with proper configuration."""
    
    @staticmethod
    def create_client(client_id: int = 1, ib_wrapper=None) -> IBClient:
        """Create configured IB client."""
        return IBClient(ib_wrapper=ib_wrapper, client_id=client_id)
    
    @staticmethod
    async def create_connected_client(
        host: str = "127.0.0.1", 
        port: int = 7497, 
        client_id: int = 1,
        ib_wrapper=None
    ) -> IBClient:
        """Create and connect IB client."""
        client = IBClient(ib_wrapper=ib_wrapper, client_id=client_id)
        await client.connect(host, port)
        return client