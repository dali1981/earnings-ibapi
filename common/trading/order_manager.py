"""
Core order management service for trading operations.
Handles order lifecycle, validation, and IB API integration.
"""

import asyncio
from decimal import Decimal
from typing import List, Optional, Dict, Any
from datetime import datetime

from common.schemas.trading import (
    Order, OrderAction, OrderType, OrderStatus, TimeInForce,
    EquityContract, OptionContract
)
from common.repositories.trading.orders import OrderRepository
from .ib_integration import IBClient, IBConnectionError, IBOrderError
from .events import OrderEvent, OrderEventType, OrderEventBus


class OrderError(Exception):
    """General order management errors."""
    pass


class OrderValidationError(OrderError):
    """Order validation specific errors."""
    pass


class OrderManager:
    """Core order management service."""
    
    def __init__(
        self, 
        order_repository: OrderRepository,
        ib_client: IBClient,
        event_bus: Optional[OrderEventBus] = None
    ):
        """Initialize order manager with dependencies."""
        self.order_repository = order_repository
        self.ib_client = ib_client
        self.event_bus = event_bus or OrderEventBus()
        self.next_order_id = 1001  # Starting order ID
    
    async def submit_order(self, order: Order) -> Order:
        """Submit order to IB and save to repository."""
        try:
            # Validate order
            self._validate_order(order)
            
            # Submit to IB
            try:
                ib_order_id = await self.ib_client.place_order(order)
            except (IBConnectionError, IBOrderError) as e:
                raise OrderError(f"IB connection error: {e}")
            
            # Update order with IB order ID and status
            order.order_id = ib_order_id
            order.status = OrderStatus.PENDING_SUBMIT
            
            # Save to repository
            self.order_repository.domain_save(order)
            
            # Publish event
            event = OrderEvent(
                event_type=OrderEventType.ORDER_SUBMITTED,
                order_id=ib_order_id,
                data={
                    "client_order_id": order.client_order_id,
                    "symbol": order.contract.symbol,
                    "action": order.action.value,
                    "quantity": order.quantity,
                    "order_type": order.order_type.value
                }
            )
            self.event_bus.publish(event)
            
            return order
            
        except OrderValidationError:
            raise
        except Exception as e:
            raise OrderError(f"Failed to submit order: {e}")
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel order by ID."""
        try:
            # Get order from repository
            orders = self.order_repository.get_orders()
            order = None
            for o in orders:
                if o.order_id == order_id:
                    order = o
                    break
            
            if not order:
                raise OrderError(f"Order {order_id} not found")
            
            # Check if order can be cancelled
            if not self._can_cancel_order(order):
                raise OrderError(f"Cannot cancel order {order_id} with status {order.status}")
            
            # Cancel with IB
            await self.ib_client.cancel_order(order_id)
            
            # Update status in repository
            self.order_repository.update_order_status(order_id, OrderStatus.PENDING_CANCEL)
            
            # Publish event
            event = OrderEvent(
                event_type=OrderEventType.CANCEL_REQUESTED,
                order_id=order_id,
                data={"previous_status": order.status.value}
            )
            self.event_bus.publish(event)
            
            return True
            
        except Exception as e:
            raise OrderError(f"Failed to cancel order: {e}")
    
    async def modify_order(
        self, 
        order_id: int, 
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        quantity: Optional[int] = None
    ) -> Order:
        """Modify existing order."""
        try:
            # Get order from repository
            orders = self.order_repository.get_orders()
            order = None
            for o in orders:
                if o.order_id == order_id:
                    order = o
                    break
            
            if not order:
                raise OrderError(f"Order {order_id} not found")
            
            # Check if order can be modified
            if not self._can_modify_order(order):
                raise OrderError(f"Cannot modify order {order_id} with status {order.status}")
            
            # Prepare modifications
            modifications = {}
            if limit_price is not None:
                modifications['limit_price'] = limit_price
            if stop_price is not None:
                modifications['stop_price'] = stop_price
            if quantity is not None:
                modifications['quantity'] = quantity
            
            # Modify with IB
            modified_order = await self.ib_client.modify_order(order_id, **modifications)
            
            # Save modified order
            self.order_repository.domain_save(modified_order)
            
            # Publish event
            event = OrderEvent(
                event_type=OrderEventType.ORDER_MODIFIED,
                order_id=order_id,
                data={
                    "modifications": modifications,
                    "previous_limit_price": float(order.limit_price) if order.limit_price else None,
                    "previous_quantity": order.quantity
                }
            )
            self.event_bus.publish(event)
            
            return modified_order
            
        except Exception as e:
            raise OrderError(f"Failed to modify order: {e}")
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders filtered by status."""
        return self.order_repository.get_orders(status=status)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders filtered by symbol."""
        return self.order_repository.get_orders_by_symbol(symbol)
    
    def get_order_by_id(self, order_id: int) -> Optional[Order]:
        """Get single order by ID."""
        orders = self.order_repository.get_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None
    
    async def handle_order_status_update(
        self, 
        order_id: int, 
        status: OrderStatus,
        filled_qty: int = 0,
        avg_fill_price: Optional[float] = None
    ) -> None:
        """Handle order status updates from IB."""
        try:
            # Update in repository
            self.order_repository.update_order_status(
                order_id, status, filled_qty, avg_fill_price
            )
            
            # Determine event type based on status
            if status == OrderStatus.FILLED:
                event_type = OrderEventType.ORDER_FILLED
            elif status == OrderStatus.CANCELLED:
                event_type = OrderEventType.ORDER_CANCELLED
            elif status == OrderStatus.REJECTED:
                event_type = OrderEventType.ORDER_REJECTED
            else:
                event_type = OrderEventType.STATUS_UPDATED
            
            # Publish event
            event = OrderEvent(
                event_type=event_type,
                order_id=order_id,
                data={
                    "status": status.value,
                    "filled_qty": filled_qty,
                    "avg_fill_price": avg_fill_price
                }
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            # Publish error event
            error_event = OrderEvent(
                event_type=OrderEventType.ERROR_OCCURRED,
                order_id=order_id,
                data={"error": str(e), "context": "status_update"}
            )
            self.event_bus.publish(error_event)
            raise OrderError(f"Failed to handle status update: {e}")
    
    def _validate_order(self, order: Order) -> None:
        """Validate order before submission."""
        # Basic validation
        if order.quantity <= 0:
            raise OrderValidationError("Quantity must be positive")
        
        if not order.contract or not order.contract.symbol:
            raise OrderValidationError("Valid contract with symbol required")
        
        # Order type specific validation
        if order.order_type == OrderType.LIMIT:
            if not order.limit_price or order.limit_price <= 0:
                raise OrderValidationError("Limit price required for limit orders")
        
        elif order.order_type == OrderType.STOP:
            if not order.stop_price or order.stop_price <= 0:
                raise OrderValidationError("Stop price required for stop orders")
        
        elif order.order_type == OrderType.STOP_LIMIT:
            if not order.limit_price or order.limit_price <= 0:
                raise OrderValidationError("Limit price required for stop-limit orders")
            if not order.stop_price or order.stop_price <= 0:
                raise OrderValidationError("Stop price required for stop-limit orders")
        
        # Action validation
        if order.action not in [OrderAction.BUY, OrderAction.SELL]:
            raise OrderValidationError(f"Invalid order action: {order.action}")
    
    def _can_cancel_order(self, order: Order) -> bool:
        """Check if order can be cancelled."""
        cancellable_statuses = [
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.SUBMITTED,
            OrderStatus.PENDING_CANCEL
        ]
        return order.status in cancellable_statuses
    
    def _can_modify_order(self, order: Order) -> bool:
        """Check if order can be modified."""
        modifiable_statuses = [
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.SUBMITTED
        ]
        return order.status in modifiable_statuses
    
    def _generate_order_id(self) -> int:
        """Generate next order ID."""
        order_id = self.next_order_id
        self.next_order_id += 1
        return order_id


class OrderManagerFactory:
    """Factory for creating order managers with proper configuration."""
    
    @staticmethod
    def create_order_manager(
        data_dir: str,
        ib_client: IBClient,
        event_bus: Optional[OrderEventBus] = None
    ) -> OrderManager:
        """Create configured order manager."""
        order_repository = OrderRepository(data_dir)
        return OrderManager(
            order_repository=order_repository,
            ib_client=ib_client,
            event_bus=event_bus
        )