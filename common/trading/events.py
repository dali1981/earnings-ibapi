"""
Trading event definitions for order lifecycle management.
Integrates with the existing JSONL event bus system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class OrderEventType(str, Enum):
    """Order event types for trading lifecycle."""
    
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_MODIFIED = "ORDER_MODIFIED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    STATUS_UPDATED = "STATUS_UPDATED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ERROR_OCCURRED = "ERROR_OCCURRED"


class PositionEventType(str, Enum):
    """Position event types for position lifecycle."""
    
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_UPDATED = "POSITION_UPDATED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_EXPIRED = "POSITION_EXPIRED"
    MARGIN_CALL = "MARGIN_CALL"


class OrderEvent(BaseModel):
    """Order event for trading system notifications."""
    
    event_type: OrderEventType = Field(..., description="Type of order event")
    order_id: int = Field(..., description="Order ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    
    @validator('order_id')
    def validate_order_id(cls, v):
        """Validate order ID is positive."""
        if v <= 0:
            raise ValueError("Order ID must be positive")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderEvent":
        """Create event from dictionary."""
        return cls(
            event_type=OrderEventType(data["event_type"]),
            order_id=data["order_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {})
        )
    
    def __eq__(self, other):
        """Compare events for equality."""
        if not isinstance(other, OrderEvent):
            return False
        return (
            self.event_type == other.event_type and
            self.order_id == other.order_id and
            self.timestamp == other.timestamp and
            self.data == other.data
        )
    
    def __str__(self):
        """String representation of event."""
        data_str = ", ".join(f"{k}={v}" for k, v in self.data.items())
        return f"OrderEvent({self.event_type.value}, order_id={self.order_id}, {data_str})"


class OrderEventBus:
    """Event bus for order events, integrating with existing event system."""
    
    def __init__(self, event_bus=None):
        """Initialize with optional existing event bus."""
        self.event_bus = event_bus
        self.subscribers = []
    
    def publish(self, event: OrderEvent) -> None:
        """Publish order event to subscribers and main event bus."""
        # Notify local subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception as e:
                # Log error but don't fail the publish
                print(f"Error notifying subscriber: {e}")
        
        # Publish to main event bus if available
        if self.event_bus:
            try:
                self.event_bus.notify_data_available(
                    dataset="trading_orders",
                    event_type=event.event_type.value,
                    order_id=event.order_id,
                    timestamp=event.timestamp.isoformat(),
                    **event.data
                )
            except Exception as e:
                print(f"Error publishing to main event bus: {e}")
    
    def subscribe(self, callback) -> None:
        """Subscribe to order events."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback) -> None:
        """Unsubscribe from order events."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)


class PositionEvent(BaseModel):
    """Position event for trading system notifications."""
    
    event_type: PositionEventType = Field(..., description="Type of position event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionEvent":
        """Create event from dictionary."""
        return cls(
            event_type=PositionEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {})
        )
    
    def __eq__(self, other):
        """Compare events for equality."""
        if not isinstance(other, PositionEvent):
            return False
        return (
            self.event_type == other.event_type and
            self.timestamp == other.timestamp and
            self.data == other.data
        )
    
    def __str__(self):
        """String representation of event."""
        data_str = ", ".join(f"{k}={v}" for k, v in self.data.items())
        return f"PositionEvent({self.event_type.value}, {data_str})"