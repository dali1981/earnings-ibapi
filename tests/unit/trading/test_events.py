"""
Unit tests for trading events.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

# These imports will fail initially (Red phase)
from common.trading.events import OrderEvent, OrderEventType


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestOrderEvent:
    """Test OrderEvent functionality."""
    
    def test_order_event_creation(self):
        """Test basic OrderEvent creation."""
        event_type = OrderEventType.ORDER_SUBMITTED
        order_id = 12345
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        data = {"symbol": "AAPL", "quantity": 100}
        
        event = OrderEvent(
            event_type=event_type,
            order_id=order_id,
            timestamp=timestamp,
            data=data
        )
        
        assert event.event_type == event_type
        assert event.order_id == order_id
        assert event.timestamp == timestamp
        assert event.data == data
    
    def test_order_event_type_enum_values(self):
        """Test OrderEventType enum has all required values."""
        required_types = [
            "ORDER_SUBMITTED",
            "ORDER_MODIFIED", 
            "CANCEL_REQUESTED",
            "STATUS_UPDATED",
            "ORDER_FILLED",
            "ORDER_CANCELLED",
            "ORDER_REJECTED",
            "ERROR_OCCURRED"
        ]
        
        for event_type in required_types:
            assert hasattr(OrderEventType, event_type)
            assert isinstance(getattr(OrderEventType, event_type), OrderEventType)
    
    def test_order_event_serialization(self):
        """Test OrderEvent can be serialized to dict."""
        event = OrderEvent(
            event_type=OrderEventType.ORDER_FILLED,
            order_id=12345,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            data={"filled_qty": 100, "avg_price": 150.25}
        )
        
        serialized = event.to_dict()
        
        assert isinstance(serialized, dict)
        assert serialized["event_type"] == "ORDER_FILLED"
        assert serialized["order_id"] == 12345
        assert "timestamp" in serialized
        assert serialized["data"]["filled_qty"] == 100
    
    def test_order_event_deserialization(self):
        """Test OrderEvent can be created from dict."""
        event_dict = {
            "event_type": "STATUS_UPDATED",
            "order_id": 12345,
            "timestamp": "2024-01-15T10:30:00",
            "data": {"old_status": "SUBMITTED", "new_status": "FILLED"}
        }
        
        event = OrderEvent.from_dict(event_dict)
        
        assert event.event_type == OrderEventType.STATUS_UPDATED
        assert event.order_id == 12345
        assert isinstance(event.timestamp, datetime)
        assert event.data["old_status"] == "SUBMITTED"
    
    def test_order_event_validation(self):
        """Test OrderEvent validation."""
        # Valid event should not raise
        OrderEvent(
            event_type=OrderEventType.ORDER_SUBMITTED,
            order_id=12345,
            timestamp=datetime.utcnow(),
            data={}
        )
        
        # Invalid order_id
        with pytest.raises(ValueError):
            OrderEvent(
                event_type=OrderEventType.ORDER_SUBMITTED,
                order_id=-1,  # Invalid
                timestamp=datetime.utcnow(),
                data={}
            )
    
    def test_order_event_optional_data(self):
        """Test OrderEvent with optional data parameter."""
        event = OrderEvent(
            event_type=OrderEventType.ORDER_SUBMITTED,
            order_id=12345,
            timestamp=datetime.utcnow()
            # data is optional
        )
        
        assert event.data == {}
    
    def test_order_event_equality(self):
        """Test OrderEvent equality comparison."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        data = {"test": "data"}
        
        event1 = OrderEvent(
            event_type=OrderEventType.ORDER_SUBMITTED,
            order_id=12345,
            timestamp=timestamp,
            data=data
        )
        
        event2 = OrderEvent(
            event_type=OrderEventType.ORDER_SUBMITTED,
            order_id=12345,
            timestamp=timestamp,
            data=data
        )
        
        event3 = OrderEvent(
            event_type=OrderEventType.ORDER_FILLED,
            order_id=12345,
            timestamp=timestamp,
            data=data
        )
        
        assert event1 == event2
        assert event1 != event3
    
    def test_order_event_string_representation(self):
        """Test OrderEvent string representation."""
        event = OrderEvent(
            event_type=OrderEventType.ORDER_SUBMITTED,
            order_id=12345,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            data={"symbol": "AAPL"}
        )
        
        str_repr = str(event)
        assert "ORDER_SUBMITTED" in str_repr
        assert "12345" in str_repr
        assert "AAPL" in str_repr


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestOrderEventType:
    """Test OrderEventType enum."""
    
    def test_order_event_type_values(self):
        """Test OrderEventType has correct string values."""
        assert OrderEventType.ORDER_SUBMITTED.value == "ORDER_SUBMITTED"
        assert OrderEventType.ORDER_MODIFIED.value == "ORDER_MODIFIED"
        assert OrderEventType.CANCEL_REQUESTED.value == "CANCEL_REQUESTED"
        assert OrderEventType.STATUS_UPDATED.value == "STATUS_UPDATED"
        assert OrderEventType.ORDER_FILLED.value == "ORDER_FILLED"
        assert OrderEventType.ORDER_CANCELLED.value == "ORDER_CANCELLED"
        assert OrderEventType.ORDER_REJECTED.value == "ORDER_REJECTED"
        assert OrderEventType.ERROR_OCCURRED.value == "ERROR_OCCURRED"
    
    def test_order_event_type_from_string(self):
        """Test creating OrderEventType from string."""
        assert OrderEventType("ORDER_SUBMITTED") == OrderEventType.ORDER_SUBMITTED
        assert OrderEventType("ORDER_FILLED") == OrderEventType.ORDER_FILLED
        
        # Invalid string should raise ValueError
        with pytest.raises(ValueError):
            OrderEventType("INVALID_EVENT_TYPE")
    
    def test_order_event_type_completeness(self):
        """Test that all order lifecycle events are covered."""
        # Core order lifecycle events
        lifecycle_events = [
            OrderEventType.ORDER_SUBMITTED,
            OrderEventType.STATUS_UPDATED,
            OrderEventType.ORDER_FILLED,
            OrderEventType.ORDER_CANCELLED,
            OrderEventType.ORDER_REJECTED
        ]
        
        # Management events
        management_events = [
            OrderEventType.ORDER_MODIFIED,
            OrderEventType.CANCEL_REQUESTED,
            OrderEventType.ERROR_OCCURRED
        ]
        
        all_events = lifecycle_events + management_events
        
        # Verify all events are defined
        for event in all_events:
            assert isinstance(event, OrderEventType)
            assert isinstance(event.value, str)
            assert event.value.upper() == event.value  # Should be uppercase