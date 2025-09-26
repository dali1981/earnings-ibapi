"""
Order management API endpoints.
"""

from __future__ import annotations

from ..schemas.trading import (
    CreateOrderRequest, CreateOrderResponse, OrderStatusResponse, ErrorResponse
)
from common.trading import OrderManagerFactory, OrderError, OrderValidationError
from common.observability import monitor_api_endpoint, capture_trading_event
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

import logging


router = APIRouter(prefix="/orders", tags=["orders"])
logger = logging.getLogger(__name__)


@router.post("", response_model=CreateOrderResponse, status_code=status.HTTP_201_CREATED)
@monitor_api_endpoint("create_order")
async def create_order(order_request: CreateOrderRequest):
    """Create a new trading order."""
    try:
        # Create order manager
        order_manager = OrderManagerFactory.create_order_manager()
        
        # Convert request to order object
        order_data = {
            "contract": order_request.contract,
            "action": order_request.action,
            "quantity": order_request.quantity,
            "order_type": order_request.order_type,
            "limit_price": order_request.limit_price,
            "stop_price": order_request.stop_price,
            "time_in_force": order_request.time_in_force,
            "account": order_request.account
        }
        
        # Create order
        result = order_manager.create_order(**order_data)
        
        # Log successful creation
        capture_trading_event(
            f"Order created successfully: {result.get('client_order_id')}",
            level="info",
            extra={
                "order_id": result.get("order_id"),
                "symbol": order_request.contract.symbol,
                "action": order_request.action,
                "quantity": order_request.quantity
            },
            tags={"operation": "create_order", "symbol": order_request.contract.symbol}
        )
        
        return CreateOrderResponse(
            order_id=result["order_id"],
            client_order_id=result["client_order_id"],
            status=result["status"],
            created_at=result.get("created_at", datetime.utcnow()),
            message=f"Order {result['order_id']} created successfully"
        )
        
    except OrderValidationError as e:
        capture_trading_event(
            f"Order validation failed: {str(e)}",
            level="warning",
            extra={"symbol": order_request.contract.symbol, "error": str(e)},
            tags={"operation": "create_order", "error_type": "validation"}
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Order validation failed: {str(e)}"
        )
    
    except OrderError as e:
        capture_trading_event(
            f"Order creation failed: {str(e)}",
            level="error",
            extra={"symbol": order_request.contract.symbol, "error": str(e)},
            tags={"operation": "create_order", "error_type": "order_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Order creation failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error creating order: {str(e)}", exc_info=True)
        capture_trading_event(
            f"Unexpected error creating order: {str(e)}",
            level="error",
            extra={"symbol": order_request.contract.symbol, "error": str(e)},
            tags={"operation": "create_order", "error_type": "internal_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{order_id}/status", response_model=OrderStatusResponse)
@monitor_api_endpoint("get_order_status")
async def get_order_status(order_id: int):
    """Get status of a specific order."""
    try:
        order_manager = OrderManagerFactory.create_order_manager()
        
        status_info = order_manager.get_order_status(order_id)
        
        capture_trading_event(
            f"Order status retrieved: {order_id}",
            level="info",
            extra={"order_id": order_id, "status": status_info.get("status")},
            tags={"operation": "get_order_status"}
        )
        
        return OrderStatusResponse(
            order_id=order_id,
            status=status_info["status"],
            filled_quantity=status_info.get("filled_quantity", 0),
            remaining_quantity=status_info.get("remaining_quantity", 0),
            avg_fill_price=status_info.get("avg_fill_price"),
            last_updated=status_info.get("last_updated", datetime.utcnow()),
            commission=status_info.get("commission")
        )
        
    except Exception as e:
        logger.error(f"Error getting order status for {order_id}: {str(e)}")
        capture_trading_event(
            f"Failed to get order status: {str(e)}",
            level="error",
            extra={"order_id": order_id, "error": str(e)},
            tags={"operation": "get_order_status", "error_type": "not_found"}
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found or error retrieving status"
        )


@router.delete("/{order_id}", response_model=OrderStatusResponse)
@monitor_api_endpoint("cancel_order")
async def cancel_order(order_id: int):
    """Cancel an existing order."""
    try:
        order_manager = OrderManagerFactory.create_order_manager()
        
        result = order_manager.cancel_order(order_id)
        
        capture_trading_event(
            f"Order cancelled: {order_id}",
            level="info",
            extra={"order_id": order_id},
            tags={"operation": "cancel_order"}
        )
        
        return OrderStatusResponse(
            order_id=order_id,
            status=result["status"],
            filled_quantity=result.get("filled_quantity", 0),
            remaining_quantity=result.get("remaining_quantity", 0),
            avg_fill_price=result.get("avg_fill_price"),
            last_updated=datetime.utcnow(),
            commission=result.get("commission")
        )
        
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {str(e)}")
        capture_trading_event(
            f"Failed to cancel order: {str(e)}",
            level="error",
            extra={"order_id": order_id, "error": str(e)},
            tags={"operation": "cancel_order", "error_type": "cancel_failed"}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to cancel order {order_id}: {str(e)}"
        )