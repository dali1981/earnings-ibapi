"""
Position management API endpoints.
"""

from __future__ import annotations

from ..schemas.trading import PositionResponse, PositionsResponse, PortfolioResponse
from common.trading import PositionManagerFactory, PositionManagerError
from common.observability import monitor_api_endpoint, capture_trading_event
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

import logging


router = APIRouter(prefix="/positions", tags=["positions"])
logger = logging.getLogger(__name__)


@router.get("", response_model=PositionsResponse)
@monitor_api_endpoint("get_all_positions")
async def get_positions():
    """Get all current positions."""
    try:
        position_manager = PositionManagerFactory.create_position_manager()
        
        positions = position_manager.get_all_positions()
        
        # Convert to response format
        position_responses = []
        for pos in positions:
            position_responses.append(PositionResponse(
                contract=pos["contract"],
                position_size=pos["position_size"],
                avg_cost=pos["avg_cost"],
                market_price=pos["market_price"],
                market_value=pos["market_value"],
                unrealized_pnl=pos["unrealized_pnl"],
                realized_pnl=pos.get("realized_pnl", 0),
                last_updated=pos.get("last_updated", datetime.utcnow())
            ))
        
        capture_trading_event(
            f"Retrieved {len(positions)} positions",
            level="info",
            extra={"positions_count": len(positions)},
            tags={"operation": "get_all_positions"}
        )
        
        return PositionsResponse(
            positions=position_responses,
            total_positions=len(positions),
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        capture_trading_event(
            f"Failed to get positions: {str(e)}",
            level="error",
            extra={"error": str(e)},
            tags={"operation": "get_all_positions", "error_type": "fetch_failed"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve positions"
        )


@router.get("/{symbol}", response_model=PositionResponse)
@monitor_api_endpoint("get_position", extract_symbol=lambda args, kwargs: args[0] if args else None)
async def get_position(symbol: str):
    """Get position for a specific symbol."""
    try:
        position_manager = PositionManagerFactory.create_position_manager()
        
        position = position_manager.get_position(symbol)
        
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for symbol {symbol}"
            )
        
        capture_trading_event(
            f"Retrieved position for {symbol}",
            level="info",
            extra={"symbol": symbol, "position_size": position["position_size"]},
            tags={"operation": "get_position", "symbol": symbol}
        )
        
        return PositionResponse(
            contract=position["contract"],
            position_size=position["position_size"],
            avg_cost=position["avg_cost"],
            market_price=position["market_price"],
            market_value=position["market_value"],
            unrealized_pnl=position["unrealized_pnl"],
            realized_pnl=position.get("realized_pnl", 0),
            last_updated=position.get("last_updated", datetime.utcnow())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {str(e)}")
        capture_trading_event(
            f"Failed to get position for {symbol}: {str(e)}",
            level="error",
            extra={"symbol": symbol, "error": str(e)},
            tags={"operation": "get_position", "symbol": symbol, "error_type": "fetch_failed"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve position for {symbol}"
        )


@router.get("/portfolio/summary", response_model=PortfolioResponse)
@monitor_api_endpoint("get_portfolio_summary")
async def get_portfolio_summary():
    """Get portfolio summary."""
    try:
        position_manager = PositionManagerFactory.create_position_manager()
        
        summary = position_manager.get_portfolio_summary()
        
        capture_trading_event(
            "Retrieved portfolio summary",
            level="info",
            extra={
                "total_value": float(summary["total_value"]),
                "unrealized_pnl": float(summary["unrealized_pnl"]),
                "positions_count": len(summary.get("positions", []))
            },
            tags={"operation": "get_portfolio_summary"}
        )
        
        return PortfolioResponse(
            total_value=summary["total_value"],
            buying_power=summary["buying_power"],
            net_liquidation=summary["net_liquidation"],
            unrealized_pnl=summary["unrealized_pnl"],
            realized_pnl=summary["realized_pnl"],
            positions_count=len(summary.get("positions", [])),
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {str(e)}")
        capture_trading_event(
            f"Failed to get portfolio summary: {str(e)}",
            level="error",
            extra={"error": str(e)},
            tags={"operation": "get_portfolio_summary", "error_type": "fetch_failed"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio summary"
        )