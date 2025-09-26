"""
Strategy management API endpoints.
"""

from __future__ import annotations

from ..schemas.trading import (
    StrategyStatusResponse, StrategyActionResponse, SignalGenerationResponse,
    StrategyParametersRequest
)
from common.trading import StrategyFactory, StrategyError
from common.observability import monitor_api_endpoint, capture_trading_event
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import time

import logging


router = APIRouter(prefix="/strategies", tags=["strategies"])
logger = logging.getLogger(__name__)


@router.get("/status", response_model=StrategyStatusResponse)
@monitor_api_endpoint("get_strategy_status")
async def get_strategy_status():
    """Get status of all strategies."""
    try:
        # Create strategy manager with minimal dependencies for status
        strategy_manager = StrategyFactory.create_strategy_manager(
            strategy_engine=StrategyFactory.create_strategy_engine(),
            market_data_manager=None,
            order_manager=None,
            risk_manager=None
        )
        
        status_info = strategy_manager.get_manager_status()
        engine_status = status_info["engine_status"]
        
        capture_trading_event(
            "Retrieved strategy status",
            level="info",
            extra={
                "total_strategies": engine_status["total_strategies"],
                "enabled_strategies": len(engine_status["enabled_strategies"])
            },
            tags={"operation": "get_strategy_status"}
        )
        
        return StrategyStatusResponse(
            enabled=engine_status["enabled"],
            total_strategies=engine_status["total_strategies"],
            enabled_strategies=engine_status["enabled_strategies"],
            disabled_strategies=engine_status["disabled_strategies"],
            total_signals_generated=engine_status["total_signals_generated"],
            last_signal_time=None  # Would be populated from execution history
        )
        
    except Exception as e:
        logger.error(f"Error getting strategy status: {str(e)}")
        capture_trading_event(
            f"Failed to get strategy status: {str(e)}",
            level="error",
            extra={"error": str(e)},
            tags={"operation": "get_strategy_status", "error_type": "fetch_failed"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy status"
        )


@router.post("/signals/generate", response_model=SignalGenerationResponse)
@monitor_api_endpoint("generate_signals")
async def generate_signals():
    """Manually trigger signal generation from all enabled strategies."""
    try:
        start_time = time.time()
        
        # Create strategy manager with dependencies
        strategy_manager = StrategyFactory.create_strategy_manager(
            strategy_engine=StrategyFactory.create_strategy_engine(),
            market_data_manager=None,  # Would be injected in production
            order_manager=None,        # Would be injected in production
            risk_manager=None          # Would be injected in production
        )
        
        # Run strategies
        results = await strategy_manager.run_strategies()
        
        execution_time = time.time() - start_time
        
        capture_trading_event(
            f"Generated {len(results)} signals",
            level="info",
            extra={
                "signals_count": len(results),
                "execution_time": execution_time
            },
            tags={"operation": "generate_signals"}
        )
        
        return SignalGenerationResponse(
            results=results,
            total_signals=len(results),
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        capture_trading_event(
            f"Failed to generate signals: {str(e)}",
            level="error",
            extra={"error": str(e)},
            tags={"operation": "generate_signals", "error_type": "generation_failed"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate signals"
        )


@router.post("/{strategy_name}/enable", response_model=StrategyActionResponse)
@monitor_api_endpoint("enable_strategy")
async def enable_strategy(strategy_name: str):
    """Enable a specific strategy."""
    try:
        strategy_manager = StrategyFactory.create_strategy_manager(
            strategy_engine=StrategyFactory.create_strategy_engine(),
            market_data_manager=None,
            order_manager=None,
            risk_manager=None
        )
        
        strategy_manager.strategy_engine.enable_strategy(strategy_name)
        
        capture_trading_event(
            f"Strategy enabled: {strategy_name}",
            level="info",
            extra={"strategy_name": strategy_name},
            tags={"operation": "enable_strategy", "strategy": strategy_name}
        )
        
        return StrategyActionResponse(
            strategy_name=strategy_name,
            enabled=True,
            action_time=datetime.utcnow(),
            message=f"Strategy {strategy_name} enabled successfully"
        )
        
    except StrategyError as e:
        capture_trading_event(
            f"Failed to enable strategy {strategy_name}: {str(e)}",
            level="warning",
            extra={"strategy_name": strategy_name, "error": str(e)},
            tags={"operation": "enable_strategy", "strategy": strategy_name, "error_type": "strategy_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_name} not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error enabling strategy {strategy_name}: {str(e)}")
        capture_trading_event(
            f"Failed to enable strategy {strategy_name}: {str(e)}",
            level="error",
            extra={"strategy_name": strategy_name, "error": str(e)},
            tags={"operation": "enable_strategy", "strategy": strategy_name, "error_type": "internal_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enable strategy"
        )


@router.post("/{strategy_name}/disable", response_model=StrategyActionResponse)
@monitor_api_endpoint("disable_strategy")
async def disable_strategy(strategy_name: str):
    """Disable a specific strategy."""
    try:
        strategy_manager = StrategyFactory.create_strategy_manager(
            strategy_engine=StrategyFactory.create_strategy_engine(),
            market_data_manager=None,
            order_manager=None,
            risk_manager=None
        )
        
        strategy_manager.strategy_engine.disable_strategy(strategy_name)
        
        capture_trading_event(
            f"Strategy disabled: {strategy_name}",
            level="info",
            extra={"strategy_name": strategy_name},
            tags={"operation": "disable_strategy", "strategy": strategy_name}
        )
        
        return StrategyActionResponse(
            strategy_name=strategy_name,
            enabled=False,
            action_time=datetime.utcnow(),
            message=f"Strategy {strategy_name} disabled successfully"
        )
        
    except StrategyError as e:
        capture_trading_event(
            f"Failed to disable strategy {strategy_name}: {str(e)}",
            level="warning",
            extra={"strategy_name": strategy_name, "error": str(e)},
            tags={"operation": "disable_strategy", "strategy": strategy_name, "error_type": "strategy_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_name} not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error disabling strategy {strategy_name}: {str(e)}")
        capture_trading_event(
            f"Failed to disable strategy {strategy_name}: {str(e)}",
            level="error",
            extra={"strategy_name": strategy_name, "error": str(e)},
            tags={"operation": "disable_strategy", "strategy": strategy_name, "error_type": "internal_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable strategy"
        )


@router.put("/{strategy_name}/parameters", response_model=StrategyActionResponse)
@monitor_api_endpoint("update_strategy_parameters")
async def update_strategy_parameters(strategy_name: str, parameters_request: StrategyParametersRequest):
    """Update parameters for a specific strategy."""
    try:
        strategy_manager = StrategyFactory.create_strategy_manager(
            strategy_engine=StrategyFactory.create_strategy_engine(),
            market_data_manager=None,
            order_manager=None,
            risk_manager=None
        )
        
        strategy = strategy_manager.strategy_engine.get_strategy(strategy_name)
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy {strategy_name} not found"
            )
        
        strategy.update_parameters(parameters_request.parameters)
        
        capture_trading_event(
            f"Strategy parameters updated: {strategy_name}",
            level="info",
            extra={"strategy_name": strategy_name, "parameters": parameters_request.parameters},
            tags={"operation": "update_strategy_parameters", "strategy": strategy_name}
        )
        
        return StrategyActionResponse(
            strategy_name=strategy_name,
            enabled=strategy.enabled,
            action_time=datetime.utcnow(),
            message=f"Strategy {strategy_name} parameters updated successfully"
        )
        
    except StrategyError as e:
        capture_trading_event(
            f"Failed to update strategy {strategy_name} parameters: {str(e)}",
            level="warning",
            extra={"strategy_name": strategy_name, "error": str(e)},
            tags={"operation": "update_strategy_parameters", "strategy": strategy_name, "error_type": "strategy_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update strategy parameters: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_name} parameters: {str(e)}")
        capture_trading_event(
            f"Failed to update strategy {strategy_name} parameters: {str(e)}",
            level="error",
            extra={"strategy_name": strategy_name, "error": str(e)},
            tags={"operation": "update_strategy_parameters", "strategy": strategy_name, "error_type": "internal_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update strategy parameters"
        )