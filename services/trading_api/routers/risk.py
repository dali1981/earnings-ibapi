"""
Risk management API endpoints.
"""

from __future__ import annotations

from ..schemas.trading import (
    TradeRiskAssessmentRequest, RiskAssessmentResponse, RiskLimitsResponse
)
from common.trading import RiskManagerFactory, RiskManagerError
from common.observability import monitor_api_endpoint, capture_trading_event
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

import logging


router = APIRouter(prefix="/risk", tags=["risk"])
logger = logging.getLogger(__name__)


@router.post("/assess", response_model=RiskAssessmentResponse)
@monitor_api_endpoint("assess_trade_risk")
async def assess_trade_risk(risk_request: TradeRiskAssessmentRequest):
    """Assess risk for a proposed trade."""
    try:
        risk_manager = RiskManagerFactory.create_risk_manager()
        
        # Perform risk assessment
        assessment = risk_manager.assess_trade_risk(
            positions=risk_request.positions,
            portfolio_value=risk_request.portfolio_value,
            strategy_name=risk_request.strategy_name
        )
        
        capture_trading_event(
            f"Risk assessment completed: {'APPROVED' if assessment['approved'] else 'REJECTED'}",
            level="info",
            extra={
                "approved": assessment["approved"],
                "max_loss": float(assessment["max_loss"]),
                "portfolio_heat": assessment.get("portfolio_heat"),
                "strategy_name": risk_request.strategy_name
            },
            tags={"operation": "assess_trade_risk", "strategy": risk_request.strategy_name}
        )
        
        return RiskAssessmentResponse(
            approved=assessment["approved"],
            max_loss=assessment["max_loss"],
            win_probability=assessment.get("win_probability"),
            risk_reward_ratio=assessment.get("risk_reward_ratio"),
            portfolio_heat=assessment.get("portfolio_heat"),
            risk_message=assessment.get("message"),
            assessment_time=datetime.utcnow()
        )
        
    except RiskManagerError as e:
        capture_trading_event(
            f"Risk assessment failed: {str(e)}",
            level="warning",
            extra={"error": str(e), "strategy_name": risk_request.strategy_name},
            tags={"operation": "assess_trade_risk", "error_type": "risk_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Risk assessment failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error assessing trade risk: {str(e)}")
        capture_trading_event(
            f"Risk assessment error: {str(e)}",
            level="error",
            extra={"error": str(e), "strategy_name": risk_request.strategy_name},
            tags={"operation": "assess_trade_risk", "error_type": "internal_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assess trade risk"
        )


@router.get("/limits", response_model=RiskLimitsResponse)
@monitor_api_endpoint("get_risk_limits")
async def get_risk_limits():
    """Get current risk limits and usage."""
    try:
        risk_manager = RiskManagerFactory.create_risk_manager()
        
        limits = risk_manager.get_risk_limits()
        
        # Get current usage metrics if available
        current_metrics = {}
        try:
            current_metrics = risk_manager.get_current_risk_metrics()
        except Exception:
            # Current metrics may not be available, that's ok
            pass
        
        capture_trading_event(
            "Retrieved risk limits",
            level="info",
            extra={
                "max_position_size": limits["max_position_size"],
                "max_portfolio_heat": limits["max_portfolio_heat"],
                "current_portfolio_heat": current_metrics.get("portfolio_heat")
            },
            tags={"operation": "get_risk_limits"}
        )
        
        return RiskLimitsResponse(
            max_position_size=limits["max_position_size"],
            max_portfolio_heat=limits["max_portfolio_heat"],
            max_daily_loss=limits["max_daily_loss"],
            max_sector_concentration=limits["max_sector_concentration"],
            current_portfolio_heat=current_metrics.get("portfolio_heat"),
            current_daily_pnl=current_metrics.get("daily_pnl")
        )
        
    except Exception as e:
        logger.error(f"Error getting risk limits: {str(e)}")
        capture_trading_event(
            f"Failed to get risk limits: {str(e)}",
            level="error",
            extra={"error": str(e)},
            tags={"operation": "get_risk_limits", "error_type": "fetch_failed"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve risk limits"
        )


@router.put("/limits", response_model=RiskLimitsResponse)
@monitor_api_endpoint("update_risk_limits")
async def update_risk_limits(limits_request: dict):
    """Update risk limits (admin operation)."""
    try:
        risk_manager = RiskManagerFactory.create_risk_manager()
        
        # Update limits
        updated_limits = risk_manager.update_risk_limits(limits_request)
        
        capture_trading_event(
            "Risk limits updated",
            level="info",
            extra={"updated_limits": limits_request},
            tags={"operation": "update_risk_limits"}
        )
        
        return RiskLimitsResponse(
            max_position_size=updated_limits["max_position_size"],
            max_portfolio_heat=updated_limits["max_portfolio_heat"],
            max_daily_loss=updated_limits["max_daily_loss"],
            max_sector_concentration=updated_limits["max_sector_concentration"]
        )
        
    except RiskManagerError as e:
        capture_trading_event(
            f"Failed to update risk limits: {str(e)}",
            level="warning",
            extra={"error": str(e), "limits_request": limits_request},
            tags={"operation": "update_risk_limits", "error_type": "validation_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update risk limits: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error updating risk limits: {str(e)}")
        capture_trading_event(
            f"Risk limits update error: {str(e)}",
            level="error",
            extra={"error": str(e), "limits_request": limits_request},
            tags={"operation": "update_risk_limits", "error_type": "internal_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update risk limits"
        )