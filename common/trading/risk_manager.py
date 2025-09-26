"""
Risk Management Framework for trading operations.
Provides comprehensive risk checks and portfolio monitoring.
"""

import logging
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from common.schemas.trading import Order, Position, EquityContract, OptionContract, OrderAction
from common.repositories.trading.positions import PositionRepository


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RiskViolation:
    """Risk violation information."""
    check_type: str
    severity: str  # "warning", "error"
    message: str
    current_value: float
    limit_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type,
            "severity": self.severity,
            "message": self.message,
            "current_value": self.current_value,
            "limit_value": self.limit_value
        }


@dataclass
class RiskCheckResult:
    """Risk check result."""
    approved: bool
    risk_level: RiskLevel
    violations: List[RiskViolation]
    risk_score: float = 0.0
    override_user: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approved": self.approved,
            "risk_level": self.risk_level.value,
            "violations": [v.to_dict() for v in self.violations],
            "risk_score": self.risk_score,
            "override_user": self.override_user
        }


class RiskManagerError(Exception):
    """Risk manager related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class RiskManager:
    """Comprehensive risk management for trading operations."""
    
    def __init__(
        self, 
        position_repository: PositionRepository,
        risk_config: Dict[str, Any]
    ):
        """Initialize risk manager."""
        self.position_repository = position_repository
        self.risk_config = risk_config
        self.enabled_checks = set(risk_config.get("enabled_checks", []))
        self.logger = logger
    
    def check_order_risk(
        self, 
        order: Order, 
        account: str, 
        estimated_price: Decimal,
        override_user: Optional[str] = None
    ) -> RiskCheckResult:
        """Comprehensive order risk check."""
        if order is None:
            raise ValueError("Order cannot be None")
        
        if not account or account.strip() == "":
            raise ValueError("Account cannot be empty")
        
        if estimated_price <= 0:
            raise ValueError("Estimated price must be positive")
        
        try:
            violations = []
            
            # Position size checks
            if "position_size" in self.enabled_checks:
                violations.extend(self._validate_order_size(order, account))
            
            # Concentration checks
            if "concentration" in self.enabled_checks:
                violations.extend(self._validate_concentration_limits(order, account, estimated_price))
            
            # Buying power checks
            if "buying_power" in self.enabled_checks:
                violations.extend(self._validate_buying_power(order, account, estimated_price))
            
            # Portfolio value checks
            if "portfolio_value" in self.enabled_checks:
                violations.extend(self._validate_portfolio_limits(order, account, estimated_price))
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(violations)
            
            # Determine approval
            approved = len([v for v in violations if v.severity == "error"]) == 0
            
            # Handle overrides
            if not approved and override_user:
                # Log override but allow for authorized users
                self.logger.warning(f"Risk override applied by {override_user} for order {order.client_order_id}")
            
            return RiskCheckResult(
                approved=approved,
                risk_level=risk_level,
                violations=violations,
                risk_score=self._calculate_risk_score(violations),
                override_user=override_user
            )
            
        except Exception as e:
            self.logger.error(f"Error checking order risk: {e}")
            raise RiskManagerError(f"Failed to check order risk: {e}")
    
    def check_portfolio_risk(self, account: str) -> RiskCheckResult:
        """Check overall portfolio risk."""
        try:
            violations = []
            
            # Get portfolio summary
            portfolio_summary = self.position_repository.get_portfolio_summary(account)
            
            # Daily loss checks
            total_pnl = portfolio_summary.get("total_pnl", 0.0)
            max_daily_loss = self.risk_config.get("max_daily_loss", 0.0)
            
            if abs(total_pnl) > max_daily_loss and total_pnl < 0:
                violations.append(RiskViolation(
                    check_type="daily_loss",
                    severity="error",
                    message=f"Daily loss limit exceeded: ${abs(total_pnl):,.2f} > ${max_daily_loss:,.2f}",
                    current_value=abs(total_pnl),
                    limit_value=max_daily_loss
                ))
            
            # Portfolio value checks
            total_value = portfolio_summary.get("total_market_value", 0.0)
            max_portfolio_value = self.risk_config.get("max_portfolio_value", 0.0)
            
            if total_value > max_portfolio_value:
                violations.append(RiskViolation(
                    check_type="portfolio_value",
                    severity="warning",
                    message=f"Portfolio value exceeds limit: ${total_value:,.2f} > ${max_portfolio_value:,.2f}",
                    current_value=total_value,
                    limit_value=max_portfolio_value
                ))
            
            risk_level = self._calculate_risk_level(violations)
            approved = len([v for v in violations if v.severity == "error"]) == 0
            
            return RiskCheckResult(
                approved=approved,
                risk_level=risk_level,
                violations=violations,
                risk_score=self._calculate_risk_score(violations)
            )
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio risk for {account}: {e}")
            raise RiskManagerError(f"Failed to check portfolio risk: {e}")
    
    def get_position_risk_metrics(self, account: str) -> Dict[str, Any]:
        """Calculate position risk metrics."""
        try:
            positions = self.position_repository.get_positions_by_account(account)
            
            if not positions:
                return {
                    "total_exposure": 0.0,
                    "largest_position": None,
                    "concentration_by_symbol": {},
                    "position_count": 0
                }
            
            # Calculate total exposure
            total_exposure = sum(abs(float(p.market_value)) for p in positions)
            
            # Find largest position
            largest_position = max(positions, key=lambda p: abs(float(p.market_value)))
            largest_percentage = (abs(float(largest_position.market_value)) / total_exposure * 100) if total_exposure > 0 else 0
            
            # Calculate concentration by symbol
            concentration_by_symbol = {}
            for position in positions:
                symbol = position.contract.symbol
                value = abs(float(position.market_value))
                percentage = (value / total_exposure * 100) if total_exposure > 0 else 0
                
                concentration_by_symbol[symbol] = {
                    "value": value,
                    "percentage": percentage,
                    "position_size": position.position_size
                }
            
            return {
                "total_exposure": total_exposure,
                "largest_position": {
                    "symbol": largest_position.contract.symbol,
                    "value": abs(float(largest_position.market_value)),
                    "percentage": largest_percentage
                },
                "concentration_by_symbol": concentration_by_symbol,
                "position_count": len(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics for {account}: {e}")
            raise RiskManagerError(f"Failed to calculate risk metrics: {e}")
    
    def monitor_portfolio_risk(self, account: str) -> List[RiskViolation]:
        """Real-time portfolio risk monitoring."""
        try:
            portfolio_result = self.check_portfolio_risk(account)
            
            # Return only violations that require immediate attention
            alerts = [v for v in portfolio_result.violations if v.severity == "error"]
            
            # Log warnings
            for violation in portfolio_result.violations:
                if violation.severity == "warning":
                    self.logger.warning(f"Portfolio risk warning for {account}: {violation.message}")
                elif violation.severity == "error":
                    self.logger.error(f"Portfolio risk alert for {account}: {violation.message}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error monitoring portfolio risk for {account}: {e}")
            return []
    
    def update_risk_config(self, new_config: Dict[str, Any]) -> None:
        """Update risk configuration."""
        try:
            self.risk_config.update(new_config)
            
            # Update enabled checks
            if "enabled_checks" in new_config:
                self.enabled_checks = set(new_config["enabled_checks"])
            
            self.logger.info(f"Risk configuration updated: {new_config}")
            
        except Exception as e:
            self.logger.error(f"Error updating risk config: {e}")
            raise RiskManagerError(f"Failed to update risk config: {e}")
    
    def _validate_order_size(self, order: Order, account: str) -> List[RiskViolation]:
        """Validate order size against limits."""
        violations = []
        
        max_position_size = self.risk_config.get("max_position_size", 0)
        
        if order.quantity > max_position_size:
            violations.append(RiskViolation(
                check_type="position_size",
                severity="error",
                message=f"Order size exceeds limit: {order.quantity} > {max_position_size}",
                current_value=order.quantity,
                limit_value=max_position_size
            ))
        elif order.quantity > max_position_size * 0.8:  # Warning at 80%
            violations.append(RiskViolation(
                check_type="position_size",
                severity="warning",
                message=f"Order size approaching limit: {order.quantity} (limit: {max_position_size})",
                current_value=order.quantity,
                limit_value=max_position_size
            ))
        
        return violations
    
    def _validate_concentration_limits(
        self, 
        order: Order, 
        account: str, 
        estimated_price: Decimal
    ) -> List[RiskViolation]:
        """Validate concentration limits."""
        violations = []
        
        try:
            # Get current positions
            positions = self.position_repository.get_positions_by_account(account)
            portfolio_summary = self.position_repository.get_portfolio_summary(account)
            
            current_portfolio_value = portfolio_summary.get("total_market_value", 0.0)
            
            # Calculate order value
            if isinstance(order.contract, OptionContract):
                multiplier = int(getattr(order.contract, 'multiplier', 100))
                order_value = float(estimated_price) * order.quantity * multiplier
            else:
                order_value = float(estimated_price) * order.quantity
            
            # Find existing position for this symbol
            existing_position_value = 0.0
            for position in positions:
                if position.contract.symbol == order.contract.symbol:
                    existing_position_value += abs(float(position.market_value))
            
            # Calculate new total value for this symbol
            if order.action == OrderAction.BUY:
                new_symbol_value = existing_position_value + order_value
            else:  # SELL
                new_symbol_value = max(0, existing_position_value - order_value)
            
            # Calculate new portfolio value (only add for BUY orders)
            if order.action == OrderAction.BUY:
                new_portfolio_value = current_portfolio_value + order_value
            else:
                new_portfolio_value = current_portfolio_value
            
            # Check concentration
            max_concentration = self.risk_config.get("max_concentration", 1.0)
            
            if new_portfolio_value > 0:
                concentration_percentage = new_symbol_value / new_portfolio_value
                
                if concentration_percentage > max_concentration:
                    violations.append(RiskViolation(
                        check_type="concentration",
                        severity="error",
                        message=f"Symbol concentration too high: {concentration_percentage:.1%} > {max_concentration:.1%}",
                        current_value=concentration_percentage * 100,
                        limit_value=max_concentration * 100
                    ))
        
        except Exception as e:
            self.logger.warning(f"Error validating concentration limits: {e}")
        
        return violations
    
    def _validate_buying_power(
        self, 
        order: Order, 
        account: str, 
        estimated_price: Decimal
    ) -> List[RiskViolation]:
        """Validate buying power requirements."""
        violations = []
        
        try:
            buying_power = self.risk_config.get("buying_power", 0.0)
            
            # Calculate order cost
            if isinstance(order.contract, OptionContract):
                multiplier = int(getattr(order.contract, 'multiplier', 100))
                order_cost = float(estimated_price) * order.quantity * multiplier
            else:
                order_cost = float(estimated_price) * order.quantity
            
            # Only check buying power for buy orders
            if order.action == OrderAction.BUY:
                # Get current portfolio value
                portfolio_summary = self.position_repository.get_portfolio_summary(account)
                current_value = portfolio_summary.get("total_market_value", 0.0)
                
                # Check if order would exceed buying power
                if current_value + order_cost > buying_power:
                    violations.append(RiskViolation(
                        check_type="buying_power",
                        severity="error",
                        message=f"Insufficient buying power: ${current_value + order_cost:,.2f} > ${buying_power:,.2f}",
                        current_value=current_value + order_cost,
                        limit_value=buying_power
                    ))
        
        except Exception as e:
            self.logger.warning(f"Error validating buying power: {e}")
        
        return violations
    
    def _validate_portfolio_limits(
        self, 
        order: Order, 
        account: str, 
        estimated_price: Decimal
    ) -> List[RiskViolation]:
        """Validate portfolio value limits."""
        violations = []
        
        try:
            max_portfolio_value = self.risk_config.get("max_portfolio_value", 0.0)
            
            if max_portfolio_value <= 0:
                return violations
            
            # Calculate order value
            if isinstance(order.contract, OptionContract):
                multiplier = int(getattr(order.contract, 'multiplier', 100))
                order_value = float(estimated_price) * order.quantity * multiplier
            else:
                order_value = float(estimated_price) * order.quantity
            
            # Get current portfolio value
            portfolio_summary = self.position_repository.get_portfolio_summary(account)
            current_value = portfolio_summary.get("total_market_value", 0.0)
            
            # Check if order would exceed portfolio limit
            if order.action == OrderAction.BUY:
                new_portfolio_value = current_value + order_value
                
                if new_portfolio_value > max_portfolio_value:
                    violations.append(RiskViolation(
                        check_type="portfolio_value",
                        severity="warning",
                        message=f"Portfolio value would exceed limit: ${new_portfolio_value:,.2f} > ${max_portfolio_value:,.2f}",
                        current_value=new_portfolio_value,
                        limit_value=max_portfolio_value
                    ))
        
        except Exception as e:
            self.logger.warning(f"Error validating portfolio limits: {e}")
        
        return violations
    
    def _calculate_risk_level(self, violations: List[RiskViolation]) -> RiskLevel:
        """Calculate risk level based on violations."""
        if not violations:
            return RiskLevel.LOW
        
        # Check for any error-level violations
        error_violations = [v for v in violations if v.severity == "error"]
        if error_violations:
            return RiskLevel.HIGH
        
        # Check for multiple warnings
        warning_violations = [v for v in violations if v.severity == "warning"]
        if len(warning_violations) >= 2:
            return RiskLevel.HIGH
        elif len(warning_violations) == 1:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _calculate_risk_score(self, violations: List[RiskViolation]) -> float:
        """Calculate numerical risk score (0-100)."""
        if not violations:
            return 0.0
        
        score = 0.0
        
        for violation in violations:
            if violation.severity == "error":
                score += 50.0
            elif violation.severity == "warning":
                score += 25.0
        
        return min(score, 100.0)


class RiskManagerFactory:
    """Factory for creating risk managers with different configurations."""
    
    @staticmethod
    def create_conservative_risk_manager(position_repository: PositionRepository) -> RiskManager:
        """Create risk manager with conservative settings."""
        config = {
            "max_position_size": 500,
            "max_portfolio_value": 50000.0,
            "max_daily_loss": 2500.0,
            "max_concentration": 0.15,  # 15% max per symbol
            "max_leverage": 1.5,
            "buying_power": 25000.0,
            "enabled_checks": ["position_size", "portfolio_value", "concentration", "buying_power", "daily_loss"]
        }
        return RiskManager(position_repository, config)
    
    @staticmethod
    def create_aggressive_risk_manager(position_repository: PositionRepository) -> RiskManager:
        """Create risk manager with aggressive settings."""
        config = {
            "max_position_size": 2000,
            "max_portfolio_value": 200000.0,
            "max_daily_loss": 10000.0,
            "max_concentration": 0.30,  # 30% max per symbol
            "max_leverage": 3.0,
            "buying_power": 100000.0,
            "enabled_checks": ["position_size", "portfolio_value", "concentration", "buying_power"]
        }
        return RiskManager(position_repository, config)
    
    @staticmethod
    def create_custom_risk_manager(
        position_repository: PositionRepository, 
        config: Dict[str, Any]
    ) -> RiskManager:
        """Create risk manager with custom configuration."""
        return RiskManager(position_repository, config)