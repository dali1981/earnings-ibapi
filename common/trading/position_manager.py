"""
Position Manager for tracking and managing trading positions.
Handles position updates from executions and provides portfolio analytics.
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Union
from datetime import datetime

from common.schemas.trading import Position, ExecutionReport, EquityContract, OptionContract, OrderAction
from common.repositories.trading.positions import PositionRepository
from common.trading.events import PositionEvent, PositionEventType
from common.utils.events.bus import FileEventBus


logger = logging.getLogger(__name__)


class PositionManagerError(Exception):
    """Position manager related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class PositionManager:
    """Manager for trading positions with real-time updates and analytics."""
    
    def __init__(
        self, 
        position_repository: PositionRepository,
        event_bus: Optional[FileEventBus] = None
    ):
        """Initialize position manager."""
        self.position_repository = position_repository
        self.event_bus = event_bus or FileEventBus()
        self.logger = logger
    
    def process_execution(self, execution: ExecutionReport, account: str) -> Position:
        """Process execution report and update positions."""
        if execution is None:
            raise ValueError("Execution cannot be None")
        
        if not account or account.strip() == "":
            raise ValueError("Account cannot be empty")
        
        try:
            # Get existing position
            existing_position = self.position_repository.get_position_by_contract(
                execution.contract, account
            )
            
            if existing_position is None:
                # Create new position
                position = self._create_new_position(execution, account)
            else:
                # Update existing position
                position = self._update_existing_position(existing_position, execution, account)
            
            # Save updated position
            self.position_repository.domain_save(position)
            
            # Publish position update event
            self._publish_position_event(
                PositionEventType.POSITION_UPDATED,
                position,
                {"execution_id": execution.execution_id}
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error processing execution {execution.execution_id}: {e}")
            raise PositionManagerError(f"Failed to process execution: {e}")
    
    def update_market_prices(self, market_data: Dict[str, Decimal], account: str) -> None:
        """Update market prices for positions."""
        try:
            for symbol, price in market_data.items():
                # For simplicity, assume equity contracts for price updates
                # In practice, you'd need to handle both equities and options
                contract = EquityContract(symbol=symbol)
                self.position_repository.update_position_prices(contract, price, account)
                
        except Exception as e:
            self.logger.error(f"Error updating market prices: {e}")
            raise PositionManagerError(f"Failed to update market prices: {e}")
    
    def get_positions_by_account(self, account: str) -> List[Position]:
        """Get all positions for an account."""
        try:
            return self.position_repository.get_positions_by_account(account)
        except Exception as e:
            self.logger.error(f"Error retrieving positions for account {account}: {e}")
            raise PositionManagerError(f"Failed to retrieve positions: {e}")
    
    def get_position_by_contract(
        self, 
        contract: Union[EquityContract, OptionContract], 
        account: str
    ) -> Optional[Position]:
        """Get specific position by contract."""
        try:
            return self.position_repository.get_position_by_contract(contract, account)
        except Exception as e:
            self.logger.error(f"Error retrieving position for {contract.symbol}: {e}")
            raise PositionManagerError(f"Failed to retrieve position: {e}")
    
    def get_portfolio_summary(self, account: str) -> Dict[str, float]:
        """Get portfolio summary statistics."""
        try:
            return self.position_repository.get_portfolio_summary(account)
        except Exception as e:
            self.logger.error(f"Error calculating portfolio summary for {account}: {e}")
            raise PositionManagerError(f"Failed to calculate portfolio summary: {e}")
    
    def close_position(
        self, 
        contract: Union[EquityContract, OptionContract],
        realized_pnl: Decimal,
        account: str
    ) -> None:
        """Close a position completely."""
        try:
            self.position_repository.close_position(contract, realized_pnl, account)
            
            # Publish position closed event
            self._publish_position_event(
                PositionEventType.POSITION_CLOSED,
                None,  # No position object for closed position
                {
                    "symbol": contract.symbol,
                    "account": account,
                    "realized_pnl": float(realized_pnl)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error closing position for {contract.symbol}: {e}")
            raise PositionManagerError(f"Failed to close position: {e}")
    
    def _create_new_position(self, execution: ExecutionReport, account: str) -> Position:
        """Create new position from execution."""
        # Calculate position size based on action
        position_size = execution.quantity if execution.action == OrderAction.BUY else -execution.quantity
        
        # Calculate market value based on contract type
        if isinstance(execution.contract, OptionContract):
            multiplier = int(getattr(execution.contract, 'multiplier', 100))
            market_value = execution.price * position_size * multiplier
        else:
            market_value = execution.price * position_size
        
        # For new position, unrealized P&L starts at 0 (market_price = avg_cost)
        unrealized_pnl = Decimal("0")
        
        return Position(
            contract=execution.contract,
            position_size=position_size,
            avg_cost=execution.price,
            market_price=execution.price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            account=account
        )
    
    def _update_existing_position(
        self, 
        existing_position: Position, 
        execution: ExecutionReport, 
        account: str
    ) -> Position:
        """Update existing position with new execution."""
        if execution.action == OrderAction.BUY:
            return self._process_buy_execution(existing_position, execution)
        else:
            return self._process_sell_execution(existing_position, execution)
    
    def _process_buy_execution(self, position: Position, execution: ExecutionReport) -> Position:
        """Process buy execution - adds to position."""
        new_avg_cost = self._calculate_weighted_average_cost(
            position.avg_cost,
            abs(position.position_size),
            execution.price,
            execution.quantity
        )
        
        new_position_size = position.position_size + execution.quantity
        
        # Update market value based on contract type
        if isinstance(execution.contract, OptionContract):
            multiplier = int(getattr(execution.contract, 'multiplier', 100))
            market_value = position.market_price * new_position_size * multiplier
        else:
            market_value = position.market_price * new_position_size
        
        # Calculate unrealized P&L based on current market price
        if isinstance(execution.contract, OptionContract):
            multiplier = int(getattr(execution.contract, 'multiplier', 100))
            unrealized_pnl = (position.market_price - new_avg_cost) * new_position_size * multiplier
        else:
            unrealized_pnl = (position.market_price - new_avg_cost) * new_position_size
        
        return Position(
            contract=position.contract,
            position_size=new_position_size,
            avg_cost=new_avg_cost,
            market_price=position.market_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=position.realized_pnl,
            account=position.account
        )
    
    def _process_sell_execution(self, position: Position, execution: ExecutionReport) -> Position:
        """Process sell execution - reduces position and realizes P&L."""
        # Calculate realized P&L for the sold portion
        realized_pnl_from_sale = self._calculate_realized_pnl(
            position.avg_cost, execution.price, execution.quantity
        )
        
        # Update position
        new_position_size = position.position_size - execution.quantity
        total_realized_pnl = position.realized_pnl + realized_pnl_from_sale
        
        # Update market value
        if isinstance(execution.contract, OptionContract):
            multiplier = int(getattr(execution.contract, 'multiplier', 100))
            market_value = position.market_price * new_position_size * multiplier
        else:
            market_value = position.market_price * new_position_size
        
        # Calculate unrealized P&L for remaining position
        if new_position_size != 0:
            if isinstance(execution.contract, OptionContract):
                multiplier = int(getattr(execution.contract, 'multiplier', 100))
                unrealized_pnl = (position.market_price - position.avg_cost) * new_position_size * multiplier
            else:
                unrealized_pnl = (position.market_price - position.avg_cost) * new_position_size
        else:
            unrealized_pnl = Decimal("0")
        
        return Position(
            contract=position.contract,
            position_size=new_position_size,
            avg_cost=position.avg_cost,  # Cost basis doesn't change on sale
            market_price=position.market_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=total_realized_pnl,
            account=position.account
        )
    
    def _calculate_weighted_average_cost(
        self, 
        existing_cost: Decimal, 
        existing_quantity: int,
        new_cost: Decimal, 
        new_quantity: int
    ) -> Decimal:
        """Calculate weighted average cost for position updates."""
        total_cost = (existing_cost * existing_quantity) + (new_cost * new_quantity)
        total_quantity = existing_quantity + new_quantity
        
        if total_quantity == 0:
            return Decimal("0")
        
        return round(total_cost / total_quantity, 2)
    
    def _calculate_realized_pnl(
        self, 
        avg_cost: Decimal, 
        sell_price: Decimal, 
        quantity: int
    ) -> Decimal:
        """Calculate realized P&L for a sale."""
        return (sell_price - avg_cost) * quantity
    
    def _publish_position_event(
        self, 
        event_type: PositionEventType,
        position: Optional[Position],
        additional_data: Dict = None
    ) -> None:
        """Publish position-related events."""
        try:
            event_data = additional_data or {}
            
            if position:
                event_data.update({
                    "symbol": position.contract.symbol,
                    "account": position.account,
                    "position_size": position.position_size,
                    "market_value": float(position.market_value),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "realized_pnl": float(position.realized_pnl)
                })
            
            event = PositionEvent(
                event_type=event_type,
                timestamp=datetime.utcnow(),
                data=event_data
            )
            
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.warning(f"Failed to publish position event: {e}")


# Factory function for creating position manager
class PositionManagerFactory:
    """Factory for creating position manager components."""
    
    @staticmethod
    def create_position_manager() -> PositionManager:
        """Create a position manager with default dependencies."""
        from common.repositories.trading.positions import PositionRepository
        
        position_repository = PositionRepository()
        event_bus = FileEventBus()
        
        return PositionManager(
            position_repository=position_repository,
            event_bus=event_bus
        )