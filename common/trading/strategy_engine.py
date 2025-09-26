"""
Strategy engine framework for trading strategies.
Provides base classes and execution framework for trading strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal

from common.schemas.trading import TradeSignal, Position
from common.trading.market_data import MarketData
from common.schemas.base import BaseAPIModel


class StrategyError(Exception):
    """Strategy-related errors."""
    pass


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, strategy_name: str, parameters: Dict[str, Any]):
        """Initialize base strategy."""
        self.strategy_name = strategy_name
        self.parameters = parameters
        self.enabled = True
        self.created_at = datetime.utcnow()
        self.last_signal_time: Optional[datetime] = None
        self.signal_count = 0
    
    @abstractmethod
    def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate trading signals based on market data."""
        pass
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        # Default implementation - should be overridden by concrete strategies
        return True
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        self.parameters.update(parameters)
        # Re-validate after update
        if not self.validate_parameters():
            raise StrategyError(f"Invalid parameters for strategy {self.strategy_name}")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a strategy parameter."""
        return self.parameters.get(key, default)
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"Strategy({self.strategy_name}, enabled={self.enabled})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"BaseStrategy(name={self.strategy_name}, "
                f"enabled={self.enabled}, "
                f"parameters={self.parameters})")


class StrategyEngine:
    """Engine for managing and executing trading strategies."""
    
    def __init__(self):
        """Initialize strategy engine."""
        self._strategies: Dict[str, BaseStrategy] = {}
        self.enabled = True
        self.created_at = datetime.utcnow()
        self.total_signals_generated = 0
    
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy with the engine."""
        if strategy.strategy_name in self._strategies:
            raise StrategyError(f"Strategy {strategy.strategy_name} already registered")
        
        if not isinstance(strategy, BaseStrategy):
            raise StrategyError("Strategy must inherit from BaseStrategy")
        
        self._strategies[strategy.strategy_name] = strategy
    
    def unregister_strategy(self, strategy_name: str) -> None:
        """Unregister a strategy from the engine."""
        if strategy_name not in self._strategies:
            raise StrategyError(f"Strategy {strategy_name} not found")
        
        del self._strategies[strategy_name]
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Get a registered strategy by name."""
        return self._strategies.get(strategy_name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    def enable_strategy(self, strategy_name: str) -> None:
        """Enable a specific strategy."""
        if strategy_name not in self._strategies:
            raise StrategyError(f"Strategy {strategy_name} not found")
        
        self._strategies[strategy_name].enable()
    
    def disable_strategy(self, strategy_name: str) -> None:
        """Disable a specific strategy."""
        if strategy_name not in self._strategies:
            raise StrategyError(f"Strategy {strategy_name} not found")
        
        self._strategies[strategy_name].disable()
    
    def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate signals from all enabled strategies."""
        if market_data is None:
            raise ValueError("Market data cannot be None")
        
        if not self.enabled:
            return []
        
        all_signals = []
        
        for strategy_name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            
            try:
                signals = strategy.generate_signals(market_data)
                if signals:
                    all_signals.extend(signals)
                    strategy.last_signal_time = datetime.utcnow()
                    strategy.signal_count += len(signals)
                    self.total_signals_generated += len(signals)
            
            except Exception as e:
                # Log error but continue with other strategies
                print(f"Error generating signals for strategy {strategy_name}: {e}")
                continue
        
        return all_signals
    
    def enable(self) -> None:
        """Enable the strategy engine."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the strategy engine."""
        self.enabled = False
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get strategy engine status."""
        enabled_strategies = [name for name, strategy in self._strategies.items() if strategy.enabled]
        disabled_strategies = [name for name, strategy in self._strategies.items() if not strategy.enabled]
        
        return {
            "enabled": self.enabled,
            "total_strategies": len(self._strategies),
            "enabled_strategies": enabled_strategies,
            "disabled_strategies": disabled_strategies,
            "total_signals_generated": self.total_signals_generated,
            "created_at": self.created_at
        }


class StrategyManager:
    """High-level strategy management and execution."""
    
    def __init__(self, strategy_engine: StrategyEngine, market_data_manager, order_manager, risk_manager):
        """Initialize strategy manager."""
        self.strategy_engine = strategy_engine
        self.market_data_manager = market_data_manager
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
    
    async def run_strategies(self) -> List[Dict[str, Any]]:
        """Run all enabled strategies and execute signals."""
        results = []
        
        # Get current market data
        market_data_dict = self._get_current_market_data()
        
        if not market_data_dict:
            return results
        
        # Generate signals for each symbol with market data
        for symbol, market_data in market_data_dict.items():
            try:
                signals = self.strategy_engine.generate_signals(market_data)
                
                for signal in signals:
                    # Process each signal
                    result = await self._process_signal(signal)
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing signals for {symbol}: {e}")
                continue
        
        return results
    
    async def _process_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """Process a trading signal."""
        result = {
            "signal_id": signal.signal_id,
            "strategy_name": signal.strategy_name,
            "symbol": signal.symbol,
            "signal_type": signal.signal_type,
            "confidence": signal.confidence,
            "processed_at": datetime.utcnow(),
            "executed": False,
            "risk_approved": False,
            "orders_created": []
        }
        
        try:
            # Risk check first
            if signal.target_positions:
                # For now, just validate we have target positions
                result["risk_approved"] = True
            
            # If risk approved, could execute orders here
            if result["risk_approved"]:
                # For now, just mark as executed
                result["executed"] = True
                
        except Exception as e:
            result["error"] = str(e)
        
        # Cache execution history
        self.execution_history.append(result)
        
        return result
    
    def _get_current_market_data(self) -> Dict[str, MarketData]:
        """Get current market data for strategy processing."""
        if not self.market_data_manager:
            return {}
        
        try:
            return self.market_data_manager.get_all_market_data()
        except Exception:
            return {}
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the engine."""
        self.strategy_engine.register_strategy(strategy)
    
    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a strategy from the engine."""
        self.strategy_engine.unregister_strategy(strategy_name)
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance metrics for a strategy."""
        # Return cached performance or calculate basic metrics
        if strategy_name in self.performance_cache:
            return self.performance_cache[strategy_name]
        
        # Basic performance metrics
        strategy_executions = [
            ex for ex in self.execution_history 
            if ex.get("strategy_name") == strategy_name
        ]
        
        performance = {
            "strategy_name": strategy_name,
            "total_signals": len(strategy_executions),
            "successful_executions": len([ex for ex in strategy_executions if ex.get("executed")]),
            "risk_approved_signals": len([ex for ex in strategy_executions if ex.get("risk_approved")]),
            "last_signal_time": max([ex.get("processed_at") for ex in strategy_executions], default=None),
            "success_rate": 0.0
        }
        
        if performance["total_signals"] > 0:
            performance["success_rate"] = performance["successful_executions"] / performance["total_signals"]
        
        # Cache performance
        self.performance_cache[strategy_name] = performance
        
        return performance
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return self.execution_history[-limit:]
    
    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        self.performance_cache.clear()
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get strategy manager status."""
        engine_status = self.strategy_engine.get_engine_status()
        
        return {
            "engine_status": engine_status,
            "total_executions": len(self.execution_history),
            "cached_performance_strategies": list(self.performance_cache.keys()),
            "market_data_available": len(self._get_current_market_data()) > 0
        }


# Factory function for creating strategy components
class StrategyFactory:
    """Factory for creating strategy components."""
    
    @staticmethod
    def create_strategy_engine() -> StrategyEngine:
        """Create a strategy engine."""
        return StrategyEngine()
    
    @staticmethod
    def create_strategy_manager(
        strategy_engine: StrategyEngine,
        market_data_manager,
        order_manager,
        risk_manager
    ) -> StrategyManager:
        """Create a strategy manager with dependencies."""
        return StrategyManager(
            strategy_engine=strategy_engine,
            market_data_manager=market_data_manager,
            order_manager=order_manager,
            risk_manager=risk_manager
        )