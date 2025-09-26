"""
Unit tests for strategy engine framework.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional

# These imports will fail initially (Red phase)
from common.trading.strategy_engine import StrategyEngine, BaseStrategy, StrategyError, StrategyManager
from common.schemas.trading import TradeSignal, Position, RiskAssessment, PositionTarget
from common.trading.market_data import MarketData


class MockEarningsStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self):
        super().__init__("earnings_straddle", {"days_to_earnings": 7})
    
    def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        return [
            TradeSignal(
                signal_id="test-signal-1",
                strategy_name=self.strategy_name,
                symbol=market_data.symbol,
                signal_type="ENTRY",
                confidence=0.8,
                target_positions=[],
                risk_metrics=RiskAssessment(max_loss=Decimal("100.00")),
                generated_at=datetime.utcnow()
            )
        ]


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.strategy
class TestBaseStrategy:
    """Test BaseStrategy abstract class."""
    
    def test_base_strategy_initialization(self):
        """Test BaseStrategy initialization with name and parameters."""
        strategy = MockEarningsStrategy()
        
        assert strategy.strategy_name == "earnings_straddle"
        assert strategy.parameters == {"days_to_earnings": 7}
        assert strategy.enabled is True
        assert isinstance(strategy.created_at, datetime)
    
    def test_base_strategy_validation(self):
        """Test strategy parameter validation."""
        strategy = MockEarningsStrategy()
        
        # Should implement validation in concrete classes
        result = strategy.validate_parameters()
        assert isinstance(result, bool)
    
    def test_base_strategy_string_representation(self):
        """Test string representation of strategy."""
        strategy = MockEarningsStrategy()
        str_repr = str(strategy)
        
        assert "earnings_straddle" in str_repr
        assert strategy.strategy_name in str_repr


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.strategy
class TestStrategyEngine:
    """Test StrategyEngine functionality."""
    
    @pytest.fixture
    def strategy_engine(self):
        """Create strategy engine for testing."""
        return StrategyEngine()
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        return MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02"),
            volume=1000,
            timestamp=datetime.utcnow()
        )
    
    def test_strategy_engine_initialization(self, strategy_engine):
        """Test StrategyEngine initialization."""
        assert isinstance(strategy_engine._strategies, dict)
        assert len(strategy_engine._strategies) == 0
        assert strategy_engine.enabled is True
    
    def test_register_strategy_adds_to_available_strategies(self, strategy_engine):
        """Test strategy registration."""
        strategy = MockEarningsStrategy()
        
        strategy_engine.register_strategy(strategy)
        
        assert "earnings_straddle" in strategy_engine._strategies
        assert strategy_engine._strategies["earnings_straddle"] == strategy
    
    def test_register_strategy_prevents_duplicates(self, strategy_engine):
        """Test that duplicate strategy names are prevented."""
        strategy1 = MockEarningsStrategy()
        strategy2 = MockEarningsStrategy()
        
        strategy_engine.register_strategy(strategy1)
        
        with pytest.raises(StrategyError, match="Strategy .* already registered"):
            strategy_engine.register_strategy(strategy2)
    
    def test_unregister_strategy_removes_from_available(self, strategy_engine):
        """Test strategy unregistration."""
        strategy = MockEarningsStrategy()
        strategy_engine.register_strategy(strategy)
        
        strategy_engine.unregister_strategy("earnings_straddle")
        
        assert "earnings_straddle" not in strategy_engine._strategies
    
    def test_unregister_nonexistent_strategy_raises_error(self, strategy_engine):
        """Test unregistering non-existent strategy raises error."""
        with pytest.raises(StrategyError, match="Strategy .* not found"):
            strategy_engine.unregister_strategy("nonexistent")
    
    def test_generate_signals_calls_all_strategies(self, strategy_engine, mock_market_data):
        """Test signal generation from registered strategies."""
        strategy1 = Mock(spec=BaseStrategy)
        strategy1.strategy_name = "strategy1"
        strategy1.enabled = True
        strategy1.generate_signals.return_value = []
        
        strategy2 = Mock(spec=BaseStrategy) 
        strategy2.strategy_name = "strategy2"
        strategy2.enabled = True
        strategy2.generate_signals.return_value = []
        
        strategy_engine.register_strategy(strategy1)
        strategy_engine.register_strategy(strategy2)
        
        signals = strategy_engine.generate_signals(mock_market_data)
        
        strategy1.generate_signals.assert_called_once_with(mock_market_data)
        strategy2.generate_signals.assert_called_once_with(mock_market_data)
        assert signals == []
    
    def test_generate_signals_skips_disabled_strategies(self, strategy_engine, mock_market_data):
        """Test that disabled strategies are skipped."""
        enabled_strategy = Mock(spec=BaseStrategy)
        enabled_strategy.strategy_name = "enabled"
        enabled_strategy.enabled = True
        enabled_strategy.generate_signals.return_value = []
        
        disabled_strategy = Mock(spec=BaseStrategy)
        disabled_strategy.strategy_name = "disabled"
        disabled_strategy.enabled = False
        disabled_strategy.generate_signals.return_value = []
        
        strategy_engine.register_strategy(enabled_strategy)
        strategy_engine.register_strategy(disabled_strategy)
        
        strategy_engine.generate_signals(mock_market_data)
        
        enabled_strategy.generate_signals.assert_called_once()
        disabled_strategy.generate_signals.assert_not_called()
    
    def test_generate_signals_aggregates_results(self, strategy_engine, mock_market_data):
        """Test that signals from multiple strategies are aggregated."""
        signal1 = TradeSignal(
            signal_id="signal-1",
            strategy_name="strategy1",
            symbol="AAPL",
            signal_type="ENTRY",
            confidence=0.8,
            target_positions=[],
            risk_metrics=RiskAssessment(max_loss=Decimal("100.00")),
            generated_at=datetime.utcnow()
        )
        
        signal2 = TradeSignal(
            signal_id="signal-2", 
            strategy_name="strategy2",
            symbol="AAPL",
            signal_type="EXIT",
            confidence=0.7,
            target_positions=[],
            risk_metrics=RiskAssessment(max_loss=Decimal("50.00")),
            generated_at=datetime.utcnow()
        )
        
        strategy1 = Mock(spec=BaseStrategy)
        strategy1.strategy_name = "strategy1"
        strategy1.enabled = True
        strategy1.generate_signals.return_value = [signal1]
        
        strategy2 = Mock(spec=BaseStrategy)
        strategy2.strategy_name = "strategy2"
        strategy2.enabled = True
        strategy2.generate_signals.return_value = [signal2]
        
        strategy_engine.register_strategy(strategy1)
        strategy_engine.register_strategy(strategy2)
        
        signals = strategy_engine.generate_signals(mock_market_data)
        
        assert len(signals) == 2
        assert signal1 in signals
        assert signal2 in signals
    
    def test_get_strategy_returns_registered_strategy(self, strategy_engine):
        """Test retrieving a registered strategy."""
        strategy = MockEarningsStrategy()
        strategy_engine.register_strategy(strategy)
        
        retrieved = strategy_engine.get_strategy("earnings_straddle")
        
        assert retrieved == strategy
    
    def test_get_strategy_returns_none_for_nonexistent(self, strategy_engine):
        """Test retrieving non-existent strategy returns None."""
        result = strategy_engine.get_strategy("nonexistent")
        assert result is None
    
    def test_list_strategies_returns_all_registered(self, strategy_engine):
        """Test listing all registered strategies."""
        strategy1 = MockEarningsStrategy()
        strategy2 = Mock(spec=BaseStrategy)
        strategy2.strategy_name = "test_strategy"
        
        strategy_engine.register_strategy(strategy1)
        strategy_engine.register_strategy(strategy2)
        
        strategies = strategy_engine.list_strategies()
        
        assert len(strategies) == 2
        assert "earnings_straddle" in strategies
        assert "test_strategy" in strategies
    
    def test_enable_disable_strategy(self, strategy_engine):
        """Test enabling and disabling strategies."""
        strategy = MockEarningsStrategy()
        strategy_engine.register_strategy(strategy)
        
        # Disable strategy
        strategy_engine.disable_strategy("earnings_straddle")
        assert not strategy_engine._strategies["earnings_straddle"].enabled
        
        # Enable strategy
        strategy_engine.enable_strategy("earnings_straddle")
        assert strategy_engine._strategies["earnings_straddle"].enabled
    
    def test_error_handling_invalid_market_data(self, strategy_engine):
        """Test error handling for invalid market data."""
        strategy = MockEarningsStrategy()
        strategy_engine.register_strategy(strategy)
        
        with pytest.raises(ValueError, match="Market data cannot be None"):
            strategy_engine.generate_signals(None)


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.strategy
class TestStrategyManager:
    """Test StrategyManager high-level functionality."""
    
    @pytest.fixture
    def mock_strategy_engine(self):
        """Create mock strategy engine."""
        return Mock(spec=StrategyEngine)
    
    @pytest.fixture
    def mock_market_data_manager(self):
        """Create mock market data manager."""
        return Mock()
    
    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager."""
        return Mock()
    
    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock()
    
    @pytest.fixture
    def strategy_manager(self, mock_strategy_engine, mock_market_data_manager, mock_order_manager, mock_risk_manager):
        """Create strategy manager for testing."""
        return StrategyManager(
            strategy_engine=mock_strategy_engine,
            market_data_manager=mock_market_data_manager,
            order_manager=mock_order_manager,
            risk_manager=mock_risk_manager
        )
    
    def test_strategy_manager_initialization(self, strategy_manager, mock_strategy_engine, mock_market_data_manager):
        """Test StrategyManager initialization."""
        assert strategy_manager.strategy_engine == mock_strategy_engine
        assert strategy_manager.market_data_manager == mock_market_data_manager
        assert strategy_manager.order_manager is not None
        assert strategy_manager.risk_manager is not None
    
    @pytest.mark.asyncio
    async def test_run_strategies_generates_and_executes_signals(self, strategy_manager, mock_strategy_engine, mock_market_data_manager):
        """Test running strategies with signal generation and execution."""
        # Mock market data
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                last=Decimal("150.00"),
                timestamp=datetime.utcnow()
            )
        }
        mock_market_data_manager.get_all_market_data.return_value = market_data
        
        # Mock signals
        signal = TradeSignal(
            signal_id="test-signal",
            strategy_name="test_strategy",
            symbol="AAPL",
            signal_type="ENTRY",
            confidence=0.8,
            target_positions=[],
            risk_metrics=RiskAssessment(max_loss=Decimal("100.00")),
            generated_at=datetime.utcnow()
        )
        mock_strategy_engine.generate_signals.return_value = [signal]
        
        # Run strategies
        results = await strategy_manager.run_strategies()
        
        # Verify calls
        mock_market_data_manager.get_all_market_data.assert_called_once()
        mock_strategy_engine.generate_signals.assert_called()
        
        assert len(results) > 0
    
    def test_add_strategy_delegates_to_engine(self, strategy_manager, mock_strategy_engine):
        """Test adding strategy delegates to engine."""
        strategy = MockEarningsStrategy()
        
        strategy_manager.add_strategy(strategy)
        
        mock_strategy_engine.register_strategy.assert_called_once_with(strategy)
    
    def test_remove_strategy_delegates_to_engine(self, strategy_manager, mock_strategy_engine):
        """Test removing strategy delegates to engine."""
        strategy_manager.remove_strategy("test_strategy")
        
        mock_strategy_engine.unregister_strategy.assert_called_once_with("test_strategy")
    
    def test_get_strategy_performance_returns_metrics(self, strategy_manager):
        """Test getting strategy performance metrics."""
        # This would return cached performance data
        performance = strategy_manager.get_strategy_performance("test_strategy")
        
        assert isinstance(performance, dict)
        # Performance metrics would include P&L, win rate, etc.
    
    def test_error_handling_no_market_data(self, strategy_manager, mock_market_data_manager):
        """Test error handling when no market data available."""
        mock_market_data_manager.get_all_market_data.return_value = {}
        
        # Should handle gracefully when no market data
        results = strategy_manager._get_current_market_data()
        assert len(results) == 0