"""Tests for query optimization utilities."""

import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date

from common.repositories.query_optimizer import (
    QueryOptimizer,
    QueryPlan,
    FilterCondition,
    OptimizationStrategy,
    create_filter_condition,
    optimize_query_filters,
    estimate_query_cost
)


class TestFilterCondition:
    """Test filter condition dataclass."""
    
    def test_filter_condition_creation(self):
        """Test FilterCondition creation."""
        condition = FilterCondition(
            column="symbol",
            operator="=",
            value="AAPL",
            selectivity=0.01
        )
        
        assert condition.column == "symbol"
        assert condition.operator == "="
        assert condition.value == "AAPL"
        assert condition.selectivity == 0.01
    
    def test_filter_condition_defaults(self):
        """Test FilterCondition default values."""
        condition = FilterCondition(
            column="date",
            operator=">=",
            value="2024-01-01"
        )
        
        assert condition.selectivity == 0.5  # Default selectivity
    
    def test_to_pyarrow_filter(self):
        """Test conversion to PyArrow filter expression."""
        condition = FilterCondition(
            column="price",
            operator=">",
            value=100.0
        )
        
        # Mock PyArrow compute
        with patch('common.repositories.query_optimizer.pc') as mock_pc:
            mock_pc.greater.return_value = "mock_filter"
            
            result = condition.to_pyarrow_filter()
            
            mock_pc.field.assert_called_with("price")
            mock_pc.greater.assert_called_once()
            assert result == "mock_filter"
    
    def test_to_pandas_filter(self):
        """Test conversion to pandas filter expression."""
        condition = FilterCondition(
            column="volume",
            operator=">=",
            value=1000
        )
        
        # Mock DataFrame
        mock_df = Mock()
        mock_series = Mock()
        mock_df.__getitem__.return_value = mock_series
        mock_series.__ge__.return_value = "pandas_filter"
        
        result = condition.to_pandas_filter(mock_df)
        
        mock_df.__getitem__.assert_called_with("volume")
        mock_series.__ge__.assert_called_with(1000)
        assert result == "pandas_filter"


class TestQueryPlan:
    """Test query plan dataclass."""
    
    def test_query_plan_creation(self):
        """Test QueryPlan creation."""
        filters = [
            FilterCondition("symbol", "=", "AAPL", 0.01),
            FilterCondition("date", ">=", "2024-01-01", 0.3)
        ]
        
        plan = QueryPlan(
            filters=filters,
            columns=["symbol", "date", "price"],
            limit=1000,
            estimated_cost=0.003,
            strategy=OptimizationStrategy.FILTER_PUSHDOWN,
            metadata={"table": "equity_bars"}
        )
        
        assert len(plan.filters) == 2
        assert plan.columns == ["symbol", "date", "price"]
        assert plan.limit == 1000
        assert plan.estimated_cost == 0.003
        assert plan.strategy == OptimizationStrategy.FILTER_PUSHDOWN
        assert plan.metadata["table"] == "equity_bars"


class TestCreateFilterCondition:
    """Test filter condition creation utility."""
    
    def test_create_simple_filter(self):
        """Test creating simple filter condition."""
        condition = create_filter_condition("symbol", "=", "AAPL")
        
        assert condition.column == "symbol"
        assert condition.operator == "="
        assert condition.value == "AAPL"
        assert condition.selectivity == 0.5  # Default
    
    def test_create_date_filter(self):
        """Test creating date filter with estimated selectivity."""
        condition = create_filter_condition(
            "date", ">=", "2024-01-01",
            total_range=("2023-01-01", "2024-12-31")
        )
        
        assert condition.column == "date"
        assert condition.operator == ">="
        assert condition.value == "2024-01-01"
        assert 0 < condition.selectivity < 1  # Should estimate based on range
    
    def test_create_numeric_filter(self):
        """Test creating numeric filter with estimated selectivity."""
        condition = create_filter_condition(
            "price", ">", 150.0,
            total_range=(100.0, 200.0)
        )
        
        assert condition.column == "price"
        assert condition.operator == ">"
        assert condition.value == 150.0
        assert 0 < condition.selectivity < 1  # Should estimate based on range


class TestOptimizeQueryFilters:
    """Test query filter optimization utility."""
    
    def test_optimize_filters_by_selectivity(self):
        """Test filter optimization by selectivity."""
        filters = [
            FilterCondition("symbol", "=", "AAPL", 0.01),      # Most selective
            FilterCondition("date", ">=", "2024-01-01", 0.3),  # Medium
            FilterCondition("price", ">", 100.0, 0.8)          # Least selective
        ]
        
        optimized = optimize_query_filters(filters)
        
        # Should be ordered by selectivity (most selective first)
        assert optimized[0].selectivity == 0.01
        assert optimized[1].selectivity == 0.3
        assert optimized[2].selectivity == 0.8
    
    def test_optimize_empty_filters(self):
        """Test optimization with empty filter list."""
        optimized = optimize_query_filters([])
        assert optimized == []


class TestEstimateQueryCost:
    """Test query cost estimation utility."""
    
    def test_estimate_cost_simple(self):
        """Test simple cost estimation."""
        filters = [FilterCondition("symbol", "=", "AAPL", 0.01)]
        
        cost = estimate_query_cost(
            filters=filters,
            total_rows=1000000,
            columns_requested=5,
            total_columns=10
        )
        
        # Cost should be based on selectivity and column ratio
        expected_cost = 1000000 * 0.01 * (5 / 10)
        assert cost == expected_cost
    
    def test_estimate_cost_multiple_filters(self):
        """Test cost estimation with multiple filters."""
        filters = [
            FilterCondition("symbol", "=", "AAPL", 0.01),
            FilterCondition("date", ">=", "2024-01-01", 0.3)
        ]
        
        cost = estimate_query_cost(
            filters=filters,
            total_rows=2000000,
            columns_requested=3,
            total_columns=8
        )
        
        # Combined selectivity: 0.01 * 0.3 = 0.003
        expected_cost = 2000000 * 0.003 * (3 / 8)
        assert cost == expected_cost
    
    def test_estimate_cost_no_filters(self):
        """Test cost estimation with no filters."""
        cost = estimate_query_cost(
            filters=[],
            total_rows=1000000,
            columns_requested=4,
            total_columns=6
        )
        
        # No filters means selectivity = 1.0
        expected_cost = 1000000 * 1.0 * (4 / 6)
        assert abs(cost - expected_cost) < 0.001


class TestQueryOptimizer:
    """Test query optimizer functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AAPL', 'GOOGL'] * 200,
            'date': pd.date_range('2024-01-01', periods=1000),
            'price': [150.0, 2800.0, 400.0, 151.0, 2850.0] * 200,
            'volume': [1000000, 500000, 800000, 1100000, 450000] * 200
        })
    
    @pytest.fixture
    def optimizer(self):
        """Create query optimizer instance."""
        return QueryOptimizer(enable_caching=True, cache_ttl=300)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.enable_caching is True
        assert optimizer.cache_ttl == 300
        assert optimizer._filter_cache == {}
        assert optimizer._stats == {
            'queries_optimized': 0,
            'cache_hits': 0,
            'average_optimization_time': 0.0
        }
    
    def test_create_query_plan(self, optimizer, sample_dataframe):
        """Test query plan creation."""
        filters = [
            ('symbol', '=', 'AAPL'),
            ('price', '>', 150.0),
            ('date', '>=', '2024-01-01')
        ]
        columns = ['symbol', 'price', 'volume']
        
        plan = optimizer.create_query_plan(
            filters=filters,
            columns=columns,
            limit=100,
            df=sample_dataframe
        )
        
        assert len(plan.filters) == 3
        assert plan.columns == columns
        assert plan.limit == 100
        assert plan.estimated_cost > 0
        assert plan.strategy in [OptimizationStrategy.FILTER_PUSHDOWN, 
                               OptimizationStrategy.COLUMN_PRUNING]
    
    def test_apply_query_plan_pandas(self, optimizer, sample_dataframe):
        """Test applying query plan to pandas DataFrame."""
        filters = [
            FilterCondition('symbol', '=', 'AAPL', 0.2),
            FilterCondition('price', '>', 150.0, 0.5)
        ]
        
        plan = QueryPlan(
            filters=filters,
            columns=['symbol', 'price'],
            limit=50,
            estimated_cost=100.0,
            strategy=OptimizationStrategy.FILTER_PUSHDOWN
        )
        
        result = optimizer.apply_query_plan(plan, sample_dataframe)
        
        # Result should be filtered
        assert len(result) <= 50
        assert list(result.columns) == ['symbol', 'price']
        assert all(result['symbol'] == 'AAPL')
        assert all(result['price'] > 150.0)
    
    @patch('common.repositories.query_optimizer.pq')
    def test_apply_query_plan_pyarrow(self, mock_pq, optimizer):
        """Test applying query plan to PyArrow table."""
        # Mock PyArrow table and operations
        mock_table = Mock()
        mock_filtered_table = Mock()
        mock_selected_table = Mock()
        mock_limited_table = Mock()
        
        mock_table.filter.return_value = mock_filtered_table
        mock_filtered_table.select.return_value = mock_selected_table
        mock_selected_table.slice.return_value = mock_limited_table
        mock_limited_table.to_pandas.return_value = pd.DataFrame({'result': [1, 2, 3]})
        
        filters = [FilterCondition('symbol', '=', 'AAPL', 0.2)]
        plan = QueryPlan(
            filters=filters,
            columns=['symbol', 'price'],
            limit=10,
            estimated_cost=50.0,
            strategy=OptimizationStrategy.FILTER_PUSHDOWN
        )
        
        result = optimizer.apply_query_plan(plan, mock_table)
        
        # Should apply operations in order: filter, select, slice
        mock_table.filter.assert_called_once()
        mock_filtered_table.select.assert_called_once_with(['symbol', 'price'])
        mock_selected_table.slice.assert_called_once_with(0, 10)
        assert isinstance(result, pd.DataFrame)
    
    def test_optimize_filters_with_dataframe_stats(self, optimizer, sample_dataframe):
        """Test filter optimization using DataFrame statistics."""
        filters = [
            ('symbol', '=', 'AAPL'),
            ('price', '>', 150.0),
            ('date', '>=', '2024-06-01')
        ]
        
        optimized_filters = optimizer._optimize_filters(filters, sample_dataframe)
        
        # Should return FilterCondition objects ordered by selectivity
        assert all(isinstance(f, FilterCondition) for f in optimized_filters)
        assert len(optimized_filters) == 3
        
        # Most selective filter should be first
        selectivities = [f.selectivity for f in optimized_filters]
        assert selectivities == sorted(selectivities)
    
    def test_estimate_selectivity_categorical(self, optimizer, sample_dataframe):
        """Test selectivity estimation for categorical columns."""
        selectivity = optimizer._estimate_selectivity(
            'symbol', '=', 'AAPL', sample_dataframe
        )
        
        # AAPL appears in 1/3 of the sample data (repeated pattern)
        expected_selectivity = len(sample_dataframe[sample_dataframe['symbol'] == 'AAPL']) / len(sample_dataframe)
        assert abs(selectivity - expected_selectivity) < 0.1
    
    def test_estimate_selectivity_numeric(self, optimizer, sample_dataframe):
        """Test selectivity estimation for numeric columns."""
        selectivity = optimizer._estimate_selectivity(
            'price', '>', 200.0, sample_dataframe
        )
        
        # Should estimate based on data distribution
        expected_selectivity = len(sample_dataframe[sample_dataframe['price'] > 200.0]) / len(sample_dataframe)
        assert abs(selectivity - expected_selectivity) < 0.1
    
    def test_choose_optimization_strategy(self, optimizer):
        """Test optimization strategy selection."""
        # Test filter pushdown strategy
        many_filters = [('col1', '=', 'val1'), ('col2', '>', 100), ('col3', '<', 200)]
        strategy1 = optimizer._choose_optimization_strategy(
            filters=many_filters,
            columns=['col1', 'col2'],
            total_columns=10,
            estimated_rows=1000000
        )
        assert strategy1 == OptimizationStrategy.FILTER_PUSHDOWN
        
        # Test column pruning strategy
        few_filters = [('col1', '=', 'val1')]
        strategy2 = optimizer._choose_optimization_strategy(
            filters=few_filters,
            columns=['col1', 'col2'],
            total_columns=20,
            estimated_rows=100000
        )
        assert strategy2 == OptimizationStrategy.COLUMN_PRUNING
    
    def test_caching_mechanism(self, optimizer, sample_dataframe):
        """Test query plan caching."""
        filters = [('symbol', '=', 'AAPL')]
        columns = ['symbol', 'price']
        
        # First call should create and cache plan
        plan1 = optimizer.create_query_plan(filters, columns, df=sample_dataframe)
        assert optimizer._stats['queries_optimized'] == 1
        assert optimizer._stats['cache_hits'] == 0
        
        # Second identical call should hit cache
        plan2 = optimizer.create_query_plan(filters, columns, df=sample_dataframe)
        assert optimizer._stats['cache_hits'] == 1
        
        # Plans should be identical
        assert plan1.filters[0].column == plan2.filters[0].column
        assert plan1.columns == plan2.columns
    
    def test_get_optimization_stats(self, optimizer, sample_dataframe):
        """Test optimization statistics."""
        # Perform some optimizations
        optimizer.create_query_plan([('symbol', '=', 'AAPL')], ['symbol'], df=sample_dataframe)
        optimizer.create_query_plan([('price', '>', 100)], ['price'], df=sample_dataframe)
        
        stats = optimizer.get_optimization_stats()
        
        assert stats['queries_optimized'] == 2
        assert stats['cache_hits'] == 0
        assert 'average_optimization_time' in stats
        assert 'cache_size' in stats
    
    def test_clear_cache(self, optimizer, sample_dataframe):
        """Test clearing optimization cache."""
        # Create some cached plans
        optimizer.create_query_plan([('symbol', '=', 'AAPL')], ['symbol'], df=sample_dataframe)
        assert len(optimizer._filter_cache) > 0
        
        # Clear cache
        optimizer.clear_cache()
        assert len(optimizer._filter_cache) == 0
    
    def test_complex_query_optimization(self, optimizer, sample_dataframe):
        """Test optimization of complex query with multiple conditions."""
        filters = [
            ('symbol', 'in', ['AAPL', 'GOOGL']),
            ('price', 'between', (100.0, 200.0)),
            ('volume', '>=', 500000),
            ('date', '>=', '2024-06-01')
        ]
        columns = ['symbol', 'date', 'price', 'volume']
        
        plan = optimizer.create_query_plan(
            filters=filters,
            columns=columns,
            limit=1000,
            df=sample_dataframe
        )
        
        assert len(plan.filters) == len(filters)
        assert plan.columns == columns
        assert plan.limit == 1000
        assert plan.estimated_cost > 0
        
        # Apply the plan
        result = optimizer.apply_query_plan(plan, sample_dataframe)
        
        # Verify results meet all conditions
        assert len(result) <= 1000
        assert all(col in result.columns for col in columns)


class TestOptimizationStrategies:
    """Test different optimization strategies."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for strategy tests."""
        return QueryOptimizer()
    
    def test_filter_pushdown_strategy(self, optimizer):
        """Test filter pushdown optimization strategy."""
        strategy = optimizer._choose_optimization_strategy(
            filters=[('col1', '=', 'val'), ('col2', '>', 100), ('col3', '<', 200)],
            columns=['col1', 'col2'],
            total_columns=8,
            estimated_rows=1000000
        )
        
        assert strategy == OptimizationStrategy.FILTER_PUSHDOWN
    
    def test_column_pruning_strategy(self, optimizer):
        """Test column pruning optimization strategy."""
        strategy = optimizer._choose_optimization_strategy(
            filters=[('col1', '=', 'val')],
            columns=['col1', 'col2'],
            total_columns=50,  # Many columns
            estimated_rows=100000
        )
        
        assert strategy == OptimizationStrategy.COLUMN_PRUNING
    
    def test_index_scan_strategy(self, optimizer):
        """Test index scan optimization strategy."""
        strategy = optimizer._choose_optimization_strategy(
            filters=[('indexed_col', '=', 'value')],  # Highly selective
            columns=['indexed_col', 'other_col'],
            total_columns=10,
            estimated_rows=10000000  # Very large dataset
        )
        
        # With very large dataset and selective filter, might choose index scan
        assert strategy in [OptimizationStrategy.INDEX_SCAN, OptimizationStrategy.FILTER_PUSHDOWN]
    
    def test_full_scan_fallback(self, optimizer):
        """Test fallback to full scan strategy."""
        strategy = optimizer._choose_optimization_strategy(
            filters=[],  # No filters
            columns=list(range(20)),  # All columns
            total_columns=20,
            estimated_rows=1000
        )
        
        assert strategy == OptimizationStrategy.FULL_SCAN


class TestQueryOptimizerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for edge case tests."""
        return QueryOptimizer()
    
    def test_empty_dataframe(self, optimizer):
        """Test optimization with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        plan = optimizer.create_query_plan(
            filters=[('col', '=', 'val')],
            columns=['col'],
            df=empty_df
        )
        
        # Should handle empty DataFrame gracefully
        assert plan.estimated_cost == 0
        assert len(plan.filters) == 1
    
    def test_invalid_filter_column(self, optimizer):
        """Test handling of invalid filter column."""
        df = pd.DataFrame({'valid_col': [1, 2, 3]})
        
        plan = optimizer.create_query_plan(
            filters=[('invalid_col', '=', 'val')],
            columns=['valid_col'],
            df=df
        )
        
        # Should handle gracefully, possibly with default selectivity
        assert len(plan.filters) == 1
        assert plan.filters[0].selectivity == 0.5  # Default selectivity
    
    def test_very_large_limit(self, optimizer):
        """Test handling of limit larger than dataset."""
        df = pd.DataFrame({'col': range(100)})
        
        plan = optimizer.create_query_plan(
            filters=[],
            columns=['col'],
            limit=1000,  # Larger than dataset
            df=df
        )
        
        result = optimizer.apply_query_plan(plan, df)
        
        # Should return all available rows
        assert len(result) == 100
    
    def test_unsupported_operator(self, optimizer):
        """Test handling of unsupported filter operators."""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        
        # This should handle unknown operators gracefully
        filters = [('col', 'unknown_op', 'val')]
        plan = optimizer.create_query_plan(filters, ['col'], df=df)
        
        # Should create filter with default selectivity
        assert len(plan.filters) == 1
        assert plan.filters[0].selectivity == 0.5