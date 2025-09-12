"""Tests for batch processing optimization."""

import asyncio
import time
import pandas as pd
import pytest
from unittest.mock import Mock, patch, call
from concurrent.futures import TimeoutError as FutureTimeoutError

from common.performance.batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchStrategy,
    BatchResult,
    DataFrameBatchProcessor,
    batch_save_to_parquet,
    batch_execute_queries
)


class TestBatchConfig:
    """Test batch configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()
        
        assert config.batch_size == 1000
        assert config.max_workers == 4
        assert config.strategy == BatchStrategy.ADAPTIVE
        assert config.timeout_per_batch == 300.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.memory_limit_mb == 1024
        assert config.enable_progress_tracking is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatchConfig(
            batch_size=500,
            max_workers=8,
            strategy=BatchStrategy.PARALLEL_THREAD,
            timeout_per_batch=60.0,
            retry_attempts=5
        )
        
        assert config.batch_size == 500
        assert config.max_workers == 8
        assert config.strategy == BatchStrategy.PARALLEL_THREAD
        assert config.timeout_per_batch == 60.0
        assert config.retry_attempts == 5


class TestBatchResult:
    """Test batch result class."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = BatchResult(
            success_count=80,
            error_count=20,
            total_items=100,
            execution_time=10.0,
            batches_processed=5,
            errors=[],
            results=[]
        )
        
        assert result.success_rate == 0.8
    
    def test_success_rate_zero_items(self):
        """Test success rate with zero items."""
        result = BatchResult(
            success_count=0,
            error_count=0,
            total_items=0,
            execution_time=0.0,
            batches_processed=0,
            errors=[],
            results=[]
        )
        
        assert result.success_rate == 0.0
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        result = BatchResult(
            success_count=100,
            error_count=0,
            total_items=100,
            execution_time=10.0,
            batches_processed=2,
            errors=[],
            results=[]
        )
        
        assert result.throughput == 10.0  # 100 items / 10 seconds
    
    def test_throughput_zero_time(self):
        """Test throughput with zero execution time."""
        result = BatchResult(
            success_count=100,
            error_count=0,
            total_items=100,
            execution_time=0.0,
            batches_processed=1,
            errors=[],
            results=[]
        )
        
        assert result.throughput == 100000.0  # Uses 0.001 minimum


class TestBatchProcessor:
    """Test batch processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create batch processor with test configuration."""
        config = BatchConfig(
            batch_size=10,
            max_workers=2,
            retry_attempts=2,
            timeout_per_batch=1.0
        )
        return BatchProcessor(config)
    
    @pytest.fixture
    def mock_processor_func(self):
        """Mock processor function."""
        def process_batch(items):
            return [f"processed_{item}" for item in items]
        return Mock(side_effect=process_batch)
    
    def test_create_batches(self, processor):
        """Test batch creation."""
        items = list(range(25))  # 25 items with batch_size=10
        batches = processor._create_batches(items)
        
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
    
    def test_choose_strategy_small_dataset(self, processor):
        """Test strategy selection for small datasets."""
        items = list(range(50))
        batches = processor._create_batches(items)
        strategy = processor._choose_strategy(items, batches)
        
        assert strategy == BatchStrategy.SEQUENTIAL
    
    def test_choose_strategy_large_dataset(self, processor):
        """Test strategy selection for large datasets."""
        items = list(range(5000))
        batches = processor._create_batches(items)
        strategy = processor._choose_strategy(items, batches)
        
        # Should choose parallel processing for large datasets
        assert strategy in [BatchStrategy.PARALLEL_THREAD, BatchStrategy.PARALLEL_PROCESS]
    
    def test_sequential_processing(self, processor, mock_processor_func):
        """Test sequential batch processing."""
        items = list(range(25))
        result = processor.process_batches(items, mock_processor_func)
        
        assert result.total_items == 25
        assert result.success_count == 30  # 3 batches * 10 batch_size
        assert result.error_count == 0
        assert result.batches_processed == 3
        assert len(result.results) == 3
    
    def test_processing_with_progress_callback(self, processor, mock_processor_func):
        """Test processing with progress callback."""
        items = list(range(25))
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        processor.process_batches(items, mock_processor_func, progress_callback)
        
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)  # Final call should be (total, total)
    
    def test_processing_with_errors(self, processor):
        """Test processing with batch errors."""
        items = list(range(25))
        
        def failing_processor(batch):
            if len(batch) == 5:  # Last batch will fail
                raise ValueError("Simulated error")
            return [f"processed_{item}" for item in batch]
        
        result = processor.process_batches(items, failing_processor)
        
        assert result.success_count == 20  # 2 successful batches * 10 batch_size
        assert result.error_count == 10   # 1 failed batch * 10 batch_size
        assert len(result.errors) == 1
        assert "Simulated error" in result.errors[0]["error"]
    
    def test_retry_logic(self, processor):
        """Test retry logic for failed batches."""
        items = list(range(10))
        attempt_count = 0
        
        def unreliable_processor(batch):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ValueError("First attempt fails")
            return [f"processed_{item}" for item in batch]
        
        result = processor.process_batches(items, unreliable_processor)
        
        assert result.success_count == 10
        assert result.error_count == 0
        assert attempt_count == 2  # Should have retried once
    
    def test_statistics_tracking(self, processor, mock_processor_func):
        """Test statistics tracking."""
        items = list(range(20))
        processor.process_batches(items, mock_processor_func)
        
        stats = processor.get_statistics()
        
        assert stats["total_batches"] == 2
        assert stats["total_items"] == 20
        assert stats["success_count"] == 20
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["average_throughput"] > 0
    
    def test_reset_statistics(self, processor, mock_processor_func):
        """Test statistics reset."""
        items = list(range(10))
        processor.process_batches(items, mock_processor_func)
        
        # Verify stats were recorded
        stats = processor.get_statistics()
        assert stats["total_items"] > 0
        
        # Reset and verify
        processor.reset_statistics()
        stats = processor.get_statistics()
        assert stats["total_items"] == 0
        assert stats["total_batches"] == 0


class TestAsyncBatchProcessor:
    """Test async batch processing."""
    
    @pytest.fixture
    def processor(self):
        """Create batch processor for async tests."""
        config = BatchConfig(batch_size=5, max_workers=2)
        return BatchProcessor(config)
    
    @pytest.mark.asyncio
    async def test_async_processing(self, processor):
        """Test asynchronous batch processing."""
        items = list(range(15))
        
        async def async_processor_func(batch):
            await asyncio.sleep(0.1)  # Simulate async work
            return [f"async_processed_{item}" for item in batch]
        
        result = await processor.process_batches_async(items, async_processor_func)
        
        assert result.total_items == 15
        assert result.success_count == 15  # 3 batches * 5 batch_size
        assert result.error_count == 0
        assert len(result.results) == 3
    
    @pytest.mark.asyncio
    async def test_async_processing_with_errors(self, processor):
        """Test async processing with errors."""
        items = list(range(10))
        
        async def failing_async_processor(batch):
            if batch[0] >= 5:  # Second batch fails
                raise ValueError("Async error")
            await asyncio.sleep(0.1)
            return [f"processed_{item}" for item in batch]
        
        result = await processor.process_batches_async(items, failing_async_processor)
        
        assert result.success_count == 5   # 1 successful batch * 5 batch_size
        assert result.error_count == 5     # 1 failed batch * 5 batch_size
        assert len(result.errors) == 1


class TestDataFrameBatchProcessor:
    """Test DataFrame batch processing."""
    
    @pytest.fixture
    def df_processor(self):
        """Create DataFrame batch processor."""
        return DataFrameBatchProcessor(chunk_size=100, max_memory_mb=64)
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'id': range(250),
            'value': range(250, 500),
            'text': [f'item_{i}' for i in range(250)]
        })
    
    def test_small_dataframe_processing(self, df_processor, sample_df):
        """Test processing DataFrame smaller than chunk size."""
        small_df = sample_df.head(50)
        
        def double_values(df):
            result = df.copy()
            result['value'] = result['value'] * 2
            return result
        
        result = df_processor.process_large_dataframe(small_df, double_values)
        
        assert len(result) == 50
        assert result['value'].iloc[0] == 500  # 250 * 2
    
    def test_large_dataframe_processing(self, df_processor, sample_df):
        """Test processing DataFrame larger than chunk size."""
        def add_processed_flag(df):
            result = df.copy()
            result['processed'] = True
            return result
        
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        result = df_processor.process_large_dataframe(
            sample_df, add_processed_flag, progress_callback
        )
        
        assert len(result) == 250
        assert 'processed' in result.columns
        assert all(result['processed'])
        assert len(progress_calls) == 3  # 250/100 = 3 chunks (rounded up)
    
    def test_dataframe_chunking(self, df_processor, sample_df):
        """Test DataFrame chunking logic."""
        chunks = list(df_processor._chunk_dataframe(sample_df))
        
        assert len(chunks) == 3  # 250/100 = 3 chunks (rounded up)
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50


class TestBatchProcessor_DataFrame_Integration:
    """Test BatchProcessor DataFrame methods."""
    
    @pytest.fixture
    def processor(self):
        """Create batch processor."""
        config = BatchConfig(batch_size=50)
        return BatchProcessor(config)
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'a': range(150),
            'b': range(150, 300)
        })
    
    def test_dataframe_batch_processing(self, processor, sample_df):
        """Test DataFrame batch processing."""
        def multiply_columns(df):
            result = df.copy()
            result['c'] = result['a'] * result['b']
            return result
        
        result = processor.process_dataframe_batches(sample_df, multiply_columns)
        
        assert len(result) == 150
        assert 'c' in result.columns
        assert result['c'].iloc[0] == 0  # 0 * 150
    
    def test_empty_dataframe(self, processor):
        """Test processing empty DataFrame."""
        empty_df = pd.DataFrame()
        
        def dummy_processor(df):
            return df
        
        result = processor.process_dataframe_batches(empty_df, dummy_processor)
        
        assert len(result) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('common.performance.batch_processor.pd.DataFrame.to_parquet')
    def test_batch_save_small_df(self, mock_to_parquet):
        """Test batch save for small DataFrame."""
        df = pd.DataFrame({'a': range(100)})
        
        batch_save_to_parquet(df, 'test.parquet', batch_size=200)
        
        mock_to_parquet.assert_called_once()
    
    @patch('common.performance.batch_processor.pq.write_table')
    @patch('common.performance.batch_processor.pq.read_table')
    @patch('common.performance.batch_processor.pd.DataFrame.to_parquet')
    def test_batch_save_large_df(self, mock_to_parquet, mock_read_table, mock_write_table):
        """Test batch save for large DataFrame."""
        df = pd.DataFrame({'a': range(250)})
        
        # Mock the PyArrow operations
        mock_table = Mock()
        mock_read_table.return_value = mock_table
        
        batch_save_to_parquet(df, 'test.parquet', batch_size=100)
        
        # Should save first batch normally, then append others
        mock_to_parquet.assert_called_once()
        assert mock_write_table.call_count >= 1
    
    def test_batch_execute_queries(self):
        """Test batch query execution."""
        queries = [f"SELECT {i}" for i in range(15)]
        
        # Mock connection and cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [(1,)]
        
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        
        results = batch_execute_queries(queries, mock_connection, batch_size=5)
        
        assert len(results) == 15
        assert all(result == [(1,)] for result in results)
        assert mock_connection.cursor.call_count == 15  # One cursor per query


class TestBatchStrategies:
    """Test different batch processing strategies."""
    
    @pytest.fixture
    def items(self):
        """Create test items."""
        return list(range(50))
    
    @pytest.fixture
    def mock_processor_func(self):
        """Mock processor function with timing."""
        def process_batch(batch):
            time.sleep(0.01)  # Simulate some work
            return [f"result_{item}" for item in batch]
        return process_batch
    
    def test_sequential_strategy(self, items, mock_processor_func):
        """Test sequential processing strategy."""
        config = BatchConfig(
            batch_size=10,
            strategy=BatchStrategy.SEQUENTIAL,
            max_workers=2
        )
        processor = BatchProcessor(config)
        
        result = processor.process_batches(items, mock_processor_func)
        
        assert result.success_count == 50
        assert result.error_count == 0
        assert result.batches_processed == 5
    
    def test_parallel_thread_strategy(self, items, mock_processor_func):
        """Test parallel thread processing strategy."""
        config = BatchConfig(
            batch_size=10,
            strategy=BatchStrategy.PARALLEL_THREAD,
            max_workers=2
        )
        processor = BatchProcessor(config)
        
        result = processor.process_batches(items, mock_processor_func)
        
        assert result.success_count == 50
        assert result.error_count == 0
        assert result.batches_processed == 5
    
    def test_adaptive_strategy_small_dataset(self, mock_processor_func):
        """Test adaptive strategy chooses sequential for small datasets."""
        items = list(range(100))  # Small dataset
        config = BatchConfig(
            batch_size=50,
            strategy=BatchStrategy.ADAPTIVE,
            max_workers=2
        )
        processor = BatchProcessor(config)
        
        # Should choose sequential for small datasets
        batches = processor._create_batches(items)
        strategy = processor._choose_strategy(items, batches)
        assert strategy == BatchStrategy.SEQUENTIAL
    
    def test_adaptive_strategy_large_dataset(self, mock_processor_func):
        """Test adaptive strategy chooses parallel for large datasets."""
        items = list(range(5000))  # Large dataset
        config = BatchConfig(
            batch_size=200,
            strategy=BatchStrategy.ADAPTIVE,
            max_workers=4
        )
        processor = BatchProcessor(config)
        
        # Should choose parallel for large datasets
        batches = processor._create_batches(items)
        strategy = processor._choose_strategy(items, batches)
        assert strategy in [BatchStrategy.PARALLEL_THREAD, BatchStrategy.PARALLEL_PROCESS]