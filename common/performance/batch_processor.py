"""Batch processing optimization for bulk data operations."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Iterator, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStrategy(Enum):
    """Batch processing strategies."""
    SEQUENTIAL = "sequential"      # Process batches one by one
    PARALLEL_THREAD = "parallel_thread"  # Process batches in parallel threads
    PARALLEL_PROCESS = "parallel_process"  # Process batches in parallel processes
    ADAPTIVE = "adaptive"          # Choose strategy based on workload


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 1000
    max_workers: int = 4
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    timeout_per_batch: Optional[float] = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    memory_limit_mb: int = 1024
    enable_progress_tracking: bool = True


@dataclass
class BatchResult:
    """Result of batch processing."""
    success_count: int
    error_count: int
    total_items: int
    execution_time: float
    batches_processed: int
    errors: List[Dict[str, Any]]
    results: List[Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.success_count / max(self.total_items, 1)
    
    @property
    def throughput(self) -> float:
        """Calculate items per second."""
        return self.total_items / max(self.execution_time, 0.001)


class BatchProcessor(Generic[T, R]):
    """High-performance batch processor for bulk operations."""
    
    def __init__(self, config: BatchConfig = None):
        """Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self._stats = {
            "total_batches": 0,
            "total_items": 0,
            "total_time": 0.0,
            "success_count": 0,
            "error_count": 0
        }
        
    def process_batches(
        self,
        items: List[T],
        processor_func: Callable[[List[T]], List[R]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process items in batches.
        
        Args:
            items: Items to process
            processor_func: Function to process each batch
            progress_callback: Optional progress callback (current, total)
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        
        # Create batches
        batches = self._create_batches(items)
        total_batches = len(batches)
        
        logger.info(f"Processing {len(items)} items in {total_batches} batches "
                   f"(batch_size={self.config.batch_size}, strategy={self.config.strategy.value})")
        
        # Choose processing strategy
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            strategy = self._choose_strategy(items, batches)
        else:
            strategy = self.config.strategy
        
        # Process batches according to strategy
        if strategy == BatchStrategy.SEQUENTIAL:
            result = self._process_sequential(batches, processor_func, progress_callback)
        elif strategy == BatchStrategy.PARALLEL_THREAD:
            result = self._process_parallel_threads(batches, processor_func, progress_callback)
        elif strategy == BatchStrategy.PARALLEL_PROCESS:
            result = self._process_parallel_processes(batches, processor_func, progress_callback)
        else:
            # Fallback to sequential
            result = self._process_sequential(batches, processor_func, progress_callback)
        
        # Update statistics
        execution_time = time.time() - start_time
        self._update_stats(result, execution_time)
        
        logger.info(f"Batch processing completed: {result.success_count}/{result.total_items} successful "
                   f"({result.success_rate:.2%}) in {execution_time:.2f}s "
                   f"({result.throughput:.1f} items/sec)")
        
        return result
    
    def process_dataframe_batches(
        self,
        df: pd.DataFrame,
        processor_func: Callable[[pd.DataFrame], pd.DataFrame],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """Process DataFrame in batches.
        
        Args:
            df: DataFrame to process
            processor_func: Function to process each batch
            progress_callback: Optional progress callback
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        start_time = time.time()
        batch_size = self.config.batch_size
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        logger.info(f"Processing DataFrame with {len(df)} rows in {total_batches} batches")
        
        results = []
        errors = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                processed_batch = self._process_single_batch(
                    batch, processor_func, f"batch_{batch_num}"
                )
                results.append(processed_batch)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                errors.append({"batch": batch_num, "error": str(e)})
                # Optionally include failed batch as-is
                results.append(batch)
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_num, total_batches)
        
        # Combine results
        combined_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        
        execution_time = time.time() - start_time
        logger.info(f"DataFrame batch processing completed in {execution_time:.2f}s")
        
        return combined_df
    
    async def process_batches_async(
        self,
        items: List[T],
        processor_func: Callable[[List[T]], R],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process batches asynchronously.
        
        Args:
            items: Items to process
            processor_func: Async function to process each batch
            progress_callback: Optional progress callback
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        batches = self._create_batches(items)
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches asynchronously")
        
        # Create semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_batch_with_semaphore(batch_data, batch_idx):
            async with semaphore:
                return await self._process_single_batch_async(
                    batch_data, processor_func, f"async_batch_{batch_idx}"
                )
        
        # Process all batches concurrently
        tasks = [
            process_batch_with_semaphore(batch, idx) 
            for idx, batch in enumerate(batches)
        ]
        
        results = []
        errors = []
        completed = 0
        
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(batches))
                    
            except Exception as e:
                errors.append({"batch": completed, "error": str(e)})
                completed += 1
        
        # Create batch result
        execution_time = time.time() - start_time
        success_count = len(results) * self.config.batch_size
        error_count = len(errors) * self.config.batch_size
        
        batch_result = BatchResult(
            success_count=success_count,
            error_count=error_count,
            total_items=len(items),
            execution_time=execution_time,
            batches_processed=len(batches),
            errors=errors,
            results=results
        )
        
        return batch_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        avg_time = self._stats["total_time"] / max(self._stats["total_batches"], 1)
        avg_throughput = self._stats["total_items"] / max(self._stats["total_time"], 1)
        
        return {
            "total_batches": self._stats["total_batches"],
            "total_items": self._stats["total_items"],
            "total_time": self._stats["total_time"],
            "success_count": self._stats["success_count"],
            "error_count": self._stats["error_count"],
            "success_rate": self._stats["success_count"] / max(self._stats["total_items"], 1),
            "average_batch_time": avg_time,
            "average_throughput": avg_throughput,
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "strategy": self.config.strategy.value
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            "total_batches": 0,
            "total_items": 0,
            "total_time": 0.0,
            "success_count": 0,
            "error_count": 0
        }
    
    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Create batches from items list."""
        batch_size = self.config.batch_size
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    def _choose_strategy(self, items: List[T], batches: List[List[T]]) -> BatchStrategy:
        """Choose optimal processing strategy based on workload."""
        total_items = len(items)
        num_batches = len(batches)
        
        # For small datasets, use sequential processing
        if total_items < 1000 or num_batches < 3:
            return BatchStrategy.SEQUENTIAL
        
        # For CPU-bound tasks with many batches, use processes
        if num_batches >= self.config.max_workers * 2:
            return BatchStrategy.PARALLEL_PROCESS
        
        # Default to threads for I/O-bound tasks
        return BatchStrategy.PARALLEL_THREAD
    
    def _process_sequential(
        self,
        batches: List[List[T]],
        processor_func: Callable[[List[T]], List[R]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process batches sequentially."""
        results = []
        errors = []
        
        for i, batch in enumerate(batches):
            try:
                batch_result = self._process_single_batch(
                    batch, processor_func, f"seq_batch_{i}"
                )
                results.append(batch_result)
                
            except Exception as e:
                errors.append({"batch": i, "error": str(e)})
            
            if progress_callback:
                progress_callback(i + 1, len(batches))
        
        return self._create_batch_result(results, errors, len(batches))
    
    def _process_parallel_threads(
        self,
        batches: List[List[T]],
        processor_func: Callable[[List[T]], List[R]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process batches in parallel threads."""
        results = []
        errors = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_single_batch, batch, processor_func, f"thread_batch_{i}"): i
                for i, batch in enumerate(batches)
            }
            
            # Process completed futures
            for future in as_completed(future_to_batch, timeout=self.config.timeout_per_batch):
                batch_idx = future_to_batch[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append({"batch": batch_idx, "error": str(e)})
                
                if progress_callback:
                    progress_callback(completed, len(batches))
        
        return self._create_batch_result(results, errors, len(batches))
    
    def _process_parallel_processes(
        self,
        batches: List[List[T]],
        processor_func: Callable[[List[T]], List[R]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process batches in parallel processes."""
        results = []
        errors = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(processor_func, batch): i
                for i, batch in enumerate(batches)
            }
            
            # Process completed futures
            for future in as_completed(future_to_batch, timeout=self.config.timeout_per_batch):
                batch_idx = future_to_batch[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append({"batch": batch_idx, "error": str(e)})
                
                if progress_callback:
                    progress_callback(completed, len(batches))
        
        return self._create_batch_result(results, errors, len(batches))
    
    def _process_single_batch(
        self,
        batch: List[T],
        processor_func: Callable[[List[T]], R],
        batch_id: str
    ) -> R:
        """Process a single batch with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f"Processing {batch_id} (attempt {attempt + 1})")
                result = processor_func(batch)
                return result
                
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(f"Batch {batch_id} failed on attempt {attempt + 1}, retrying: {e}")
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Batch {batch_id} failed after {self.config.retry_attempts} attempts: {e}")
                    raise
    
    async def _process_single_batch_async(
        self,
        batch: List[T],
        processor_func: Callable[[List[T]], R],
        batch_id: str
    ) -> R:
        """Process a single batch asynchronously with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f"Processing {batch_id} async (attempt {attempt + 1})")
                result = await processor_func(batch)
                return result
                
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(f"Batch {batch_id} failed on attempt {attempt + 1}, retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Batch {batch_id} failed after {self.config.retry_attempts} attempts: {e}")
                    raise
    
    def _create_batch_result(
        self,
        results: List[Any],
        errors: List[Dict[str, Any]],
        total_batches: int
    ) -> BatchResult:
        """Create BatchResult from processing results."""
        success_count = len(results) * self.config.batch_size
        error_count = len(errors) * self.config.batch_size
        total_items = success_count + error_count
        
        return BatchResult(
            success_count=success_count,
            error_count=error_count,
            total_items=total_items,
            execution_time=0.0,  # Will be set by caller
            batches_processed=total_batches,
            errors=errors,
            results=results
        )
    
    def _update_stats(self, result: BatchResult, execution_time: float) -> None:
        """Update processing statistics."""
        result.execution_time = execution_time
        
        self._stats["total_batches"] += result.batches_processed
        self._stats["total_items"] += result.total_items
        self._stats["total_time"] += execution_time
        self._stats["success_count"] += result.success_count
        self._stats["error_count"] += result.error_count


class DataFrameBatchProcessor:
    """Specialized batch processor for pandas DataFrames."""
    
    def __init__(self, chunk_size: int = 10000, max_memory_mb: int = 512):
        """Initialize DataFrame batch processor.
        
        Args:
            chunk_size: Number of rows per chunk
            max_memory_mb: Maximum memory usage in MB
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
    
    def process_large_dataframe(
        self,
        df: pd.DataFrame,
        processor_func: Callable[[pd.DataFrame], pd.DataFrame],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """Process large DataFrame in memory-efficient chunks.
        
        Args:
            df: Large DataFrame to process
            processor_func: Function to process each chunk
            progress_callback: Optional progress callback
            
        Returns:
            Processed DataFrame
        """
        if len(df) <= self.chunk_size:
            return processor_func(df)
        
        chunks = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"Processing large DataFrame ({len(df)} rows) in {total_chunks} chunks")
        
        for i, chunk in enumerate(self._chunk_dataframe(df)):
            try:
                processed_chunk = processor_func(chunk)
                chunks.append(processed_chunk)
                
                if progress_callback:
                    progress_callback(i + 1, total_chunks)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                # Optionally include failed chunk as-is
                chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    
    def _chunk_dataframe(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Generate DataFrame chunks."""
        for i in range(0, len(df), self.chunk_size):
            yield df.iloc[i:i + self.chunk_size].copy()


# Utility functions for common batch operations
def batch_save_to_parquet(
    df: pd.DataFrame,
    file_path: str,
    batch_size: int = 100000,
    compression: str = "snappy"
) -> None:
    """Save large DataFrame to Parquet in batches.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        batch_size: Rows per batch
        compression: Compression algorithm
    """
    if len(df) <= batch_size:
        df.to_parquet(file_path, compression=compression, index=False)
        return
    
    logger.info(f"Saving large DataFrame ({len(df)} rows) to Parquet in batches")
    
    # Save first batch to create file
    first_batch = df.iloc[:batch_size]
    first_batch.to_parquet(file_path, compression=compression, index=False)
    
    # Append remaining batches
    for i in range(batch_size, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        
        # For append mode, we need to use PyArrow directly
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        table = pa.Table.from_pandas(batch)
        
        # Read existing file and append
        existing_table = pq.read_table(file_path)
        combined_table = pa.concat_tables([existing_table, table])
        
        pq.write_table(combined_table, file_path, compression=compression)
    
    logger.info(f"Saved DataFrame to {file_path} successfully")


def batch_execute_queries(
    queries: List[str],
    connection,
    batch_size: int = 100,
    max_workers: int = 4
) -> List[Any]:
    """Execute SQL queries in batches.
    
    Args:
        queries: List of SQL queries to execute
        connection: Database connection
        batch_size: Queries per batch
        max_workers: Maximum parallel workers
        
    Returns:
        List of query results
    """
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        strategy=BatchStrategy.PARALLEL_THREAD
    )
    
    processor = BatchProcessor(config)
    
    def execute_query_batch(query_batch: List[str]) -> List[Any]:
        results = []
        for query in query_batch:
            cursor = connection.cursor()
            cursor.execute(query)
            results.append(cursor.fetchall())
            cursor.close()
        return results
    
    batch_result = processor.process_batches(queries, execute_query_batch)
    
    # Flatten results
    all_results = []
    for batch_results in batch_result.results:
        all_results.extend(batch_results)
    
    return all_results