"""Query optimization utilities for PyArrow datasets."""

import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
from datetime import date, datetime

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for optimization."""
    POINT_LOOKUP = "point_lookup"  # Single record lookup
    RANGE_SCAN = "range_scan"      # Date/time range queries
    FILTER_SCAN = "filter_scan"    # Filtered table scans
    AGGREGATION = "aggregation"    # Group by / aggregation queries


@dataclass
class QueryPlan:
    """Represents an optimized query execution plan."""
    query_type: QueryType
    filters: List[Tuple[str, str, Any]]
    columns: Optional[List[str]]
    partitions_to_scan: List[str]
    estimated_rows: int
    use_pushdown: bool
    index_hints: List[str]
    execution_time_estimate: float


class QueryOptimizer:
    """Optimizes PyArrow queries for better performance."""
    
    def __init__(self, dataset_path: str, partition_columns: List[str] = None):
        """Initialize query optimizer.
        
        Args:
            dataset_path: Path to the PyArrow dataset
            partition_columns: List of partitioned columns
        """
        self.dataset_path = dataset_path
        self.partition_columns = partition_columns or []
        self._dataset_cache: Optional[ds.Dataset] = None
        self._schema_cache: Optional[pa.Schema] = None
        self._statistics_cache: Dict[str, Any] = {}
        self._query_history: List[Dict[str, Any]] = []
        
    def get_dataset(self) -> ds.Dataset:
        """Get cached dataset instance."""
        if self._dataset_cache is None:
            self._dataset_cache = ds.dataset(self.dataset_path, format="parquet")
        return self._dataset_cache
    
    def get_schema(self) -> pa.Schema:
        """Get cached schema."""
        if self._schema_cache is None:
            dataset = self.get_dataset()
            self._schema_cache = dataset.schema
        return self._schema_cache
    
    def create_query_plan(
        self,
        filters: List[Tuple[str, str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> QueryPlan:
        """Create an optimized query plan.
        
        Args:
            filters: List of filter tuples (column, operator, value)
            columns: Columns to select (None for all)
            limit: Maximum rows to return
            
        Returns:
            Optimized query plan
        """
        filters = filters or []
        
        # Analyze query type
        query_type = self._classify_query(filters, columns, limit)
        
        # Optimize filters
        optimized_filters = self._optimize_filters(filters)
        
        # Determine partition pruning
        partitions_to_scan = self._get_partitions_to_scan(optimized_filters)
        
        # Estimate query cost
        estimated_rows = self._estimate_result_size(optimized_filters, limit)
        
        # Determine if we can use filter pushdown
        use_pushdown = self._can_use_filter_pushdown(optimized_filters)
        
        # Generate index hints
        index_hints = self._get_index_hints(optimized_filters)
        
        # Estimate execution time
        execution_time = self._estimate_execution_time(
            query_type, estimated_rows, len(partitions_to_scan)
        )
        
        plan = QueryPlan(
            query_type=query_type,
            filters=optimized_filters,
            columns=columns,
            partitions_to_scan=partitions_to_scan,
            estimated_rows=estimated_rows,
            use_pushdown=use_pushdown,
            index_hints=index_hints,
            execution_time_estimate=execution_time
        )
        
        logger.debug(f"Created query plan: {plan.query_type.value}, "
                    f"estimated_rows={plan.estimated_rows}, "
                    f"partitions={len(plan.partitions_to_scan)}")
        
        return plan
    
    def execute_optimized_query(
        self,
        filters: List[Tuple[str, str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        use_threads: bool = True
    ) -> pd.DataFrame:
        """Execute an optimized query.
        
        Args:
            filters: Query filters
            columns: Columns to select
            limit: Row limit
            use_threads: Whether to use multithreading
            
        Returns:
            Query results as DataFrame
        """
        start_time = time.time()
        
        # Create query plan
        plan = self.create_query_plan(filters, columns, limit)
        
        try:
            # Get dataset
            dataset = self.get_dataset()
            
            # Build PyArrow filter expression
            filter_expression = self._build_filter_expression(plan.filters)
            
            # Execute query with optimizations
            if plan.use_pushdown and filter_expression:
                # Use filter pushdown for better performance
                table = dataset.to_table(
                    filter=filter_expression,
                    columns=plan.columns,
                    use_threads=use_threads
                )
            else:
                # Load and filter in memory
                table = dataset.to_table(
                    columns=plan.columns,
                    use_threads=use_threads
                )
                
                if filter_expression:
                    table = table.filter(filter_expression)
            
            # Apply limit if specified
            if limit is not None:
                table = table.slice(0, limit)
            
            # Convert to pandas
            df = table.to_pandas()
            
            # Record query statistics
            execution_time = time.time() - start_time
            self._record_query_stats(plan, execution_time, len(df))
            
            logger.debug(f"Query executed in {execution_time:.3f}s, "
                        f"returned {len(df)} rows")
            
            return df
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            raise
    
    def _classify_query(
        self,
        filters: List[Tuple[str, str, Any]],
        columns: Optional[List[str]],
        limit: Optional[int]
    ) -> QueryType:
        """Classify the type of query for optimization."""
        if not filters:
            return QueryType.FILTER_SCAN
        
        # Check for point lookups (equality filters on key columns)
        equality_filters = [f for f in filters if f[1] == "="]
        if equality_filters and limit == 1:
            return QueryType.POINT_LOOKUP
        
        # Check for range scans (date/time range filters)
        range_operators = {"<", "<=", ">", ">=", "between"}
        has_range = any(f[1] in range_operators for f in filters)
        date_columns = {"date", "timestamp", "created_at", "updated_at"}
        has_date_filter = any(f[0] in date_columns for f in filters)
        
        if has_range and has_date_filter:
            return QueryType.RANGE_SCAN
        
        # Check for aggregation queries (would need additional metadata)
        # For now, default to filter scan
        return QueryType.FILTER_SCAN
    
    def _optimize_filters(
        self, 
        filters: List[Tuple[str, str, Any]]
    ) -> List[Tuple[str, str, Any]]:
        """Optimize filter conditions."""
        optimized = []
        
        for column, operator, value in filters:
            # Optimize date filters
            if self._is_date_column(column):
                optimized_filter = self._optimize_date_filter(column, operator, value)
                optimized.append(optimized_filter)
            
            # Optimize string filters  
            elif self._is_string_column(column):
                optimized_filter = self._optimize_string_filter(column, operator, value)
                optimized.append(optimized_filter)
            
            # Optimize numeric filters
            elif self._is_numeric_column(column):
                optimized_filter = self._optimize_numeric_filter(column, operator, value)
                optimized.append(optimized_filter)
            
            else:
                optimized.append((column, operator, value))
        
        # Sort filters by selectivity (most selective first)
        return self._sort_filters_by_selectivity(optimized)
    
    def _get_partitions_to_scan(
        self, 
        filters: List[Tuple[str, str, Any]]
    ) -> List[str]:
        """Determine which partitions need to be scanned."""
        if not self.partition_columns:
            return ["*"]  # All partitions
        
        partitions = []
        
        # For each partition column, check if we have filters
        for partition_col in self.partition_columns:
            partition_filters = [f for f in filters if f[0] == partition_col]
            
            if partition_filters:
                # Calculate partition values from filters
                partition_values = self._calculate_partition_values(
                    partition_col, partition_filters
                )
                partitions.extend(partition_values)
            else:
                # No filter on this partition column, scan all
                partitions.append("*")
                break
        
        return partitions or ["*"]
    
    def _estimate_result_size(
        self, 
        filters: List[Tuple[str, str, Any]], 
        limit: Optional[int]
    ) -> int:
        """Estimate the number of rows that will be returned."""
        # This is a simplified estimation
        # In practice, you'd use table statistics
        
        base_estimate = 10000  # Default estimate
        
        # Apply selectivity estimates for each filter
        for column, operator, value in filters:
            selectivity = self._estimate_filter_selectivity(column, operator, value)
            base_estimate = int(base_estimate * selectivity)
        
        if limit is not None:
            base_estimate = min(base_estimate, limit)
        
        return max(1, base_estimate)
    
    def _can_use_filter_pushdown(self, filters: List[Tuple[str, str, Any]]) -> bool:
        """Check if filters can be pushed down to PyArrow."""
        # PyArrow supports pushdown for most common filters
        supported_operators = {"=", "<", "<=", ">", ">=", "!=", "in"}
        
        for column, operator, value in filters:
            if operator not in supported_operators:
                return False
            
            # Check if column type is supported for pushdown
            if not self._is_pushdown_supported(column, operator, value):
                return False
        
        return True
    
    def _get_index_hints(self, filters: List[Tuple[str, str, Any]]) -> List[str]:
        """Generate index hints for the query."""
        hints = []
        
        # Suggest creating indices on frequently filtered columns
        filter_columns = [f[0] for f in filters]
        for column in set(filter_columns):
            hints.append(f"consider_index_{column}")
        
        return hints
    
    def _estimate_execution_time(
        self, 
        query_type: QueryType, 
        estimated_rows: int, 
        partition_count: int
    ) -> float:
        """Estimate query execution time in seconds."""
        # Simple time estimation model
        base_time = {
            QueryType.POINT_LOOKUP: 0.001,
            QueryType.RANGE_SCAN: 0.01,
            QueryType.FILTER_SCAN: 0.1,
            QueryType.AGGREGATION: 0.5
        }.get(query_type, 0.1)
        
        # Scale by data size
        row_factor = (estimated_rows / 10000) * 0.01
        partition_factor = partition_count * 0.001
        
        return base_time + row_factor + partition_factor
    
    def _build_filter_expression(
        self, 
        filters: List[Tuple[str, str, Any]]
    ) -> Optional[pa.Expression]:
        """Build PyArrow filter expression from filters."""
        if not filters:
            return None
        
        expressions = []
        
        for column, operator, value in filters:
            try:
                if operator == "=":
                    expr = pc.equal(pc.field(column), pa.scalar(value))
                elif operator == "<":
                    expr = pc.less(pc.field(column), pa.scalar(value))
                elif operator == "<=":
                    expr = pc.less_equal(pc.field(column), pa.scalar(value))
                elif operator == ">":
                    expr = pc.greater(pc.field(column), pa.scalar(value))
                elif operator == ">=":
                    expr = pc.greater_equal(pc.field(column), pa.scalar(value))
                elif operator == "!=":
                    expr = pc.not_equal(pc.field(column), pa.scalar(value))
                elif operator == "in" and isinstance(value, (list, tuple)):
                    expr = pc.is_in(pc.field(column), pa.array(value))
                else:
                    logger.warning(f"Unsupported operator: {operator}")
                    continue
                
                expressions.append(expr)
                
            except Exception as e:
                logger.warning(f"Failed to build filter for {column} {operator} {value}: {e}")
                continue
        
        if not expressions:
            return None
        
        # Combine expressions with AND
        result = expressions[0]
        for expr in expressions[1:]:
            result = pc.and_(result, expr)
        
        return result
    
    def _record_query_stats(
        self, 
        plan: QueryPlan, 
        actual_time: float, 
        actual_rows: int
    ) -> None:
        """Record query execution statistics."""
        stats = {
            "timestamp": time.time(),
            "query_type": plan.query_type.value,
            "estimated_rows": plan.estimated_rows,
            "actual_rows": actual_rows,
            "estimated_time": plan.execution_time_estimate,
            "actual_time": actual_time,
            "partitions_scanned": len(plan.partitions_to_scan),
            "used_pushdown": plan.use_pushdown
        }
        
        self._query_history.append(stats)
        
        # Keep only recent history
        if len(self._query_history) > 1000:
            self._query_history = self._query_history[-500:]
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        if not self._query_history:
            return {"total_queries": 0}
        
        recent_queries = self._query_history[-100:]  # Last 100 queries
        
        return {
            "total_queries": len(self._query_history),
            "recent_queries": len(recent_queries),
            "avg_execution_time": sum(q["actual_time"] for q in recent_queries) / len(recent_queries),
            "query_type_distribution": {
                query_type: sum(1 for q in recent_queries if q["query_type"] == query_type)
                for query_type in QueryType.__members__.values()
            },
            "pushdown_usage": sum(1 for q in recent_queries if q["used_pushdown"]) / len(recent_queries),
            "estimation_accuracy": {
                "time_accuracy": 1.0 - (
                    sum(abs(q["actual_time"] - q["estimated_time"]) / max(q["estimated_time"], 0.001) 
                        for q in recent_queries) / len(recent_queries)
                ),
                "row_accuracy": 1.0 - (
                    sum(abs(q["actual_rows"] - q["estimated_rows"]) / max(q["estimated_rows"], 1)
                        for q in recent_queries) / len(recent_queries)
                )
            }
        }
    
    def _is_date_column(self, column: str) -> bool:
        """Check if column contains date data."""
        date_keywords = {"date", "time", "created", "updated", "timestamp"}
        return any(keyword in column.lower() for keyword in date_keywords)
    
    def _is_string_column(self, column: str) -> bool:
        """Check if column contains string data."""
        try:
            schema = self.get_schema()
            field = schema.field(column)
            return pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        except:
            # Default to assuming it's a string
            return column in {"symbol", "name", "description", "exchange", "currency"}
    
    def _is_numeric_column(self, column: str) -> bool:
        """Check if column contains numeric data."""
        try:
            schema = self.get_schema()
            field = schema.field(column)
            return (pa.types.is_integer(field.type) or 
                   pa.types.is_floating(field.type) or
                   pa.types.is_decimal(field.type))
        except:
            # Default based on common numeric column names
            numeric_keywords = {"price", "volume", "count", "amount", "size", "id"}
            return any(keyword in column.lower() for keyword in numeric_keywords)
    
    def _optimize_date_filter(self, column: str, operator: str, value: Any) -> Tuple[str, str, Any]:
        """Optimize date filter conditions."""
        # Convert various date formats to standard format
        if isinstance(value, str):
            try:
                # Try parsing different date formats
                if len(value) == 10 and "-" in value:  # YYYY-MM-DD
                    parsed_date = datetime.strptime(value, "%Y-%m-%d").date()
                    return (column, operator, parsed_date)
                elif len(value) == 19:  # YYYY-MM-DD HH:MM:SS
                    parsed_date = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    return (column, operator, parsed_date)
            except ValueError:
                pass
        
        return (column, operator, value)
    
    def _optimize_string_filter(self, column: str, operator: str, value: Any) -> Tuple[str, str, Any]:
        """Optimize string filter conditions."""
        if isinstance(value, str):
            # Uppercase symbol names for consistency
            if column.lower() in {"symbol", "exchange", "currency"}:
                return (column, operator, value.upper())
            
            # Strip whitespace
            return (column, operator, value.strip())
        
        return (column, operator, value)
    
    def _optimize_numeric_filter(self, column: str, operator: str, value: Any) -> Tuple[str, str, Any]:
        """Optimize numeric filter conditions."""
        # Convert to appropriate numeric type
        if isinstance(value, str) and value.replace(".", "").replace("-", "").isdigit():
            try:
                if "." in value:
                    return (column, operator, float(value))
                else:
                    return (column, operator, int(value))
            except ValueError:
                pass
        
        return (column, operator, value)
    
    def _sort_filters_by_selectivity(
        self, 
        filters: List[Tuple[str, str, Any]]
    ) -> List[Tuple[str, str, Any]]:
        """Sort filters by estimated selectivity (most selective first)."""
        def selectivity_score(filter_tuple):
            column, operator, value = filter_tuple
            
            # Equality filters are more selective
            if operator == "=":
                return 0.1
            elif operator in {"<", "<=", ">", ">="}:
                return 0.5
            elif operator == "!=":
                return 0.9
            else:
                return 0.7
        
        return sorted(filters, key=selectivity_score)
    
    def _calculate_partition_values(
        self, 
        partition_col: str, 
        filters: List[Tuple[str, str, Any]]
    ) -> List[str]:
        """Calculate partition values from filters."""
        values = []
        
        for column, operator, value in filters:
            if column == partition_col:
                if operator == "=":
                    values.append(str(value))
                elif operator == "in" and isinstance(value, (list, tuple)):
                    values.extend(str(v) for v in value)
        
        return values or ["*"]
    
    def _estimate_filter_selectivity(
        self, 
        column: str, 
        operator: str, 
        value: Any
    ) -> float:
        """Estimate filter selectivity (0.0 = very selective, 1.0 = not selective)."""
        # Simple selectivity estimates
        if operator == "=":
            return 0.01  # Equality is very selective
        elif operator in {"<", "<=", ">", ">="}:
            return 0.3   # Range filters are moderately selective
        elif operator == "!=":
            return 0.95  # Not equal is not very selective
        elif operator == "in":
            if isinstance(value, (list, tuple)):
                return min(0.8, len(value) * 0.01)
            return 0.1
        else:
            return 0.5   # Default moderate selectivity
    
    def _is_pushdown_supported(self, column: str, operator: str, value: Any) -> bool:
        """Check if filter can be pushed down to PyArrow."""
        # Most filters are supported, but there are some exceptions
        
        # Complex operations that might not be supported
        if operator in {"regex", "like", "contains"}:
            return False
        
        # Very large IN clauses might not be efficient
        if operator == "in" and isinstance(value, (list, tuple)) and len(value) > 1000:
            return False
        
        return True