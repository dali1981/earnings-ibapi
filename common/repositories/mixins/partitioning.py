"""Partitioning functionality for repositories."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from datetime import date, datetime

from ...constants import DEFAULT_PARTITION_COLS

logger = logging.getLogger(__name__)


class PartitioningMixin:
    """Adds partitioning capabilities to repositories for better performance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_columns = kwargs.get('partition_columns', DEFAULT_PARTITION_COLS.copy())
        self.partitioning_enabled = True
        
    def get_partitions(self) -> List[Dict[str, Any]]:
        """Get information about repository partitions.
        
        Returns:
            List of partition metadata dictionaries
        """
        if not self.partitioning_enabled or not hasattr(self, 'data_root'):
            return []
        
        partitions = []
        
        try:
            # Use PyArrow dataset to discover partitions
            dataset_path = Path(self.data_root)
            if dataset_path.exists():
                dataset = ds.dataset(str(dataset_path), format="parquet")
                
                for fragment in dataset.get_fragments():
                    partition_expression = fragment.partition_expression
                    partition_info = self._extract_partition_info(partition_expression)
                    
                    if partition_info:
                        partition_info['path'] = str(fragment.path)
                        partition_info['format'] = fragment.format.type_name
                        partitions.append(partition_info)
                        
        except Exception as e:
            logger.warning(f"Failed to get partition information: {e}")
        
        return partitions
    
    def get_partition_columns(self) -> List[str]:
        """Get the columns used for partitioning.
        
        Returns:
            List of column names used for partitioning
        """
        return self.partition_columns.copy()
    
    def load_partition(
        self, 
        partition_filter: Dict[str, Any], 
        **kwargs
    ) -> pd.DataFrame:
        """Load data from a specific partition.
        
        Args:
            partition_filter: Dictionary specifying partition values
            **kwargs: Additional load parameters
            
        Returns:
            DataFrame containing partition data
        """
        if not self.partitioning_enabled:
            logger.warning("Partitioning is disabled, falling back to regular load")
            return self.load(**kwargs)
        
        # Convert partition filter to PyArrow filters
        filters = []
        for column, value in partition_filter.items():
            if column in self.partition_columns:
                filters.append((column, '=', value))
            else:
                logger.warning(f"Column {column} is not a partition column")
        
        # Add additional filters from kwargs
        additional_filters = kwargs.get('filters', [])
        filters.extend(additional_filters)
        
        # Load with the combined filters
        kwargs['filters'] = filters
        return self.load(**kwargs)
    
    def save_partitioned(
        self,
        data: pd.DataFrame,
        partition_cols: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Save data with partitioning.
        
        Args:
            data: DataFrame to save
            partition_cols: Columns to partition by (uses default if None)
            **kwargs: Additional save parameters
        """
        if not self.partitioning_enabled:
            logger.warning("Partitioning is disabled, using regular save")
            return self.save(data, **kwargs)
        
        effective_partition_cols = partition_cols or self.partition_columns
        
        # Validate partition columns exist in data
        missing_cols = set(effective_partition_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Partition columns {missing_cols} not found in data, using regular save")
            return self.save(data, **kwargs)
        
        # Prepare data for partitioned save
        prepared_data = self._prepare_data_for_partitioning(data, effective_partition_cols)
        
        # Save with partitioning
        if hasattr(self, '_save_partitioned_impl'):
            return self._save_partitioned_impl(prepared_data, effective_partition_cols, **kwargs)
        else:
            # Fallback to regular save if no partitioned implementation
            logger.warning("No partitioned save implementation found, using regular save")
            return self.save(prepared_data, **kwargs)
    
    def get_partition_statistics(self) -> Dict[str, Any]:
        """Get statistics about partitions.
        
        Returns:
            Dictionary containing partition statistics
        """
        stats = {
            'partitioning_enabled': self.partitioning_enabled,
            'partition_columns': self.partition_columns,
            'total_partitions': 0,
            'partition_sizes': {},
            'partition_record_counts': {}
        }
        
        try:
            partitions = self.get_partitions()
            stats['total_partitions'] = len(partitions)
            
            for partition in partitions:
                partition_key = self._partition_key_from_info(partition)
                
                # Try to get file size and record count
                try:
                    file_path = Path(partition['path'])
                    if file_path.exists():
                        stats['partition_sizes'][partition_key] = file_path.stat().st_size
                        
                        # Read parquet metadata for record count
                        import pyarrow.parquet as pq
                        parquet_file = pq.ParquetFile(file_path)
                        stats['partition_record_counts'][partition_key] = parquet_file.metadata.num_rows
                        
                except Exception as e:
                    logger.debug(f"Could not get stats for partition {partition_key}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to get partition statistics: {e}")
        
        return stats
    
    def optimize_partitions(self, target_size_mb: int = 100) -> Dict[str, Any]:
        """Analyze and suggest partition optimizations.
        
        Args:
            target_size_mb: Target partition size in MB
            
        Returns:
            Dictionary with optimization recommendations
        """
        stats = self.get_partition_statistics()
        recommendations = {
            'current_partitions': stats['total_partitions'],
            'target_size_bytes': target_size_mb * 1024 * 1024,
            'oversized_partitions': [],
            'undersized_partitions': [],
            'recommendations': []
        }
        
        target_size_bytes = recommendations['target_size_bytes']
        
        for partition_key, size_bytes in stats['partition_sizes'].items():
            if size_bytes > target_size_bytes * 2:  # More than 2x target
                recommendations['oversized_partitions'].append({
                    'partition': partition_key,
                    'size_mb': size_bytes / (1024 * 1024),
                    'records': stats['partition_record_counts'].get(partition_key, 'unknown')
                })
            elif size_bytes < target_size_bytes * 0.1:  # Less than 10% of target
                recommendations['undersized_partitions'].append({
                    'partition': partition_key,
                    'size_mb': size_bytes / (1024 * 1024),
                    'records': stats['partition_record_counts'].get(partition_key, 'unknown')
                })
        
        # Generate recommendations
        if recommendations['oversized_partitions']:
            recommendations['recommendations'].append(
                "Consider adding more granular partitioning (e.g., by month instead of year) "
                "for oversized partitions"
            )
        
        if recommendations['undersized_partitions']:
            recommendations['recommendations'].append(
                "Consider consolidating undersized partitions or using less granular partitioning"
            )
        
        if not recommendations['oversized_partitions'] and not recommendations['undersized_partitions']:
            recommendations['recommendations'].append("Partition sizes appear well-balanced")
        
        return recommendations
    
    def _extract_partition_info(self, partition_expression) -> Optional[Dict[str, Any]]:
        """Extract partition information from PyArrow partition expression."""
        if partition_expression is None:
            return None
        
        # This is a simplified implementation
        # In practice, you'd need to parse the expression more thoroughly
        try:
            partition_info = {}
            # Add logic to extract partition column values from expression
            # This depends on the specific PyArrow version and expression format
            return partition_info
        except Exception as e:
            logger.debug(f"Could not extract partition info: {e}")
            return None
    
    def _prepare_data_for_partitioning(
        self,
        data: pd.DataFrame,
        partition_cols: List[str]
    ) -> pd.DataFrame:
        """Prepare data for partitioned saving."""
        prepared_data = data.copy()
        
        # Ensure partition columns are in the correct format
        for col in partition_cols:
            if col in prepared_data.columns:
                # Convert date columns to string format for partitioning
                if col == 'date' or 'date' in col.lower():
                    if pd.api.types.is_datetime64_any_dtype(prepared_data[col]):
                        prepared_data[col] = prepared_data[col].dt.strftime('%Y-%m-%d')
                    elif isinstance(prepared_data[col].iloc[0], date):
                        prepared_data[col] = prepared_data[col].astype(str)
                
                # Ensure string columns are clean
                elif prepared_data[col].dtype == object:
                    prepared_data[col] = prepared_data[col].astype(str)
        
        return prepared_data
    
    def _partition_key_from_info(self, partition_info: Dict[str, Any]) -> str:
        """Generate a partition key string from partition info."""
        # Create a readable key from partition values
        key_parts = []
        for col in self.partition_columns:
            if col in partition_info:
                key_parts.append(f"{col}={partition_info[col]}")
        return "/".join(key_parts) if key_parts else "unknown_partition"
    
    def set_partitioning_enabled(self, enabled: bool) -> None:
        """Enable or disable partitioning.
        
        Args:
            enabled: True to enable partitioning, False to disable
        """
        self.partitioning_enabled = enabled
        logger.info(f"Partitioning {'enabled' if enabled else 'disabled'}")
    
    def set_partition_columns(self, columns: List[str]) -> None:
        """Set the columns to use for partitioning.
        
        Args:
            columns: List of column names to partition by
        """
        self.partition_columns = columns.copy()
        logger.info(f"Partition columns set to: {self.partition_columns}")
    
    def add_partition_column(self, column: str) -> None:
        """Add a column to the partitioning scheme.
        
        Args:
            column: Column name to add to partitioning
        """
        if column not in self.partition_columns:
            self.partition_columns.append(column)
            logger.info(f"Added {column} to partition columns: {self.partition_columns}")
    
    def remove_partition_column(self, column: str) -> None:
        """Remove a column from the partitioning scheme.
        
        Args:
            column: Column name to remove from partitioning
        """
        if column in self.partition_columns:
            self.partition_columns.remove(column)
            logger.info(f"Removed {column} from partition columns: {self.partition_columns}")