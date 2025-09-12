"""Metrics functionality for repositories."""

import time
import logging
from pathlib import Path
from typing import Dict, Union, Optional, List
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsMixin:
    """Adds metrics and monitoring capabilities to repositories."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_enabled = kwargs.get('metrics_enabled', True)
        self._operation_counts = {}
        self._operation_times = {}
        self._last_access_time = None
        self._creation_time = datetime.now()
        
    def get_metrics(self) -> Dict[str, Union[int, float, str]]:
        """Get comprehensive repository metrics.
        
        Returns:
            Dictionary with various repository metrics
        """
        metrics = {
            'metrics_enabled': self.metrics_enabled,
            'creation_time': self._creation_time.isoformat(),
            'last_access_time': self._last_access_time.isoformat() if self._last_access_time else None,
            'uptime_seconds': (datetime.now() - self._creation_time).total_seconds()
        }
        
        # Add size and record count metrics
        try:
            metrics.update({
                'size_bytes': self.get_size_bytes(),
                'size_mb': round(self.get_size_bytes() / (1024 * 1024), 2),
                'record_count': self.get_record_count()
            })
        except Exception as e:
            logger.debug(f"Could not get basic metrics: {e}")
            metrics.update({
                'size_bytes': -1,
                'size_mb': -1,
                'record_count': -1
            })
        
        # Add operation metrics
        metrics.update({
            'total_operations': sum(self._operation_counts.values()),
            'operation_counts': self._operation_counts.copy(),
            'average_operation_times': self._calculate_average_times()
        })
        
        # Add partition metrics if partitioning is supported
        if hasattr(self, 'get_partition_statistics'):
            try:
                partition_stats = self.get_partition_statistics()
                metrics['partition_count'] = partition_stats.get('total_partitions', 0)
                metrics['partitioning_enabled'] = partition_stats.get('partitioning_enabled', False)
            except Exception as e:
                logger.debug(f"Could not get partition metrics: {e}")
        
        # Add cache metrics if caching is supported
        if hasattr(self, 'cache_info'):
            try:
                cache_stats = self.cache_info()
                metrics['cache_enabled'] = cache_stats.get('enabled', False)
                metrics['cache_hit_ratio'] = cache_stats.get('hit_ratio', 0)
                metrics['cache_entries'] = cache_stats.get('total_entries', 0)
            except Exception as e:
                logger.debug(f"Could not get cache metrics: {e}")
        
        return metrics
    
    def get_size_bytes(self) -> int:
        """Get repository size in bytes.
        
        Returns:
            Size of repository data in bytes
        """
        total_size = 0
        
        try:
            if hasattr(self, 'data_root'):
                data_path = Path(self.data_root)
                if data_path.exists():
                    for file_path in data_path.rglob('*.parquet'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            else:
                logger.warning("Repository does not have data_root attribute")
                return -1
                
        except Exception as e:
            logger.warning(f"Could not calculate repository size: {e}")
            return -1
        
        return total_size
    
    def get_record_count(self, **kwargs) -> int:
        """Get number of records in repository.
        
        Args:
            **kwargs: Filter criteria for counting
            
        Returns:
            Number of records matching criteria
        """
        try:
            # Use load method with filters to get data, then count
            data = self.load(filters=kwargs.get('filters'), **kwargs)
            return len(data)
        except Exception as e:
            logger.warning(f"Could not get record count: {e}")
            return -1
    
    def get_file_metrics(self) -> Dict[str, Union[int, List[Dict]]]:
        """Get detailed file-level metrics.
        
        Returns:
            Dictionary with file-level statistics
        """
        file_metrics = {
            'total_files': 0,
            'total_size_bytes': 0,
            'file_details': []
        }
        
        try:
            if not hasattr(self, 'data_root'):
                return file_metrics
            
            data_path = Path(self.data_root)
            if not data_path.exists():
                return file_metrics
            
            for file_path in data_path.rglob('*.parquet'):
                if file_path.is_file():
                    stat = file_path.stat()
                    file_info = {
                        'path': str(file_path.relative_to(data_path)),
                        'size_bytes': stat.st_size,
                        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat()
                    }
                    
                    # Try to get record count from parquet metadata
                    try:
                        import pyarrow.parquet as pq
                        parquet_file = pq.ParquetFile(file_path)
                        file_info['record_count'] = parquet_file.metadata.num_rows
                        file_info['column_count'] = parquet_file.metadata.num_columns
                    except Exception:
                        file_info['record_count'] = -1
                        file_info['column_count'] = -1
                    
                    file_metrics['file_details'].append(file_info)
                    file_metrics['total_size_bytes'] += stat.st_size
                    file_metrics['total_files'] += 1
                    
        except Exception as e:
            logger.warning(f"Could not get file metrics: {e}")
        
        return file_metrics
    
    def get_usage_statistics(self, days: int = 30) -> Dict[str, Union[int, float, List]]:
        """Get usage statistics for the specified period.
        
        Args:
            days: Number of days to look back for statistics
            
        Returns:
            Dictionary with usage statistics
        """
        # This is a simplified implementation
        # In practice, you might want to persist operation logs
        cutoff_time = datetime.now() - timedelta(days=days)
        
        stats = {
            'period_days': days,
            'operations_in_period': sum(self._operation_counts.values()),  # Simplified
            'most_used_operation': None,
            'least_used_operation': None,
            'operation_frequency': {}
        }
        
        if self._operation_counts:
            sorted_operations = sorted(
                self._operation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            stats['most_used_operation'] = sorted_operations[0][0]
            stats['least_used_operation'] = sorted_operations[-1][0]
            
            # Calculate operations per day (simplified)
            total_days = (datetime.now() - self._creation_time).days or 1
            for operation, count in self._operation_counts.items():
                stats['operation_frequency'][operation] = round(count / total_days, 2)
        
        return stats
    
    def get_performance_metrics(self) -> Dict[str, Union[float, Dict]]:
        """Get performance-related metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'average_operation_times': self._calculate_average_times(),
            'slowest_operation': None,
            'fastest_operation': None
        }
        
        avg_times = metrics['average_operation_times']
        if avg_times:
            sorted_by_time = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
            metrics['slowest_operation'] = {
                'operation': sorted_by_time[0][0],
                'average_time_seconds': sorted_by_time[0][1]
            }
            metrics['fastest_operation'] = {
                'operation': sorted_by_time[-1][0],
                'average_time_seconds': sorted_by_time[-1][1]
            }
        
        return metrics
    
    def export_metrics(self, format: str = 'json') -> Union[str, Dict]:
        """Export metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'prometheus')
            
        Returns:
            Metrics in requested format
        """
        metrics = self.get_metrics()
        
        if format.lower() == 'json':
            import json
            return json.dumps(metrics, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Flatten metrics for CSV
            flattened = self._flatten_metrics(metrics)
            df = pd.DataFrame([flattened])
            return df.to_csv(index=False)
        
        elif format.lower() == 'prometheus':
            return self._format_prometheus_metrics(metrics)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self._operation_counts.clear()
        self._operation_times.clear()
        self._last_access_time = None
        self._creation_time = datetime.now()
        logger.info("Repository metrics reset")
    
    def _track_operation(self, operation_name: str, execution_time: float = None):
        """Track an operation for metrics collection.
        
        Args:
            operation_name: Name of the operation
            execution_time: Time taken to execute (seconds)
        """
        if not self.metrics_enabled:
            return
        
        self._last_access_time = datetime.now()
        
        # Update operation count
        self._operation_counts[operation_name] = (
            self._operation_counts.get(operation_name, 0) + 1
        )
        
        # Update operation times
        if execution_time is not None:
            if operation_name not in self._operation_times:
                self._operation_times[operation_name] = []
            self._operation_times[operation_name].append(execution_time)
            
            # Keep only recent times to prevent memory growth
            if len(self._operation_times[operation_name]) > 1000:
                self._operation_times[operation_name] = (
                    self._operation_times[operation_name][-500:]  # Keep last 500
                )
    
    def _calculate_average_times(self) -> Dict[str, float]:
        """Calculate average execution times for operations."""
        averages = {}
        for operation, times in self._operation_times.items():
            if times:
                averages[operation] = round(sum(times) / len(times), 4)
        return averages
    
    def _flatten_metrics(self, metrics: Dict) -> Dict[str, Union[str, int, float]]:
        """Flatten nested metrics dictionary for CSV export."""
        flattened = {}
        
        def flatten_recursive(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    flatten_recursive(value, new_key)
            elif isinstance(obj, list):
                flattened[prefix] = str(obj)  # Convert lists to strings
            else:
                flattened[prefix] = obj
        
        flatten_recursive(metrics)
        return flattened
    
    def _format_prometheus_metrics(self, metrics: Dict) -> str:
        """Format metrics in Prometheus format."""
        lines = []
        
        # Basic metrics
        if 'record_count' in metrics and metrics['record_count'] >= 0:
            lines.append(f"repository_record_count {metrics['record_count']}")
        
        if 'size_bytes' in metrics and metrics['size_bytes'] >= 0:
            lines.append(f"repository_size_bytes {metrics['size_bytes']}")
        
        if 'uptime_seconds' in metrics:
            lines.append(f"repository_uptime_seconds {metrics['uptime_seconds']}")
        
        # Operation counts
        for operation, count in metrics.get('operation_counts', {}).items():
            lines.append(f'repository_operation_total{{operation="{operation}"}} {count}')
        
        # Average operation times
        for operation, avg_time in metrics.get('average_operation_times', {}).items():
            lines.append(f'repository_operation_duration_seconds{{operation="{operation}"}} {avg_time}')
        
        return '\n'.join(lines)
    
    def time_operation(self, operation_name: str):
        """Decorator to time operations for metrics.
        
        Args:
            operation_name: Name of the operation being timed
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self._track_operation(operation_name, execution_time)
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self._track_operation(f"{operation_name}_error", execution_time)
                    raise
            return wrapper
        return decorator
    
    def set_metrics_enabled(self, enabled: bool) -> None:
        """Enable or disable metrics collection.
        
        Args:
            enabled: True to enable metrics, False to disable
        """
        self.metrics_enabled = enabled
        if not enabled:
            self.reset_metrics()
        logger.info(f"Repository metrics {'enabled' if enabled else 'disabled'}")