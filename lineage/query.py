"""
Lineage query engine for analyzing data lineage relationships.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from .core import DataOperation, LineageNode, LineageTracker, OperationType

try:
    from reliability import get_trading_logger
except ImportError:
    import logging
    
    def get_trading_logger(name: str):
        return logging.getLogger(name)


class LineageQueryEngine:
    """
    Query engine for data lineage analysis.
    
    Provides high-level query methods for analyzing data flow,
    dependencies, and impact analysis.
    """
    
    def __init__(self, tracker: LineageTracker):
        """
        Initialize query engine.
        
        Args:
            tracker: LineageTracker instance to query
        """
        self.tracker = tracker
        self.logger = get_trading_logger("LineageQueryEngine")
    
    def find_data_sources(self, source_type: Optional[str] = None, 
                         location_pattern: Optional[str] = None) -> List[LineageNode]:
        """
        Find data sources matching criteria.
        
        Args:
            source_type: Filter by source type (e.g., "repository", "file")
            location_pattern: Filter by location pattern (substring match)
            
        Returns:
            List of matching nodes
        """
        matching_nodes = []
        
        for node in self.tracker.nodes.values():
            # Apply source type filter
            if source_type and node.data_source.source_type != source_type:
                continue
            
            # Apply location pattern filter
            if location_pattern and location_pattern not in node.data_source.location:
                continue
            
            matching_nodes.append(node)
        
        return matching_nodes
    
    def get_data_lineage(self, node_id: str, direction: str = "both", 
                        max_depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Get complete data lineage for a node.
        
        Args:
            node_id: Node to analyze
            direction: "upstream", "downstream", or "both"
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary containing lineage information
        """
        node = self.tracker.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        lineage = {
            'root_node': node,
            'upstream_dependencies': [],
            'downstream_dependents': [],
            'related_operations': []
        }
        
        if direction in ["upstream", "both"]:
            upstream_ids = self.tracker.get_upstream_dependencies(node_id, max_depth)
            lineage['upstream_dependencies'] = [
                self.tracker.get_node(nid) for nid in upstream_ids
                if self.tracker.get_node(nid)
            ]
        
        if direction in ["downstream", "both"]:
            downstream_ids = self.tracker.get_downstream_dependents(node_id, max_depth)
            lineage['downstream_dependents'] = [
                self.tracker.get_node(nid) for nid in downstream_ids
                if self.tracker.get_node(nid)
            ]
        
        # Find related operations
        all_related_nodes = set([node_id])
        if direction in ["upstream", "both"]:
            all_related_nodes.update(upstream_ids)
        if direction in ["downstream", "both"]:
            all_related_nodes.update(downstream_ids)
        
        related_operations = []
        for op in self.tracker.operations.values():
            # Check if operation involves any of the related nodes
            op_node_ids = set()
            for source in op.inputs + op.outputs:
                related_node = self.tracker.find_node_by_source(
                    source.source_type, source.source_id
                )
                if related_node:
                    op_node_ids.add(related_node.node_id)
            
            if op_node_ids.intersection(all_related_nodes):
                related_operations.append(op)
        
        lineage['related_operations'] = related_operations
        
        return lineage
    
    def trace_data_flow(self, source_node_id: str, target_node_id: str) -> Dict[str, Any]:
        """
        Trace data flow between two nodes.
        
        Args:
            source_node_id: Starting node
            target_node_id: Target node
            
        Returns:
            Dictionary with flow path and operations
        """
        path = self.tracker.get_lineage_path(source_node_id, target_node_id)
        
        if not path:
            return {
                'path_found': False,
                'message': f"No path found from {source_node_id} to {target_node_id}"
            }
        
        # Get nodes along the path
        path_nodes = [self.tracker.get_node(nid) for nid in path]
        path_nodes = [n for n in path_nodes if n]  # Filter out None values
        
        # Find operations that created the transformations
        transformation_operations = []
        for i in range(1, len(path)):
            from_node_id = path[i-1]
            to_node_id = path[i]
            to_node = self.tracker.get_node(to_node_id)
            
            if to_node and to_node.created_by:
                op = self.tracker.get_operation(to_node.created_by)
                if op:
                    transformation_operations.append({
                        'operation': op,
                        'from_node': from_node_id,
                        'to_node': to_node_id
                    })
        
        return {
            'path_found': True,
            'path_length': len(path),
            'path_nodes': path_nodes,
            'transformation_operations': transformation_operations,
            'total_transformations': len(transformation_operations)
        }
    
    def analyze_data_freshness(self, node_id: str) -> Dict[str, Any]:
        """
        Analyze data freshness for a node and its dependencies.
        
        Args:
            node_id: Node to analyze
            
        Returns:
            Freshness analysis results
        """
        node = self.tracker.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        analysis = {
            'node_id': node_id,
            'last_accessed': node.last_accessed,
            'created_at': node.created_at,
            'access_count': node.accessed_count,
            'upstream_freshness': []
        }
        
        # Calculate time since last access
        if node.last_accessed:
            analysis['hours_since_last_access'] = (
                datetime.utcnow() - node.last_accessed
            ).total_seconds() / 3600
        
        # Calculate time since creation
        if node.created_at:
            analysis['hours_since_creation'] = (
                datetime.utcnow() - node.created_at
            ).total_seconds() / 3600
        
        # Analyze upstream dependencies
        upstream_ids = self.tracker.get_upstream_dependencies(node_id, max_depth=2)
        for upstream_id in upstream_ids:
            upstream_node = self.tracker.get_node(upstream_id)
            if upstream_node:
                upstream_info = {
                    'node_id': upstream_id,
                    'last_accessed': upstream_node.last_accessed,
                    'created_at': upstream_node.created_at,
                    'source_type': upstream_node.data_source.source_type
                }
                
                if upstream_node.last_accessed:
                    upstream_info['hours_since_last_access'] = (
                        datetime.utcnow() - upstream_node.last_accessed
                    ).total_seconds() / 3600
                
                analysis['upstream_freshness'].append(upstream_info)
        
        return analysis
    
    def find_stale_data(self, hours_threshold: float = 24.0) -> List[Dict[str, Any]]:
        """
        Find data sources that haven't been accessed recently.
        
        Args:
            hours_threshold: Hours threshold for considering data stale
            
        Returns:
            List of stale data sources
        """
        threshold_time = datetime.utcnow() - timedelta(hours=hours_threshold)
        stale_data = []
        
        for node in self.tracker.nodes.values():
            is_stale = False
            reason = ""
            
            if not node.last_accessed:
                is_stale = True
                reason = "Never accessed"
            elif node.last_accessed < threshold_time:
                is_stale = True
                hours_since = (datetime.utcnow() - node.last_accessed).total_seconds() / 3600
                reason = f"Last accessed {hours_since:.1f} hours ago"
            
            if is_stale:
                stale_data.append({
                    'node': node,
                    'reason': reason,
                    'last_accessed': node.last_accessed,
                    'access_count': node.accessed_count,
                    'source_type': node.data_source.source_type,
                    'location': node.data_source.location
                })
        
        return sorted(stale_data, key=lambda x: x['last_accessed'] or datetime.min)
    
    def analyze_data_volume_flow(self, time_window_hours: Optional[float] = 24.0) -> Dict[str, Any]:
        """
        Analyze data volume flow over time.
        
        Args:
            time_window_hours: Time window to analyze (None for all operations)
            
        Returns:
            Volume flow analysis
        """
        cutoff_time = None
        if time_window_hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        volume_stats = {
            'total_operations': 0,
            'total_records_in': 0,
            'total_records_out': 0,
            'operations_by_type': {},
            'top_volume_operations': [],
            'repository_volumes': {}
        }
        
        operations_to_analyze = []
        for op in self.tracker.operations.values():
            if cutoff_time is None or op.timestamp >= cutoff_time:
                operations_to_analyze.append(op)
        
        volume_stats['total_operations'] = len(operations_to_analyze)
        
        for op in operations_to_analyze:
            # Count by operation type
            op_type = op.operation_type.value
            if op_type not in volume_stats['operations_by_type']:
                volume_stats['operations_by_type'][op_type] = {
                    'count': 0, 'total_records_in': 0, 'total_records_out': 0
                }
            
            volume_stats['operations_by_type'][op_type]['count'] += 1
            
            # Aggregate record counts
            if op.record_count_in:
                volume_stats['total_records_in'] += op.record_count_in
                volume_stats['operations_by_type'][op_type]['total_records_in'] += op.record_count_in
            
            if op.record_count_out:
                volume_stats['total_records_out'] += op.record_count_out
                volume_stats['operations_by_type'][op_type]['total_records_out'] += op.record_count_out
            
            # Track repository volumes
            for output_source in op.outputs:
                if output_source.source_type == "repository":
                    repo_name = output_source.source_id
                    if repo_name not in volume_stats['repository_volumes']:
                        volume_stats['repository_volumes'][repo_name] = 0
                    if op.record_count_in:
                        volume_stats['repository_volumes'][repo_name] += op.record_count_in
        
        # Find top volume operations
        volume_ops = [
            (op, op.record_count_in or 0) for op in operations_to_analyze
            if op.record_count_in
        ]
        volume_ops.sort(key=lambda x: x[1], reverse=True)
        volume_stats['top_volume_operations'] = volume_ops[:10]
        
        return volume_stats
    
    def get_operation_performance_stats(self, operation_type: Optional[OperationType] = None) -> Dict[str, Any]:
        """
        Get performance statistics for operations.
        
        Args:
            operation_type: Filter by specific operation type
            
        Returns:
            Performance statistics
        """
        operations_to_analyze = []
        for op in self.tracker.operations.values():
            if operation_type is None or op.operation_type == operation_type:
                operations_to_analyze.append(op)
        
        if not operations_to_analyze:
            return {'message': 'No operations found matching criteria'}
        
        # Calculate performance metrics
        durations = [op.duration_ms for op in operations_to_analyze if op.duration_ms]
        
        stats = {
            'total_operations': len(operations_to_analyze),
            'operations_with_timing': len(durations),
            'failed_operations': len([op for op in operations_to_analyze if op.error_info])
        }
        
        if durations:
            stats.update({
                'avg_duration_ms': sum(durations) / len(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'total_duration_ms': sum(durations)
            })
            
            # Percentiles
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            stats.update({
                'p50_duration_ms': sorted_durations[n//2],
                'p90_duration_ms': sorted_durations[int(n*0.9)] if n > 10 else sorted_durations[-1],
                'p95_duration_ms': sorted_durations[int(n*0.95)] if n > 20 else sorted_durations[-1]
            })
        
        return stats
    
    def build_lineage_summary(self) -> Dict[str, Any]:
        """
        Build comprehensive lineage summary.
        
        Returns:
            Summary of entire lineage graph
        """
        # Get basic statistics
        stats = self.tracker.get_statistics()
        
        # Analyze node relationships
        nodes_with_dependencies = len([n for n in self.tracker.nodes.values() if n.dependencies])
        nodes_with_dependents = len([n for n in self.tracker.nodes.values() if n.dependents])
        
        # Find root nodes (no dependencies) and leaf nodes (no dependents)
        root_nodes = [n for n in self.tracker.nodes.values() if not n.dependencies]
        leaf_nodes = [n for n in self.tracker.nodes.values() if not n.dependents]
        
        # Calculate average dependency depth
        total_deps = sum(len(n.dependencies) for n in self.tracker.nodes.values())
        avg_dependencies = total_deps / len(self.tracker.nodes) if self.tracker.nodes else 0
        
        # Find most connected nodes
        most_dependencies = sorted(
            self.tracker.nodes.values(), 
            key=lambda n: len(n.dependencies), 
            reverse=True
        )[:5]
        
        most_dependents = sorted(
            self.tracker.nodes.values(),
            key=lambda n: len(n.dependents),
            reverse=True
        )[:5]
        
        summary = {
            'basic_stats': stats,
            'graph_structure': {
                'nodes_with_dependencies': nodes_with_dependencies,
                'nodes_with_dependents': nodes_with_dependents,
                'root_nodes_count': len(root_nodes),
                'leaf_nodes_count': len(leaf_nodes),
                'average_dependencies_per_node': avg_dependencies
            },
            'top_connected_nodes': {
                'most_dependencies': [
                    {'node_id': n.node_id, 'dependencies': len(n.dependencies), 
                     'source_type': n.data_source.source_type}
                    for n in most_dependencies
                ],
                'most_dependents': [
                    {'node_id': n.node_id, 'dependents': len(n.dependents),
                     'source_type': n.data_source.source_type}
                    for n in most_dependents
                ]
            },
            'recent_activity': self.get_recent_activity_summary()
        }
        
        return summary
    
    def get_recent_activity_summary(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get summary of recent lineage activity."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_operations = [
            op for op in self.tracker.operations.values()
            if op.timestamp >= cutoff_time
        ]
        
        recent_nodes_accessed = [
            node for node in self.tracker.nodes.values()
            if node.last_accessed and node.last_accessed >= cutoff_time
        ]
        
        return {
            'time_window_hours': hours,
            'recent_operations_count': len(recent_operations),
            'recent_nodes_accessed_count': len(recent_nodes_accessed),
            'operation_types': {
                op_type.value: len([op for op in recent_operations if op.operation_type == op_type])
                for op_type in OperationType
            }
        }