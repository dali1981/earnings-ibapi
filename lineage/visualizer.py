"""
Data lineage visualization components.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .core import DataOperation, LineageNode, LineageTracker, OperationType
from .query import LineageQueryEngine

try:
    from reliability import get_trading_logger
except ImportError:
    import logging
    
    def get_trading_logger(name: str):
        return logging.getLogger(name)


class LineageVisualizer:
    """
    Visualization generator for data lineage graphs.
    
    Generates various representations of lineage data for analysis and reporting.
    """
    
    def __init__(self, tracker: LineageTracker):
        """
        Initialize visualizer.
        
        Args:
            tracker: LineageTracker instance to visualize
        """
        self.tracker = tracker
        self.query_engine = LineageQueryEngine(tracker)
        self.logger = get_trading_logger("LineageVisualizer")
    
    def generate_graph_data(self, node_id: Optional[str] = None, 
                           max_depth: Optional[int] = None,
                           include_operations: bool = True) -> Dict[str, Any]:
        """
        Generate graph data structure for visualization.
        
        Args:
            node_id: Root node for focused view (None for full graph)
            max_depth: Maximum depth to include
            include_operations: Whether to include operation nodes
            
        Returns:
            Graph data in standard format
        """
        if node_id:
            # Generate focused graph around specific node
            return self._generate_focused_graph(node_id, max_depth, include_operations)
        else:
            # Generate full graph
            return self._generate_full_graph(max_depth, include_operations)
    
    def _generate_focused_graph(self, node_id: str, max_depth: Optional[int],
                               include_operations: bool) -> Dict[str, Any]:
        """Generate graph focused on specific node."""
        root_node = self.tracker.get_node(node_id)
        if not root_node:
            raise ValueError(f"Node {node_id} not found")
        
        # Get related nodes
        upstream_ids = self.tracker.get_upstream_dependencies(node_id, max_depth)
        downstream_ids = self.tracker.get_downstream_dependents(node_id, max_depth)
        all_node_ids = {node_id} | upstream_ids | downstream_ids
        
        # Build nodes and edges
        nodes = []
        edges = []
        
        for nid in all_node_ids:
            node = self.tracker.get_node(nid)
            if node:
                nodes.append(self._create_node_data(node, is_root=(nid == node_id)))
                
                # Add edges for dependencies
                for dep_id in node.dependencies:
                    if dep_id in all_node_ids:
                        edges.append(self._create_edge_data(dep_id, nid, "dependency"))
        
        # Add operation nodes if requested
        if include_operations:
            self._add_operation_nodes(nodes, edges, all_node_ids)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'root_node_id': node_id,
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'max_depth': max_depth,
                'includes_operations': include_operations
            }
        }
    
    def _generate_full_graph(self, max_depth: Optional[int],
                           include_operations: bool) -> Dict[str, Any]:
        """Generate full lineage graph."""
        nodes = []
        edges = []
        
        # Add all data nodes
        for node in self.tracker.nodes.values():
            nodes.append(self._create_node_data(node))
            
            # Add edges for dependencies
            for dep_id in node.dependencies:
                edges.append(self._create_edge_data(dep_id, node.node_id, "dependency"))
        
        # Add operation nodes if requested
        if include_operations:
            all_node_ids = set(self.tracker.nodes.keys())
            self._add_operation_nodes(nodes, edges, all_node_ids)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'includes_operations': include_operations
            }
        }
    
    def _create_node_data(self, node: LineageNode, is_root: bool = False) -> Dict[str, Any]:
        """Create node data structure for visualization."""
        return {
            'id': node.node_id,
            'label': self._create_node_label(node),
            'type': 'data',
            'source_type': node.data_source.source_type,
            'location': node.data_source.location,
            'created_at': node.created_at.isoformat() if node.created_at else None,
            'last_accessed': node.last_accessed.isoformat() if node.last_accessed else None,
            'access_count': node.accessed_count,
            'dependencies_count': len(node.dependencies),
            'dependents_count': len(node.dependents),
            'is_root': is_root,
            'metadata': node.data_source.metadata or {}
        }
    
    def _create_operation_node_data(self, operation: DataOperation) -> Dict[str, Any]:
        """Create operation node data structure."""
        return {
            'id': f"op_{operation.operation_id}",
            'label': self._create_operation_label(operation),
            'type': 'operation',
            'operation_type': operation.operation_type.value,
            'timestamp': operation.timestamp.isoformat(),
            'duration_ms': operation.duration_ms,
            'record_count_in': operation.record_count_in,
            'record_count_out': operation.record_count_out,
            'has_error': bool(operation.error_info),
            'parameters': operation.parameters or {}
        }
    
    def _create_edge_data(self, from_id: str, to_id: str, edge_type: str,
                         operation: Optional[DataOperation] = None) -> Dict[str, Any]:
        """Create edge data structure."""
        edge = {
            'id': f"{from_id}_{to_id}",
            'source': from_id,
            'target': to_id,
            'type': edge_type,
        }
        
        if operation:
            edge.update({
                'operation_id': operation.operation_id,
                'operation_type': operation.operation_type.value,
                'timestamp': operation.timestamp.isoformat(),
                'record_count': operation.record_count_in or operation.record_count_out
            })
        
        return edge
    
    def _create_node_label(self, node: LineageNode) -> str:
        """Create display label for node."""
        source_id = node.data_source.source_id
        source_type = node.data_source.source_type
        
        # Shorten long source IDs
        if len(source_id) > 30:
            source_id = source_id[:27] + "..."
        
        return f"{source_type}:{source_id}"
    
    def _create_operation_label(self, operation: DataOperation) -> str:
        """Create display label for operation."""
        op_type = operation.operation_type.value
        timestamp = operation.timestamp.strftime("%H:%M:%S")
        
        if operation.record_count_in:
            return f"{op_type}\\n{timestamp}\\n{operation.record_count_in} records"
        else:
            return f"{op_type}\\n{timestamp}"
    
    def _add_operation_nodes(self, nodes: List[Dict], edges: List[Dict], 
                           relevant_node_ids: Set[str]):
        """Add operation nodes and their connections to the graph."""
        for operation in self.tracker.operations.values():
            # Check if operation is relevant to the current view
            op_node_ids = set()
            for source in operation.inputs + operation.outputs:
                related_node = self.tracker.find_node_by_source(
                    source.source_type, source.source_id
                )
                if related_node:
                    op_node_ids.add(related_node.node_id)
            
            if op_node_ids.intersection(relevant_node_ids):
                # Add operation node
                op_node = self._create_operation_node_data(operation)
                nodes.append(op_node)
                
                # Add edges from inputs to operation
                for source in operation.inputs:
                    input_node = self.tracker.find_node_by_source(
                        source.source_type, source.source_id
                    )
                    if input_node and input_node.node_id in relevant_node_ids:
                        edges.append(self._create_edge_data(
                            input_node.node_id, op_node['id'], "input", operation
                        ))
                
                # Add edges from operation to outputs
                for source in operation.outputs:
                    output_node = self.tracker.find_node_by_source(
                        source.source_type, source.source_id
                    )
                    if output_node and output_node.node_id in relevant_node_ids:
                        edges.append(self._create_edge_data(
                            op_node['id'], output_node.node_id, "output", operation
                        ))
    
    def generate_timeline_data(self, hours_back: float = 24.0) -> Dict[str, Any]:
        """
        Generate timeline data for operations.
        
        Args:
            hours_back: Hours back from now to include
            
        Returns:
            Timeline data structure
        """
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_operations = [
            op for op in self.tracker.operations.values()
            if op.timestamp >= cutoff_time
        ]
        
        # Sort by timestamp
        recent_operations.sort(key=lambda x: x.timestamp)
        
        timeline_events = []
        for operation in recent_operations:
            event = {
                'id': operation.operation_id,
                'timestamp': operation.timestamp.isoformat(),
                'type': operation.operation_type.value,
                'duration_ms': operation.duration_ms,
                'record_count_in': operation.record_count_in,
                'record_count_out': operation.record_count_out,
                'has_error': bool(operation.error_info),
                'input_sources': [
                    {
                        'type': source.source_type,
                        'id': source.source_id
                    }
                    for source in operation.inputs
                ],
                'output_sources': [
                    {
                        'type': source.source_type,
                        'id': source.source_id
                    }
                    for source in operation.outputs
                ]
            }
            timeline_events.append(event)
        
        return {
            'events': timeline_events,
            'time_range': {
                'start': cutoff_time.isoformat(),
                'end': datetime.utcnow().isoformat(),
                'duration_hours': hours_back
            },
            'summary': {
                'total_events': len(timeline_events),
                'operation_types': {
                    op_type.value: len([e for e in timeline_events if e['type'] == op_type.value])
                    for op_type in OperationType
                },
                'failed_operations': len([e for e in timeline_events if e['has_error']])
            }
        }
    
    def generate_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """
        Generate impact analysis for changing a specific node.
        
        Args:
            node_id: Node to analyze impact for
            
        Returns:
            Impact analysis data
        """
        node = self.tracker.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        # Get all downstream dependents
        downstream_ids = self.tracker.get_downstream_dependents(node_id)
        downstream_nodes = [
            self.tracker.get_node(nid) for nid in downstream_ids
        ]
        downstream_nodes = [n for n in downstream_nodes if n]  # Filter None
        
        # Categorize impact by source type
        impact_by_type = {}
        for dep_node in downstream_nodes:
            source_type = dep_node.data_source.source_type
            if source_type not in impact_by_type:
                impact_by_type[source_type] = []
            impact_by_type[source_type].append({
                'node_id': dep_node.node_id,
                'source_id': dep_node.data_source.source_id,
                'location': dep_node.data_source.location,
                'last_accessed': dep_node.last_accessed.isoformat() if dep_node.last_accessed else None
            })
        
        # Find operations that would be affected
        affected_operations = []
        for op in self.tracker.operations.values():
            for source in op.inputs:
                related_node = self.tracker.find_node_by_source(source.source_type, source.source_id)
                if related_node and related_node.node_id in downstream_ids:
                    affected_operations.append({
                        'operation_id': op.operation_id,
                        'operation_type': op.operation_type.value,
                        'timestamp': op.timestamp.isoformat()
                    })
                    break
        
        return {
            'target_node': {
                'node_id': node_id,
                'source_type': node.data_source.source_type,
                'source_id': node.data_source.source_id,
                'location': node.data_source.location
            },
            'total_downstream_nodes': len(downstream_nodes),
            'impact_by_source_type': impact_by_type,
            'affected_operations': affected_operations,
            'impact_summary': {
                'critical': len([n for n in downstream_nodes if n.accessed_count > 10]),
                'moderate': len([n for n in downstream_nodes if 1 < n.accessed_count <= 10]),
                'low': len([n for n in downstream_nodes if n.accessed_count <= 1])
            }
        }
    
    def export_to_dot(self, node_id: Optional[str] = None, max_depth: Optional[int] = None) -> str:
        """
        Export lineage graph to DOT format for Graphviz.
        
        Args:
            node_id: Root node for focused view
            max_depth: Maximum depth to include
            
        Returns:
            DOT format string
        """
        graph_data = self.generate_graph_data(node_id, max_depth, include_operations=False)
        
        dot_lines = ["digraph lineage {"]
        dot_lines.append("  rankdir=TB;")
        dot_lines.append("  node [shape=box, style=filled];")
        
        # Add nodes
        for node in graph_data['nodes']:
            node_id = node['id']
            label = node['label']
            source_type = node['source_type']
            
            # Color by source type
            color = {
                'repository': 'lightblue',
                'dataframe': 'lightgreen', 
                'file': 'lightyellow',
                'api': 'lightcoral'
            }.get(source_type, 'lightgray')
            
            dot_lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')
        
        # Add edges
        for edge in graph_data['edges']:
            source = edge['source']
            target = edge['target']
            dot_lines.append(f'  "{source}" -> "{target}";')
        
        dot_lines.append("}")
        
        return "\\n".join(dot_lines)
    
    def export_to_json(self, node_id: Optional[str] = None, 
                      max_depth: Optional[int] = None,
                      include_operations: bool = True,
                      pretty_print: bool = True) -> str:
        """
        Export lineage graph to JSON format.
        
        Args:
            node_id: Root node for focused view
            max_depth: Maximum depth to include
            include_operations: Whether to include operation nodes
            pretty_print: Whether to pretty-print the JSON
            
        Returns:
            JSON string
        """
        graph_data = self.generate_graph_data(node_id, max_depth, include_operations)
        
        if pretty_print:
            return json.dumps(graph_data, indent=2, default=str)
        else:
            return json.dumps(graph_data, default=str)
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary report of lineage.
        
        Returns:
            Text report
        """
        summary = self.query_engine.build_lineage_summary()
        
        report_lines = [
            "DATA LINEAGE SUMMARY REPORT",
            "=" * 50,
            "",
            "BASIC STATISTICS:",
            f"  Total Operations: {summary['basic_stats']['total_operations']}",
            f"  Total Nodes: {summary['basic_stats']['total_nodes']}",
            f"  Active Operations: {summary['basic_stats'].get('active_operations', 0)}",
            "",
            "GRAPH STRUCTURE:",
            f"  Nodes with Dependencies: {summary['graph_structure']['nodes_with_dependencies']}",
            f"  Nodes with Dependents: {summary['graph_structure']['nodes_with_dependents']}",
            f"  Root Nodes: {summary['graph_structure']['root_nodes_count']}",
            f"  Leaf Nodes: {summary['graph_structure']['leaf_nodes_count']}",
            f"  Avg Dependencies per Node: {summary['graph_structure']['average_dependencies_per_node']:.2f}",
            ""
        ]
        
        # Operation types
        if summary['basic_stats'].get('operation_types'):
            report_lines.append("OPERATIONS BY TYPE:")
            for op_type, count in summary['basic_stats']['operation_types'].items():
                if count > 0:
                    report_lines.append(f"  {op_type}: {count}")
            report_lines.append("")
        
        # Source types  
        if summary['basic_stats'].get('source_types'):
            report_lines.append("DATA SOURCES BY TYPE:")
            for source_type, count in summary['basic_stats']['source_types'].items():
                report_lines.append(f"  {source_type}: {count}")
            report_lines.append("")
        
        # Top connected nodes
        if summary['top_connected_nodes']['most_dependencies']:
            report_lines.append("MOST CONNECTED NODES (by dependencies):")
            for node_info in summary['top_connected_nodes']['most_dependencies'][:3]:
                report_lines.append(
                    f"  {node_info['node_id']}: {node_info['dependencies']} dependencies "
                    f"({node_info['source_type']})"
                )
            report_lines.append("")
        
        # Recent activity
        recent = summary['recent_activity']
        report_lines.extend([
            f"RECENT ACTIVITY (last {recent['time_window_hours']} hours):",
            f"  Operations: {recent['recent_operations_count']}",
            f"  Nodes Accessed: {recent['recent_nodes_accessed_count']}",
            ""
        ])
        
        report_lines.append(f"Report generated at: {datetime.utcnow().isoformat()}")
        
        return "\\n".join(report_lines)