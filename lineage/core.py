"""
Core data lineage tracking components.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

try:
    from reliability import get_trading_logger, log_context
except ImportError:
    import logging
    from contextlib import contextmanager
    
    def get_trading_logger(name: str):
        return logging.getLogger(name)
    
    @contextmanager
    def log_context(**kwargs):
        yield


class OperationType(Enum):
    """Types of data operations for lineage tracking."""
    READ = "read"
    WRITE = "write"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    JOIN = "join"
    DERIVE = "derive"


@dataclass
class DataSource:
    """Represents a data source in lineage tracking."""
    source_type: str  # "repository", "file", "api", "calculation"
    source_id: str    # unique identifier
    location: str     # path, table name, etc.
    schema_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class DataOperation:
    """Represents a single data operation in the lineage graph."""
    operation_id: str
    operation_type: OperationType
    timestamp: datetime
    inputs: List[DataSource] = field(default_factory=list)
    outputs: List[DataSource] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    user_info: Optional[str] = None
    duration_ms: Optional[float] = None
    record_count_in: Optional[int] = None
    record_count_out: Optional[int] = None
    error_info: Optional[str] = None
    
    @classmethod
    def create(cls, operation_type: OperationType, **kwargs) -> 'DataOperation':
        """Create a new data operation with generated ID."""
        return cls(
            operation_id=str(uuid.uuid4()),
            operation_type=operation_type,
            timestamp=datetime.utcnow(),
            **kwargs
        )


@dataclass
class LineageNode:
    """Represents a node in the data lineage graph."""
    node_id: str
    data_source: DataSource
    created_by: Optional[str] = None  # operation_id that created this data
    created_at: Optional[datetime] = None
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)  # node_ids this depends on
    dependents: Set[str] = field(default_factory=set)    # node_ids that depend on this
    
    def add_dependency(self, node_id: str):
        """Add a dependency relationship."""
        self.dependencies.add(node_id)
    
    def add_dependent(self, node_id: str):
        """Add a dependent relationship."""
        self.dependents.add(node_id)
    
    def record_access(self):
        """Record that this node was accessed."""
        self.accessed_count += 1
        self.last_accessed = datetime.utcnow()


class LineageTracker:
    """
    Central lineage tracking system.
    
    Tracks data operations, maintains lineage graph, and provides
    querying capabilities for data lineage analysis.
    """
    
    def __init__(self, storage_backend=None):
        """
        Initialize lineage tracker.
        
        Args:
            storage_backend: Optional storage backend for persistence
        """
        self.logger = get_trading_logger("LineageTracker")
        self.storage = storage_backend
        
        # In-memory lineage graph
        self.operations: Dict[str, DataOperation] = {}
        self.nodes: Dict[str, LineageNode] = {}
        self.source_to_node: Dict[str, str] = {}  # source_id -> node_id mapping
        
        # Active operation tracking
        self._active_operations: Dict[str, DataOperation] = {}
        
    def start_operation(self, operation_type: OperationType, 
                       parameters: Optional[Dict[str, Any]] = None,
                       execution_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking a new operation.
        
        Args:
            operation_type: Type of operation
            parameters: Operation parameters
            execution_context: Additional context information
            
        Returns:
            Operation ID for tracking
        """
        operation = DataOperation.create(
            operation_type=operation_type,
            parameters=parameters or {},
            execution_context=execution_context or {}
        )
        
        self._active_operations[operation.operation_id] = operation
        
        self.logger.debug(
            f"Started operation {operation.operation_id} of type {operation_type.value}"
        )
        
        return operation.operation_id
    
    def add_input(self, operation_id: str, data_source: DataSource):
        """Add input data source to an active operation."""
        if operation_id not in self._active_operations:
            raise ValueError(f"Operation {operation_id} is not active")
        
        operation = self._active_operations[operation_id]
        operation.inputs.append(data_source)
        
        # Create or get node for this data source
        node = self._get_or_create_node(data_source)
        node.record_access()
        
        self.logger.debug(
            f"Added input {data_source.source_id} to operation {operation_id}"
        )
    
    def add_output(self, operation_id: str, data_source: DataSource):
        """Add output data source to an active operation."""
        if operation_id not in self._active_operations:
            raise ValueError(f"Operation {operation_id} is not active")
        
        operation = self._active_operations[operation_id]
        operation.outputs.append(data_source)
        
        # Create node for output data source
        node = self._get_or_create_node(data_source)
        node.created_by = operation_id
        node.created_at = datetime.utcnow()
        
        self.logger.debug(
            f"Added output {data_source.source_id} to operation {operation_id}"
        )
    
    def complete_operation(self, operation_id: str, 
                          record_count_in: Optional[int] = None,
                          record_count_out: Optional[int] = None,
                          error_info: Optional[str] = None) -> DataOperation:
        """
        Complete an active operation and update lineage graph.
        
        Args:
            operation_id: ID of operation to complete
            record_count_in: Number of input records processed
            record_count_out: Number of output records produced
            error_info: Error information if operation failed
            
        Returns:
            Completed DataOperation
        """
        if operation_id not in self._active_operations:
            raise ValueError(f"Operation {operation_id} is not active")
        
        operation = self._active_operations.pop(operation_id)
        
        # Calculate duration
        operation.duration_ms = (datetime.utcnow() - operation.timestamp).total_seconds() * 1000
        operation.record_count_in = record_count_in
        operation.record_count_out = record_count_out
        operation.error_info = error_info
        
        # Update lineage graph
        self._update_lineage_graph(operation)
        
        # Store completed operation
        self.operations[operation_id] = operation
        
        # Persist if storage backend available
        if self.storage:
            try:
                self.storage.save_operation(operation)
            except Exception as e:
                self.logger.error(f"Failed to persist operation {operation_id}: {e}")
        
        status = "failed" if error_info else "completed"
        self.logger.info(
            f"Operation {operation_id} {status} in {operation.duration_ms:.1f}ms",
            extra={
                'operation_id': operation_id,
                'operation_type': operation.operation_type.value,
                'duration_ms': operation.duration_ms,
                'record_count_in': record_count_in,
                'record_count_out': record_count_out,
                'status': status
            }
        )
        
        return operation
    
    def _get_or_create_node(self, data_source: DataSource) -> LineageNode:
        """Get existing node or create new one for data source."""
        source_key = f"{data_source.source_type}:{data_source.source_id}"
        
        if source_key in self.source_to_node:
            node_id = self.source_to_node[source_key]
            return self.nodes[node_id]
        
        # Create new node
        node_id = str(uuid.uuid4())
        node = LineageNode(
            node_id=node_id,
            data_source=data_source
        )
        
        self.nodes[node_id] = node
        self.source_to_node[source_key] = node_id
        
        return node
    
    def _update_lineage_graph(self, operation: DataOperation):
        """Update lineage graph based on completed operation."""
        # Get input and output nodes
        input_node_ids = []
        output_node_ids = []
        
        for input_source in operation.inputs:
            node = self._get_or_create_node(input_source)
            input_node_ids.append(node.node_id)
        
        for output_source in operation.outputs:
            node = self._get_or_create_node(output_source)
            output_node_ids.append(node.node_id)
        
        # Update dependencies
        for output_node_id in output_node_ids:
            output_node = self.nodes[output_node_id]
            
            for input_node_id in input_node_ids:
                input_node = self.nodes[input_node_id]
                
                # Output depends on input
                output_node.add_dependency(input_node_id)
                # Input has output as dependent
                input_node.add_dependent(output_node_id)
    
    def get_operation(self, operation_id: str) -> Optional[DataOperation]:
        """Get operation by ID."""
        return self.operations.get(operation_id)
    
    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def find_node_by_source(self, source_type: str, source_id: str) -> Optional[LineageNode]:
        """Find node by data source information."""
        source_key = f"{source_type}:{source_id}"
        node_id = self.source_to_node.get(source_key)
        return self.nodes.get(node_id) if node_id else None
    
    def get_upstream_dependencies(self, node_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all upstream dependencies of a node."""
        visited = set()
        to_visit = [(node_id, 0)]
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited:
                continue
            if max_depth and depth > max_depth:
                continue
                
            visited.add(current_id)
            
            node = self.nodes.get(current_id)
            if node:
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        to_visit.append((dep_id, depth + 1))
        
        visited.discard(node_id)  # Remove the starting node
        return visited
    
    def get_downstream_dependents(self, node_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all downstream dependents of a node."""
        visited = set()
        to_visit = [(node_id, 0)]
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited:
                continue
            if max_depth and depth > max_depth:
                continue
                
            visited.add(current_id)
            
            node = self.nodes.get(current_id)
            if node:
                for dep_id in node.dependents:
                    if dep_id not in visited:
                        to_visit.append((dep_id, depth + 1))
        
        visited.discard(node_id)  # Remove the starting node
        return visited
    
    def get_lineage_path(self, source_node_id: str, target_node_id: str) -> List[str]:
        """
        Find lineage path between two nodes using BFS.
        
        Returns list of node IDs representing the path from source to target.
        """
        if source_node_id == target_node_id:
            return [source_node_id]
        
        queue = [(source_node_id, [source_node_id])]
        visited = {source_node_id}
        
        while queue:
            current_id, path = queue.pop(0)
            node = self.nodes.get(current_id)
            
            if not node:
                continue
            
            # Check downstream dependents
            for dependent_id in node.dependents:
                if dependent_id == target_node_id:
                    return path + [dependent_id]
                
                if dependent_id not in visited:
                    visited.add(dependent_id)
                    queue.append((dependent_id, path + [dependent_id]))
        
        return []  # No path found
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lineage tracking statistics."""
        return {
            'total_operations': len(self.operations),
            'total_nodes': len(self.nodes),
            'active_operations': len(self._active_operations),
            'operation_types': {
                op_type.value: len([op for op in self.operations.values() 
                                  if op.operation_type == op_type])
                for op_type in OperationType
            },
            'source_types': {
                source_type: len([node for node in self.nodes.values()
                                if node.data_source.source_type == source_type])
                for source_type in set(node.data_source.source_type 
                                     for node in self.nodes.values())
            }
        }