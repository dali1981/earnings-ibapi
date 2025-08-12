"""
Metadata storage for data lineage information.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .core import DataOperation, LineageNode, OperationType

try:
    from reliability import get_trading_logger, StorageException
except ImportError:
    import logging
    
    def get_trading_logger(name: str):
        return logging.getLogger(name)
    
    class StorageException(Exception):
        pass


class LineageMetadataRepository:
    """
    Repository for persisting lineage metadata using Parquet.
    
    Stores operations and nodes in separate datasets with Hive partitioning.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize lineage metadata repository.
        
        Args:
            base_path: Base directory for storing lineage data
        """
        self.base_path = Path(base_path)
        self.operations_path = self.base_path / "lineage_operations"
        self.nodes_path = self.base_path / "lineage_nodes"
        self.relationships_path = self.base_path / "lineage_relationships"
        
        # Create directories
        self.operations_path.mkdir(parents=True, exist_ok=True)
        self.nodes_path.mkdir(parents=True, exist_ok=True)
        self.relationships_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_trading_logger("LineageMetadataRepository")
        
        # Initialize schemas
        self._operations_schema = self._create_operations_schema()
        self._nodes_schema = self._create_nodes_schema()
        self._relationships_schema = self._create_relationships_schema()
        
        # Initialize empty datasets
        self._initialize_datasets()
    
    def _create_operations_schema(self) -> pa.Schema:
        """Create schema for operations dataset."""
        return pa.schema([
            pa.field('operation_id', pa.string()),
            pa.field('operation_type', pa.string()),
            pa.field('timestamp', pa.timestamp('ns')),
            pa.field('date', pa.date32()),  # For partitioning
            pa.field('inputs', pa.string()),  # JSON serialized
            pa.field('outputs', pa.string()), # JSON serialized
            pa.field('parameters', pa.string(), nullable=True),
            pa.field('execution_context', pa.string(), nullable=True),
            pa.field('user_info', pa.string(), nullable=True),
            pa.field('duration_ms', pa.float64(), nullable=True),
            pa.field('record_count_in', pa.int64(), nullable=True),
            pa.field('record_count_out', pa.int64(), nullable=True),
            pa.field('error_info', pa.string(), nullable=True),
        ])
    
    def _create_nodes_schema(self) -> pa.Schema:
        """Create schema for nodes dataset."""
        return pa.schema([
            pa.field('node_id', pa.string()),
            pa.field('source_type', pa.string()),
            pa.field('source_id', pa.string()),
            pa.field('location', pa.string()),
            pa.field('schema_info', pa.string(), nullable=True),
            pa.field('metadata', pa.string(), nullable=True),
            pa.field('created_by', pa.string(), nullable=True),
            pa.field('created_at', pa.timestamp('ns'), nullable=True),
            pa.field('accessed_count', pa.int64()),
            pa.field('last_accessed', pa.timestamp('ns'), nullable=True),
        ])
    
    def _create_relationships_schema(self) -> pa.Schema:
        """Create schema for node relationships dataset."""
        return pa.schema([
            pa.field('from_node_id', pa.string()),
            pa.field('to_node_id', pa.string()),
            pa.field('relationship_type', pa.string()),  # 'dependency' or 'dependent'
            pa.field('created_at', pa.timestamp('ns')),
            pa.field('operation_id', pa.string(), nullable=True),  # Operation that created this relationship
        ])
    
    def _initialize_datasets(self):
        """Initialize empty Parquet datasets if they don't exist."""
        datasets = [
            (self.operations_path, self._operations_schema),
            (self.nodes_path, self._nodes_schema),
            (self.relationships_path, self._relationships_schema)
        ]
        
        for dataset_path, schema in datasets:
            if not any(dataset_path.rglob("*.parquet")):
                try:
                    empty_df = pd.DataFrame(columns=[field.name for field in schema])
                    empty_table = pa.Table.from_pandas(empty_df, schema=schema)
                    
                    pq.write_to_dataset(
                        empty_table,
                        root_path=str(dataset_path),
                        partition_cols=[],
                        use_dictionary=True,
                        compression='snappy'
                    )
                    
                    self.logger.debug(f"Created empty dataset at {dataset_path}")
                    
                except Exception as e:
                    raise StorageException(f"Failed to initialize dataset {dataset_path}: {e}")
    
    def save_operation(self, operation: DataOperation) -> None:
        """Save operation to storage."""
        try:
            # Convert operation to DataFrame row
            df = pd.DataFrame([{
                'operation_id': operation.operation_id,
                'operation_type': operation.operation_type.value,
                'timestamp': operation.timestamp,
                'date': operation.timestamp.date(),
                'inputs': json.dumps([self._serialize_data_source(ds) for ds in operation.inputs]),
                'outputs': json.dumps([self._serialize_data_source(ds) for ds in operation.outputs]),
                'parameters': json.dumps(operation.parameters) if operation.parameters else None,
                'execution_context': json.dumps(operation.execution_context) if operation.execution_context else None,
                'user_info': operation.user_info,
                'duration_ms': operation.duration_ms,
                'record_count_in': operation.record_count_in,
                'record_count_out': operation.record_count_out,
                'error_info': operation.error_info,
            }])
            
            # Convert to Arrow table
            table = pa.Table.from_pandas(df, schema=self._operations_schema, preserve_index=False)
            
            # Write to dataset with date partitioning
            pq.write_to_dataset(
                table,
                root_path=str(self.operations_path),
                partition_cols=['date'],
                use_dictionary=True,
                compression='snappy',
                existing_data_behavior='overwrite_or_ignore'
            )
            
            self.logger.debug(f"Saved operation {operation.operation_id}")
            
        except Exception as e:
            raise StorageException(f"Failed to save operation {operation.operation_id}: {e}")
    
    def save_node(self, node: LineageNode) -> None:
        """Save node to storage."""
        try:
            # Convert node to DataFrame row
            df = pd.DataFrame([{
                'node_id': node.node_id,
                'source_type': node.data_source.source_type,
                'source_id': node.data_source.source_id,
                'location': node.data_source.location,
                'schema_info': json.dumps(node.data_source.schema_info) if node.data_source.schema_info else None,
                'metadata': json.dumps(node.data_source.metadata) if node.data_source.metadata else None,
                'created_by': node.created_by,
                'created_at': node.created_at,
                'accessed_count': node.accessed_count,
                'last_accessed': node.last_accessed,
            }])
            
            # Convert to Arrow table
            table = pa.Table.from_pandas(df, schema=self._nodes_schema, preserve_index=False)
            
            # Write to dataset
            pq.write_to_dataset(
                table,
                root_path=str(self.nodes_path),
                partition_cols=[],
                use_dictionary=True,
                compression='snappy',
                existing_data_behavior='overwrite_or_ignore'
            )
            
            self.logger.debug(f"Saved node {node.node_id}")
            
        except Exception as e:
            raise StorageException(f"Failed to save node {node.node_id}: {e}")
    
    def save_relationships(self, node: LineageNode, operation_id: Optional[str] = None) -> None:
        """Save node relationships to storage."""
        try:
            relationships = []
            timestamp = datetime.utcnow()
            
            # Save dependencies
            for dep_node_id in node.dependencies:
                relationships.append({
                    'from_node_id': dep_node_id,
                    'to_node_id': node.node_id,
                    'relationship_type': 'dependency',
                    'created_at': timestamp,
                    'operation_id': operation_id
                })
            
            # Save dependents
            for dep_node_id in node.dependents:
                relationships.append({
                    'from_node_id': node.node_id,
                    'to_node_id': dep_node_id,
                    'relationship_type': 'dependent',
                    'created_at': timestamp,
                    'operation_id': operation_id
                })
            
            if relationships:
                df = pd.DataFrame(relationships)
                table = pa.Table.from_pandas(df, schema=self._relationships_schema, preserve_index=False)
                
                # Write to dataset
                pq.write_to_dataset(
                    table,
                    root_path=str(self.relationships_path),
                    partition_cols=[],
                    use_dictionary=True,
                    compression='snappy',
                    existing_data_behavior='overwrite_or_ignore'
                )
                
                self.logger.debug(f"Saved {len(relationships)} relationships for node {node.node_id}")
            
        except Exception as e:
            raise StorageException(f"Failed to save relationships for node {node.node_id}: {e}")
    
    def load_operations(self, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None,
                       operation_type: Optional[OperationType] = None) -> List[DataOperation]:
        """Load operations from storage with filtering."""
        try:
            dataset = ds.dataset(
                str(self.operations_path),
                format="parquet",
                partitioning="hive"
            )
            
            # Build filter expression
            filters = []
            if start_date:
                filters.append(ds.field('timestamp') >= start_date)
            if end_date:
                filters.append(ds.field('timestamp') <= end_date)
            if operation_type:
                filters.append(ds.field('operation_type') == operation_type.value)
            
            filter_expr = None
            for f in filters:
                filter_expr = f if filter_expr is None else (filter_expr & f)
            
            # Load data
            table = dataset.to_table(filter=filter_expr)
            df = table.to_pandas()
            
            # Convert back to DataOperation objects
            operations = []
            for _, row in df.iterrows():
                op = self._deserialize_operation(row)
                operations.append(op)
            
            self.logger.debug(f"Loaded {len(operations)} operations")
            return operations
            
        except Exception as e:
            raise StorageException(f"Failed to load operations: {e}")
    
    def load_nodes(self, source_type: Optional[str] = None) -> List[LineageNode]:
        """Load nodes from storage with filtering."""
        try:
            dataset = ds.dataset(
                str(self.nodes_path),
                format="parquet"
            )
            
            # Build filter expression
            filter_expr = None
            if source_type:
                filter_expr = ds.field('source_type') == source_type
            
            # Load data
            table = dataset.to_table(filter=filter_expr)
            df = table.to_pandas()
            
            # Convert back to LineageNode objects
            nodes = []
            for _, row in df.iterrows():
                node = self._deserialize_node(row)
                nodes.append(node)
            
            # Load relationships for each node
            self._load_node_relationships(nodes)
            
            self.logger.debug(f"Loaded {len(nodes)} nodes")
            return nodes
            
        except Exception as e:
            raise StorageException(f"Failed to load nodes: {e}")
    
    def _load_node_relationships(self, nodes: List[LineageNode]):
        """Load relationships for a list of nodes."""
        try:
            dataset = ds.dataset(
                str(self.relationships_path),
                format="parquet"
            )
            
            table = dataset.to_table()
            df = table.to_pandas()
            
            # Build node lookup
            node_lookup = {node.node_id: node for node in nodes}
            
            # Process relationships
            for _, row in df.iterrows():
                from_node_id = row['from_node_id']
                to_node_id = row['to_node_id']
                rel_type = row['relationship_type']
                
                from_node = node_lookup.get(from_node_id)
                to_node = node_lookup.get(to_node_id)
                
                if from_node and to_node:
                    if rel_type == 'dependency':
                        to_node.add_dependency(from_node_id)
                        from_node.add_dependent(to_node_id)
                    elif rel_type == 'dependent':
                        from_node.add_dependent(to_node_id)
                        to_node.add_dependency(from_node_id)
            
        except Exception as e:
            self.logger.warning(f"Failed to load relationships: {e}")
    
    def _serialize_data_source(self, data_source) -> Dict[str, Any]:
        """Serialize DataSource to dictionary."""
        return {
            'source_type': data_source.source_type,
            'source_id': data_source.source_id,
            'location': data_source.location,
            'schema_info': data_source.schema_info,
            'metadata': data_source.metadata
        }
    
    def _deserialize_operation(self, row: pd.Series) -> DataOperation:
        """Deserialize operation from pandas Series."""
        from .core import DataSource  # Import here to avoid circular imports
        
        # Deserialize inputs and outputs
        inputs = []
        if pd.notna(row['inputs']):
            for input_data in json.loads(row['inputs']):
                inputs.append(DataSource(**input_data))
        
        outputs = []
        if pd.notna(row['outputs']):
            for output_data in json.loads(row['outputs']):
                outputs.append(DataSource(**output_data))
        
        return DataOperation(
            operation_id=row['operation_id'],
            operation_type=OperationType(row['operation_type']),
            timestamp=row['timestamp'],
            inputs=inputs,
            outputs=outputs,
            parameters=json.loads(row['parameters']) if pd.notna(row['parameters']) else {},
            execution_context=json.loads(row['execution_context']) if pd.notna(row['execution_context']) else {},
            user_info=row['user_info'] if pd.notna(row['user_info']) else None,
            duration_ms=row['duration_ms'] if pd.notna(row['duration_ms']) else None,
            record_count_in=row['record_count_in'] if pd.notna(row['record_count_in']) else None,
            record_count_out=row['record_count_out'] if pd.notna(row['record_count_out']) else None,
            error_info=row['error_info'] if pd.notna(row['error_info']) else None
        )
    
    def _deserialize_node(self, row: pd.Series) -> LineageNode:
        """Deserialize node from pandas Series."""
        from .core import DataSource  # Import here to avoid circular imports
        
        data_source = DataSource(
            source_type=row['source_type'],
            source_id=row['source_id'],
            location=row['location'],
            schema_info=json.loads(row['schema_info']) if pd.notna(row['schema_info']) else None,
            metadata=json.loads(row['metadata']) if pd.notna(row['metadata']) else None
        )
        
        return LineageNode(
            node_id=row['node_id'],
            data_source=data_source,
            created_by=row['created_by'] if pd.notna(row['created_by']) else None,
            created_at=row['created_at'] if pd.notna(row['created_at']) else None,
            accessed_count=row['accessed_count'],
            last_accessed=row['last_accessed'] if pd.notna(row['last_accessed']) else None
        )
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'operations_path': str(self.operations_path),
            'nodes_path': str(self.nodes_path),
            'relationships_path': str(self.relationships_path)
        }
        
        # Count files in each dataset
        for name, path_str in [('operations', str(self.operations_path)), 
                               ('nodes', str(self.nodes_path)),
                               ('relationships', str(self.relationships_path))]:
            try:
                path = Path(path_str)
                parquet_files = list(path.rglob("*.parquet"))
                stats[f'{name}_file_count'] = len(parquet_files)
                stats[f'{name}_total_size'] = sum(f.stat().st_size for f in parquet_files)
            except Exception as e:
                stats[f'{name}_error'] = str(e)
        
        return stats