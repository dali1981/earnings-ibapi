# Data Lineage Tracking System

A comprehensive data lineage tracking system for the trading data pipeline. This system automatically tracks data transformations, dependencies, and flow across all data repositories.

## Features

- **Automatic Tracking**: Seamlessly tracks data operations in existing repositories
- **Comprehensive Metadata**: Stores detailed operation metadata, timing, and data volumes
- **Lineage Graph**: Builds complete data dependency graphs
- **Query Engine**: Advanced querying capabilities for lineage analysis
- **Impact Analysis**: Understand downstream impact of data changes
- **Visualization**: Multiple visualization formats including graphs and timelines
- **Persistent Storage**: Parquet-based metadata storage with Hive partitioning
- **Performance Monitoring**: Track operation performance and data volumes

## Architecture

The system consists of several key components:

### Core Components

- **LineageTracker**: Central tracking engine that manages operations and nodes
- **DataSource**: Represents data sources in the lineage graph  
- **DataOperation**: Represents operations that transform data
- **LineageNode**: Represents nodes in the lineage dependency graph

### Storage Layer

- **LineageMetadataRepository**: Persistent storage using Parquet datasets
- Stores operations, nodes, and relationships with Hive partitioning
- Supports time-based queries and efficient data retrieval

### Query & Analysis

- **LineageQueryEngine**: Advanced querying for lineage analysis
- **LineageVisualizer**: Generate visualizations and reports
- Impact analysis and data freshness tracking
- Performance statistics and volume flow analysis

### Integration Layer

- **Decorators**: Automatic tracking integration with existing code
- **Configuration**: Flexible configuration system
- Repository integration with minimal code changes

## Quick Start

### 1. Basic Setup

```python
from lineage import LineageTracker, LineageMetadataRepository, set_global_tracker
from pathlib import Path

# Create metadata storage
metadata_repo = LineageMetadataRepository("/path/to/lineage/data")

# Create tracker with persistence
tracker = LineageTracker(storage_backend=metadata_repo)

# Set as global tracker for automatic integration
set_global_tracker(tracker)
```

### 2. Repository Integration

The system automatically integrates with existing repositories:

```python
from repositories.equity_bars import EquityBarRepository

# Repository operations are automatically tracked
repo = EquityBarRepository("/path/to/data")

# This operation will be automatically tracked
repo.save_daily_bars(df, symbol='AAPL')

# This operation will also be tracked
loaded_data = repo.load_symbol_data('AAPL')
```

### 3. Manual Tracking

For custom operations:

```python
from lineage.decorators import track_lineage
from lineage.core import OperationType

@track_lineage(
    operation_type=OperationType.TRANSFORM,
    input_sources=['input_data']
)
def process_data(input_data, multiplier=2):
    return input_data * multiplier
```

### 4. Querying Lineage

```python
from lineage.query import LineageQueryEngine

query_engine = LineageQueryEngine(tracker)

# Find data sources
repo_sources = query_engine.find_data_sources(source_type="repository")

# Get complete lineage for a node
lineage = query_engine.get_data_lineage(node_id)

# Trace data flow between nodes
flow = query_engine.trace_data_flow(source_node_id, target_node_id)

# Analyze data volume flows
volume_stats = query_engine.analyze_data_volume_flow()
```

### 5. Visualization

```python
from lineage.visualizer import LineageVisualizer

visualizer = LineageVisualizer(tracker)

# Generate graph data for visualization tools
graph_data = visualizer.generate_graph_data()

# Create impact analysis
impact = visualizer.generate_impact_analysis(node_id)

# Export to various formats
dot_graph = visualizer.export_to_dot()
json_data = visualizer.export_to_json()
report = visualizer.generate_summary_report()
```

## Configuration

### Environment Variables

```bash
export LINEAGE_BASE_PATH="/data/lineage"
export LINEAGE_ENABLE_PERSISTENCE="true"
export LINEAGE_TRACK_REPOS="true"
export LINEAGE_MAX_GRAPH_NODES="2000"
```

### Configuration File

```json
{
  "base_path": "./data/lineage",
  "enable_persistence": true,
  "track_repository_operations": true,
  "enabled_operation_types": ["read", "write", "transform"],
  "max_graph_nodes": 1000,
  "repository_settings": {
    "equity_bars": {
      "track_all_operations": true,
      "enable_volume_tracking": true
    }
  }
}
```

```python
from lineage.config import LineageConfig

# Load from file
config = LineageConfig.from_file(Path("lineage_config.json"))

# Load from environment
config = LineageConfig.from_env()
```

## Use Cases

### 1. Data Quality Monitoring

Track data transformations to identify quality issues:

```python
# Find operations with errors
failed_ops = [op for op in tracker.operations.values() if op.error_info]

# Analyze data volume changes
volume_analysis = query_engine.analyze_data_volume_flow()
```

### 2. Impact Analysis

Understand the impact of changing upstream data:

```python
# Find what would be affected by changing a data source
impact = visualizer.generate_impact_analysis(source_node_id)
print(f"Downstream impact: {impact['total_downstream_nodes']} nodes")
```

### 3. Data Freshness Tracking

Monitor when data was last updated:

```python
# Find stale data sources
stale_data = query_engine.find_stale_data(hours_threshold=24.0)
for item in stale_data:
    print(f"Stale: {item['node'].data_source.source_id}")
```

### 4. Performance Monitoring

Track operation performance over time:

```python
# Get performance statistics
perf_stats = query_engine.get_operation_performance_stats(
    operation_type=OperationType.READ
)
print(f"Average read time: {perf_stats['avg_duration_ms']}ms")
```

### 5. Compliance & Auditing

Maintain complete audit trail of data operations:

```python
# Get timeline of operations
timeline = visualizer.generate_timeline_data(hours_back=24.0)

# Generate comprehensive report
report = visualizer.generate_summary_report()
```

## Data Model

### Operations

Each operation tracks:
- Operation type (READ, WRITE, TRANSFORM, etc.)
- Input and output data sources
- Parameters and execution context
- Performance metrics (duration, record counts)
- Error information if applicable

### Nodes

Each node represents a data source with:
- Source type and identifier
- Location and schema information
- Access patterns and statistics
- Dependency relationships

### Relationships

Dependencies are tracked bidirectionally:
- Upstream dependencies (what this depends on)
- Downstream dependents (what depends on this)

## Storage Format

The system uses Parquet datasets with Hive partitioning:

```
lineage_metadata/
├── lineage_operations/
│   └── date=2024-01-15/
│       └── operations.parquet
├── lineage_nodes/
│   └── nodes.parquet
└── lineage_relationships/
    └── relationships.parquet
```

## Testing

Run the test suite:

```bash
pytest tests/test_lineage.py -v
```

The tests cover:
- Core functionality
- Metadata persistence  
- Query operations
- Visualization generation
- Integration with repositories

## Examples

See `examples/lineage_example.py` for a comprehensive demonstration of all features.

## Performance Considerations

- **Memory Usage**: The system maintains in-memory graphs for active analysis
- **Storage Size**: Parquet compression keeps metadata storage efficient
- **Query Performance**: Hive partitioning enables efficient time-based queries
- **Scalability**: Designed to handle thousands of operations and nodes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the lineage module is in your Python path
2. **Permission Errors**: Check write permissions to the lineage data directory
3. **Memory Issues**: Use the cleanup functionality for long-running processes

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger("lineage").setLevel(logging.DEBUG)
```

### Configuration Validation

```python
from lineage.config import get_config

config = get_config()
issues = config.validate()
if issues:
    print(f"Configuration issues: {issues}")
```

## Contributing

The lineage system is designed to be extensible:

1. **Custom Operation Types**: Add new operation types to the OperationType enum
2. **Storage Backends**: Implement alternative storage backends
3. **Visualization Formats**: Add new export formats to the visualizer
4. **Query Methods**: Extend the query engine with custom analysis methods

## Future Enhancements

- Real-time lineage streaming
- Integration with workflow orchestrators
- Advanced ML-based anomaly detection
- Web-based visualization dashboard
- Data catalog integration