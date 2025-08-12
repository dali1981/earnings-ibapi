"""
Tests for data lineage tracking system.
"""
import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import tempfile
import shutil

from lineage.core import (
    LineageTracker, DataSource, DataOperation, LineageNode, OperationType
)
from lineage.metadata import LineageMetadataRepository
from lineage.query import LineageQueryEngine
from lineage.visualizer import LineageVisualizer
from lineage.decorators import track_lineage, set_global_tracker


class TestLineageCore:
    """Test core lineage tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = LineageTracker()
    
    def test_create_data_source(self):
        """Test DataSource creation."""
        source = DataSource(
            source_type="repository",
            source_id="equity_bars", 
            location="/data/equity_bars",
            metadata={"schema": "ohlcv"}
        )
        
        assert source.source_type == "repository"
        assert source.source_id == "equity_bars"
        assert source.location == "/data/equity_bars"
        assert source.metadata["schema"] == "ohlcv"
    
    def test_start_and_complete_operation(self):
        """Test basic operation lifecycle."""
        # Start operation
        op_id = self.tracker.start_operation(
            OperationType.READ,
            parameters={"symbol": "AAPL"},
            execution_context={"user": "test"}
        )
        
        assert op_id in self.tracker._active_operations
        
        # Add input source
        input_source = DataSource("repository", "equity_bars", "/data/equity_bars")
        self.tracker.add_input(op_id, input_source)
        
        # Add output source
        output_source = DataSource("dataframe", "result_df", "memory")
        self.tracker.add_output(op_id, output_source)
        
        # Complete operation
        completed_op = self.tracker.complete_operation(op_id, record_count_out=1000)
        
        assert op_id not in self.tracker._active_operations
        assert op_id in self.tracker.operations
        assert completed_op.record_count_out == 1000
        assert completed_op.duration_ms is not None
    
    def test_lineage_graph_creation(self):
        """Test that lineage graph is built correctly."""
        # Create a simple data flow: source -> transform -> destination
        
        # Operation 1: Read from source
        op1_id = self.tracker.start_operation(OperationType.READ)
        source1 = DataSource("repository", "source_data", "/data/source")
        output1 = DataSource("dataframe", "df1", "memory")
        self.tracker.add_input(op1_id, source1)
        self.tracker.add_output(op1_id, output1)
        self.tracker.complete_operation(op1_id)
        
        # Operation 2: Transform data
        op2_id = self.tracker.start_operation(OperationType.TRANSFORM)
        input2 = DataSource("dataframe", "df1", "memory") 
        output2 = DataSource("dataframe", "df2", "memory")
        self.tracker.add_input(op2_id, input2)
        self.tracker.add_output(op2_id, output2)
        self.tracker.complete_operation(op2_id)
        
        # Operation 3: Write to destination
        op3_id = self.tracker.start_operation(OperationType.WRITE)
        input3 = DataSource("dataframe", "df2", "memory")
        dest = DataSource("repository", "dest_data", "/data/dest")
        self.tracker.add_input(op3_id, input3)
        self.tracker.add_output(op3_id, dest)
        self.tracker.complete_operation(op3_id)
        
        # Verify lineage relationships
        dest_node = self.tracker.find_node_by_source("repository", "dest_data")
        assert dest_node is not None
        
        # Trace upstream dependencies
        upstream = self.tracker.get_upstream_dependencies(dest_node.node_id)
        assert len(upstream) >= 2  # Should include intermediate and source nodes
    
    def test_get_lineage_path(self):
        """Test finding paths between nodes."""
        # Create linear chain: A -> B -> C
        op1_id = self.tracker.start_operation(OperationType.TRANSFORM)
        source_a = DataSource("repository", "A", "/data/A")
        output_b = DataSource("repository", "B", "/data/B") 
        self.tracker.add_input(op1_id, source_a)
        self.tracker.add_output(op1_id, output_b)
        self.tracker.complete_operation(op1_id)
        
        op2_id = self.tracker.start_operation(OperationType.TRANSFORM)
        input_b = DataSource("repository", "B", "/data/B")
        output_c = DataSource("repository", "C", "/data/C")
        self.tracker.add_input(op2_id, input_b)
        self.tracker.add_output(op2_id, output_c)
        self.tracker.complete_operation(op2_id)
        
        # Find nodes
        node_a = self.tracker.find_node_by_source("repository", "A")
        node_c = self.tracker.find_node_by_source("repository", "C")
        
        assert node_a is not None
        assert node_c is not None
        
        # Find path from A to C
        path = self.tracker.get_lineage_path(node_a.node_id, node_c.node_id)
        assert len(path) == 3  # A -> B -> C
    
    def test_statistics(self):
        """Test statistics collection."""
        # Create some operations
        for i in range(3):
            op_id = self.tracker.start_operation(OperationType.READ)
            source = DataSource("repository", f"data_{i}", f"/data/data_{i}")
            self.tracker.add_input(op_id, source)
            self.tracker.complete_operation(op_id)
        
        stats = self.tracker.get_statistics()
        
        assert stats['total_operations'] == 3
        assert stats['total_nodes'] == 3
        assert stats['operation_types']['read'] == 3


class TestLineageMetadata:
    """Test lineage metadata persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo = LineageMetadataRepository(self.temp_dir)
        self.tracker = LineageTracker(storage_backend=self.repo)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_operations(self):
        """Test saving and loading operations."""
        # Create operation
        op_id = self.tracker.start_operation(
            OperationType.WRITE,
            parameters={"symbol": "AAPL"},
            execution_context={"user": "test"}
        )
        
        input_source = DataSource("dataframe", "df", "memory")
        output_source = DataSource("repository", "equity_bars", "/data/equity_bars")
        
        self.tracker.add_input(op_id, input_source)
        self.tracker.add_output(op_id, output_source) 
        
        completed_op = self.tracker.complete_operation(op_id, record_count_in=1000)
        
        # Load operations from storage
        loaded_ops = self.repo.load_operations()
        
        assert len(loaded_ops) == 1
        loaded_op = loaded_ops[0]
        
        assert loaded_op.operation_id == completed_op.operation_id
        assert loaded_op.operation_type == OperationType.WRITE
        assert loaded_op.record_count_in == 1000
        assert len(loaded_op.inputs) == 1
        assert len(loaded_op.outputs) == 1
    
    def test_save_and_load_nodes(self):
        """Test saving and loading nodes."""
        # Create operation that generates nodes
        op_id = self.tracker.start_operation(OperationType.READ)
        source = DataSource("repository", "test_data", "/data/test")
        self.tracker.add_input(op_id, source)
        self.tracker.complete_operation(op_id)
        
        # Load nodes from storage
        loaded_nodes = self.repo.load_nodes()
        
        assert len(loaded_nodes) == 1
        node = loaded_nodes[0]
        
        assert node.data_source.source_type == "repository"
        assert node.data_source.source_id == "test_data"
        assert node.data_source.location == "/data/test"
    
    def test_storage_stats(self):
        """Test storage statistics."""
        stats = self.repo.get_storage_stats()
        
        assert 'operations_path' in stats
        assert 'nodes_path' in stats
        assert 'relationships_path' in stats


class TestLineageQuery:
    """Test lineage querying functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = LineageTracker()
        self.query_engine = LineageQueryEngine(self.tracker)
        self._create_sample_lineage()
    
    def _create_sample_lineage(self):
        """Create sample lineage data for testing."""
        # Create a more complex lineage graph for testing
        
        # Raw data ingestion
        op1_id = self.tracker.start_operation(OperationType.READ)
        raw_source = DataSource("api", "market_data", "https://api.example.com")
        raw_df = DataSource("dataframe", "raw_df", "memory")
        self.tracker.add_input(op1_id, raw_source)
        self.tracker.add_output(op1_id, raw_df)
        self.tracker.complete_operation(op1_id, record_count_out=5000)
        
        # Data cleaning
        op2_id = self.tracker.start_operation(OperationType.TRANSFORM)
        clean_df = DataSource("dataframe", "clean_df", "memory")
        self.tracker.add_input(op2_id, raw_df)
        self.tracker.add_output(op2_id, clean_df)
        self.tracker.complete_operation(op2_id, record_count_in=5000, record_count_out=4800)
        
        # Split into different datasets
        op3_id = self.tracker.start_operation(OperationType.WRITE)
        equity_data = DataSource("repository", "equity_bars", "/data/equity_bars")
        self.tracker.add_input(op3_id, clean_df)
        self.tracker.add_output(op3_id, equity_data)
        self.tracker.complete_operation(op3_id, record_count_in=4800)
        
        op4_id = self.tracker.start_operation(OperationType.WRITE)
        option_data = DataSource("repository", "option_bars", "/data/option_bars")
        self.tracker.add_input(op4_id, clean_df)
        self.tracker.add_output(op4_id, option_data)
        self.tracker.complete_operation(op4_id, record_count_in=4800)
    
    def test_find_data_sources(self):
        """Test finding data sources by criteria."""
        # Find repository sources
        repo_sources = self.query_engine.find_data_sources(source_type="repository")
        assert len(repo_sources) == 2
        
        # Find sources by location pattern
        data_sources = self.query_engine.find_data_sources(location_pattern="/data/")
        assert len(data_sources) == 2
    
    def test_get_data_lineage(self):
        """Test getting complete data lineage."""
        equity_node = self.tracker.find_node_by_source("repository", "equity_bars")
        assert equity_node is not None
        
        lineage = self.query_engine.get_data_lineage(equity_node.node_id)
        
        assert lineage['root_node'] == equity_node
        assert len(lineage['upstream_dependencies']) >= 2  # Should include raw and clean data
        assert len(lineage['related_operations']) >= 2
    
    def test_trace_data_flow(self):
        """Test tracing data flow between nodes."""
        raw_node = self.tracker.find_node_by_source("api", "market_data")
        equity_node = self.tracker.find_node_by_source("repository", "equity_bars")
        
        assert raw_node is not None
        assert equity_node is not None
        
        flow = self.query_engine.trace_data_flow(raw_node.node_id, equity_node.node_id)
        
        assert flow['path_found'] is True
        assert flow['path_length'] >= 3  # raw -> clean -> equity
        assert len(flow['transformation_operations']) >= 2
    
    def test_analyze_data_volume_flow(self):
        """Test data volume flow analysis."""
        volume_analysis = self.query_engine.analyze_data_volume_flow()
        
        assert volume_analysis['total_operations'] == 4
        assert volume_analysis['total_records_in'] > 0
        assert volume_analysis['total_records_out'] > 0
        assert 'read' in volume_analysis['operations_by_type']
        assert 'write' in volume_analysis['operations_by_type']
    
    def test_get_operation_performance_stats(self):
        """Test operation performance statistics."""
        stats = self.query_engine.get_operation_performance_stats()
        
        assert stats['total_operations'] == 4
        assert stats['operations_with_timing'] == 4
        assert stats['failed_operations'] == 0
        assert 'avg_duration_ms' in stats
    
    def test_build_lineage_summary(self):
        """Test building lineage summary."""
        summary = self.query_engine.build_lineage_summary()
        
        assert 'basic_stats' in summary
        assert 'graph_structure' in summary
        assert 'top_connected_nodes' in summary
        assert 'recent_activity' in summary
        
        assert summary['basic_stats']['total_operations'] == 4
        assert summary['graph_structure']['nodes_with_dependencies'] >= 2


class TestLineageVisualization:
    """Test lineage visualization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = LineageTracker()
        self.visualizer = LineageVisualizer(self.tracker)
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for visualization tests."""
        op_id = self.tracker.start_operation(OperationType.TRANSFORM)
        
        input_source = DataSource("repository", "input_data", "/data/input")
        output_source = DataSource("repository", "output_data", "/data/output")
        
        self.tracker.add_input(op_id, input_source)
        self.tracker.add_output(op_id, output_source)
        self.tracker.complete_operation(op_id, record_count_in=1000, record_count_out=900)
    
    def test_generate_graph_data(self):
        """Test generating graph data structure."""
        graph_data = self.visualizer.generate_graph_data()
        
        assert 'nodes' in graph_data
        assert 'edges' in graph_data
        assert 'metadata' in graph_data
        
        assert len(graph_data['nodes']) >= 2  # Input and output nodes
        assert len(graph_data['edges']) >= 1  # At least one dependency edge
    
    def test_generate_graph_data_focused(self):
        """Test generating focused graph data."""
        output_node = self.tracker.find_node_by_source("repository", "output_data")
        assert output_node is not None
        
        graph_data = self.visualizer.generate_graph_data(
            node_id=output_node.node_id,
            max_depth=2
        )
        
        assert graph_data['metadata']['root_node_id'] == output_node.node_id
        assert len(graph_data['nodes']) >= 1
    
    def test_generate_timeline_data(self):
        """Test generating timeline data."""
        timeline = self.visualizer.generate_timeline_data(hours_back=1.0)
        
        assert 'events' in timeline
        assert 'time_range' in timeline
        assert 'summary' in timeline
        
        assert len(timeline['events']) == 1  # One operation
        assert timeline['summary']['total_events'] == 1
    
    def test_generate_impact_analysis(self):
        """Test generating impact analysis."""
        input_node = self.tracker.find_node_by_source("repository", "input_data")
        assert input_node is not None
        
        impact = self.visualizer.generate_impact_analysis(input_node.node_id)
        
        assert 'target_node' in impact
        assert 'total_downstream_nodes' in impact
        assert 'impact_by_source_type' in impact
        assert 'impact_summary' in impact
        
        assert impact['total_downstream_nodes'] >= 1  # Output node depends on input
    
    def test_export_to_dot(self):
        """Test exporting to DOT format."""
        dot_output = self.visualizer.export_to_dot()
        
        assert dot_output.startswith("digraph lineage")
        assert "rankdir=TB" in dot_output
        assert "input_data" in dot_output
        assert "output_data" in dot_output
    
    def test_export_to_json(self):
        """Test exporting to JSON format."""
        json_output = self.visualizer.export_to_json(pretty_print=True)
        
        assert '"nodes":' in json_output
        assert '"edges":' in json_output
        assert '"metadata":' in json_output
    
    def test_generate_summary_report(self):
        """Test generating summary report."""
        report = self.visualizer.generate_summary_report()
        
        assert "DATA LINEAGE SUMMARY REPORT" in report
        assert "BASIC STATISTICS:" in report
        assert "Total Operations:" in report
        assert "Total Nodes:" in report


class TestLineageDecorators:
    """Test lineage tracking decorators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = LineageTracker()
        set_global_tracker(self.tracker)
    
    def test_track_lineage_decorator(self):
        """Test basic lineage tracking decorator."""
        @track_lineage(
            operation_type=OperationType.TRANSFORM,
            input_sources=['data'],
            repository_type='test'
        )
        def process_data(data, multiplier=2):
            return data * multiplier
        
        # Execute tracked function
        test_data = pd.DataFrame({'value': [1, 2, 3]})
        result = process_data(test_data, multiplier=3)
        
        # Verify tracking occurred
        assert len(self.tracker.operations) == 1
        
        operation = list(self.tracker.operations.values())[0]
        assert operation.operation_type == OperationType.TRANSFORM
        assert len(operation.inputs) == 1
        assert operation.parameters['multiplier'] == 3
    
    def test_track_lineage_decorator_with_error(self):
        """Test lineage tracking with errors."""
        @track_lineage(
            operation_type=OperationType.READ,
            input_sources=['path']
        )
        def failing_function(path):
            raise ValueError("Test error")
        
        # Execute function that raises error
        with pytest.raises(ValueError):
            failing_function("/test/path")
        
        # Verify error was tracked
        assert len(self.tracker.operations) == 1
        
        operation = list(self.tracker.operations.values())[0]
        assert operation.error_info == "Test error"


class TestIntegrationWithRepositories:
    """Test integration with existing repository classes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = LineageTracker()
        set_global_tracker(self.tracker)
        
        # Create temporary directory for test repository
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_repository_integration(self):
        """Test that repository operations are tracked."""
        from repositories.equity_bars import EquityBarRepository
        
        # Create repository
        repo = EquityBarRepository(self.temp_dir)
        
        # Create test data
        test_data = pd.DataFrame({
            'datetime': ['20240101 09:30:00'],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        })
        
        # Save data (should be tracked)
        repo.save(test_data, symbol='TEST', bar_size='1 day')
        
        # Load data (should be tracked)  
        loaded_data = repo.load(symbol='TEST')
        
        # Verify tracking occurred
        operations = list(self.tracker.operations.values())
        
        # Should have at least save and load operations
        save_ops = [op for op in operations if op.operation_type == OperationType.WRITE]
        load_ops = [op for op in operations if op.operation_type == OperationType.READ]
        
        assert len(save_ops) >= 1
        assert len(load_ops) >= 1
        
        # Verify save operation tracked input data
        save_op = save_ops[0]
        assert len(save_op.inputs) >= 1
        assert save_op.record_count_in == 1
        
        # Verify load operation produced output
        load_op = load_ops[0]
        assert len(load_op.outputs) >= 1
        assert load_op.record_count_out == len(loaded_data)