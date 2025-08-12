"""
Example usage of the data lineage tracking system.

This example demonstrates how to set up and use the lineage tracking system
with the trading data repositories.
"""
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import tempfile

# Import lineage components
from lineage import (
    LineageTracker, LineageMetadataRepository, LineageQueryEngine,
    LineageVisualizer, set_global_tracker
)
from lineage.core import OperationType

# Import repositories
from repositories.equity_bars import EquityBarRepository
from repositories.option_bars import OptionBarRepository
from repositories.option_chains import OptionChainSnapshotRepository


def setup_lineage_system(base_path: str) -> tuple:
    """
    Set up the complete lineage tracking system.
    
    Returns:
        Tuple of (tracker, query_engine, visualizer)
    """
    # Create metadata storage
    lineage_path = Path(base_path) / "lineage_metadata"
    metadata_repo = LineageMetadataRepository(lineage_path)
    
    # Create tracker with persistent storage
    tracker = LineageTracker(storage_backend=metadata_repo)
    
    # Set as global tracker for decorator integration
    set_global_tracker(tracker)
    
    # Create query and visualization engines
    query_engine = LineageQueryEngine(tracker)
    visualizer = LineageVisualizer(tracker)
    
    return tracker, query_engine, visualizer


def create_sample_data():
    """Create sample data for demonstration."""
    # Sample equity bar data
    equity_data = pd.DataFrame({
        'datetime': [
            '20240101 09:30:00', '20240101 09:31:00', '20240101 09:32:00'
        ],
        'open': [100.0, 100.5, 101.0],
        'high': [100.5, 101.0, 101.5], 
        'low': [99.5, 100.0, 100.5],
        'close': [100.5, 101.0, 101.2],
        'volume': [1000, 1200, 800]
    })
    
    # Sample option chain data
    option_data = pd.DataFrame({
        'underlying': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'expiry': ['2024-01-19', '2024-01-19', '2024-01-19', '2024-01-19'],
        'strike': [100.0, 105.0, 100.0, 105.0],
        'right': ['C', 'C', 'P', 'P'],
        'bid': [2.5, 0.8, 1.2, 4.1],
        'ask': [2.7, 1.0, 1.4, 4.3],
        'volume': [100, 50, 75, 25],
        'implied_volatility': [0.25, 0.28, 0.23, 0.30]
    })
    
    return equity_data, option_data


def demonstrate_basic_tracking():
    """Demonstrate basic lineage tracking."""
    print("\\n=== Basic Lineage Tracking Demo ===")
    
    # Set up lineage system
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker, query_engine, visualizer = setup_lineage_system(temp_dir)
        
        # Create repositories
        equity_repo = EquityBarRepository(Path(temp_dir) / "data")
        option_chain_repo = OptionChainSnapshotRepository(Path(temp_dir) / "data")
        
        # Get sample data
        equity_data, option_data = create_sample_data()
        
        print(f"Created sample equity data: {len(equity_data)} rows")
        print(f"Created sample option data: {len(option_data)} rows")
        
        # Save data (automatically tracked)
        print("\\nSaving data to repositories...")
        equity_repo.save_daily_bars(equity_data, symbol='AAPL')
        option_chain_repo.save_chain_snapshot(option_data, underlying='AAPL', snapshot_date=date(2024, 1, 15))
        
        # Load data (automatically tracked)
        print("Loading data from repositories...")
        loaded_equity = equity_repo.load_symbol_data('AAPL')
        loaded_options = option_chain_repo.load_chain_snapshot('AAPL', date(2024, 1, 15))
        
        print(f"Loaded equity data: {len(loaded_equity)} rows")
        print(f"Loaded option data: {len(loaded_options)} rows")
        
        # Show lineage statistics
        print("\\n=== Lineage Statistics ===")
        stats = tracker.get_statistics()
        print(f"Total operations: {stats['total_operations']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Operation types: {stats['operation_types']}")
        print(f"Source types: {stats['source_types']}")


def demonstrate_lineage_analysis():
    """Demonstrate lineage analysis capabilities."""
    print("\\n=== Lineage Analysis Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker, query_engine, visualizer = setup_lineage_system(temp_dir)
        
        # Create more complex data flow
        equity_repo = EquityBarRepository(Path(temp_dir) / "data")
        option_repo = OptionBarRepository(Path(temp_dir) / "data")
        
        # Create sample data flow
        print("Creating complex data flow...")
        
        # 1. Save raw equity data
        equity_data, _ = create_sample_data()
        equity_repo.save_intraday_bars(equity_data, symbol='AAPL', bar_size='1 min')
        
        # 2. Load and aggregate to daily bars
        intraday_data = equity_repo.load_symbol_data('AAPL', bar_size='1 min')
        
        # Simulate daily aggregation
        daily_data = pd.DataFrame({
            'datetime': ['20240101'],
            'open': [intraday_data['open'].iloc[0]],
            'high': [intraday_data['high'].max()],
            'low': [intraday_data['low'].min()],
            'close': [intraday_data['close'].iloc[-1]],
            'volume': [intraday_data['volume'].sum()]
        })
        
        equity_repo.save_daily_bars(daily_data, symbol='AAPL')
        
        # 3. Create derived option data
        option_data = pd.DataFrame({
            'datetime': ['20240101 09:30:00'],
            'underlying': ['AAPL'],
            'expiry': [date(2024, 1, 19)],
            'strike': [100.0],
            'right': ['C'],
            'open': [2.5],
            'high': [2.7],
            'low': [2.3],
            'close': [2.6],
            'volume': [100]
        })
        
        option_repo.save_option_bars(
            option_data, 
            underlying='AAPL',
            expiry=date(2024, 1, 19),
            strike=100.0,
            right='C',
            bar_size='1 day'
        )
        
        print("\\n=== Query Analysis Results ===")
        
        # 1. Find data sources
        repo_sources = query_engine.find_data_sources(source_type="repository")
        print(f"Repository data sources found: {len(repo_sources)}")
        
        # 2. Analyze data volume flow
        volume_analysis = query_engine.analyze_data_volume_flow()
        print(f"Total operations: {volume_analysis['total_operations']}")
        print(f"Total records processed: {volume_analysis['total_records_in']}")
        
        # 3. Get performance stats
        perf_stats = query_engine.get_operation_performance_stats()
        print(f"Average operation duration: {perf_stats.get('avg_duration_ms', 0):.2f}ms")
        
        # 4. Find if any nodes are connected
        if len(repo_sources) >= 2:
            source_node = repo_sources[0]
            target_node = repo_sources[-1]
            
            flow = query_engine.trace_data_flow(source_node.node_id, target_node.node_id)
            if flow['path_found']:
                print(f"Data flow path found: {flow['path_length']} steps")
            else:
                print("No direct data flow path found between sources")


def demonstrate_impact_analysis():
    """Demonstrate impact analysis."""
    print("\\n=== Impact Analysis Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker, query_engine, visualizer = setup_lineage_system(temp_dir)
        
        # Create simple linear flow for clear impact analysis
        equity_repo = EquityBarRepository(Path(temp_dir) / "data")
        
        # Create source data
        equity_data, _ = create_sample_data()
        equity_repo.save_intraday_bars(equity_data, symbol='SOURCE', bar_size='1 min')
        
        # Create dependent data
        source_data = equity_repo.load_symbol_data('SOURCE', bar_size='1 min')
        
        # Simulate transformation and save as derived data
        derived_data = source_data.copy()
        derived_data['close'] = derived_data['close'] * 1.1  # Simulate some transformation
        equity_repo.save_daily_bars(derived_data, symbol='DERIVED')
        
        # Find source node
        repo_sources = query_engine.find_data_sources(source_type="repository")
        source_nodes = [node for node in repo_sources 
                       if 'SOURCE' in node.data_source.source_id]
        
        if source_nodes:
            source_node = source_nodes[0]
            print(f"Analyzing impact of changes to: {source_node.data_source.source_id}")
            
            # Perform impact analysis
            impact = visualizer.generate_impact_analysis(source_node.node_id)
            
            print(f"Total downstream nodes affected: {impact['total_downstream_nodes']}")
            print(f"Impact by source type: {impact['impact_by_source_type']}")
            print(f"Impact summary: {impact['impact_summary']}")
        else:
            print("No source nodes found for impact analysis")


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print("\\n=== Visualization Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker, query_engine, visualizer = setup_lineage_system(temp_dir)
        
        # Create some tracked operations
        equity_repo = EquityBarRepository(Path(temp_dir) / "data")
        equity_data, _ = create_sample_data()
        
        equity_repo.save_daily_bars(equity_data, symbol='VIZ_TEST')
        loaded_data = equity_repo.load_symbol_data('VIZ_TEST')
        
        print("\\n--- Graph Data Structure ---")
        graph_data = visualizer.generate_graph_data(include_operations=True)
        print(f"Nodes: {len(graph_data['nodes'])}")
        print(f"Edges: {len(graph_data['edges'])}")
        
        # Show sample node
        if graph_data['nodes']:
            sample_node = graph_data['nodes'][0]
            print(f"Sample node: {sample_node['label']} ({sample_node['type']})")
        
        print("\\n--- Timeline Data ---")
        timeline = visualizer.generate_timeline_data(hours_back=1.0)
        print(f"Timeline events: {len(timeline['events'])}")
        print(f"Event summary: {timeline['summary']}")
        
        print("\\n--- DOT Export (first 200 chars) ---")
        dot_output = visualizer.export_to_dot()
        print(dot_output[:200] + "..." if len(dot_output) > 200 else dot_output)
        
        print("\\n--- Summary Report ---")
        report = visualizer.generate_summary_report()
        print(report[:500] + "..." if len(report) > 500 else report)


def demonstrate_persistence():
    """Demonstrate metadata persistence."""
    print("\\n=== Persistence Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # First session - create and save lineage
        print("Session 1: Creating lineage data...")
        tracker1, _, _ = setup_lineage_system(temp_dir)
        
        equity_repo = EquityBarRepository(Path(temp_dir) / "data")
        equity_data, _ = create_sample_data()
        
        equity_repo.save_daily_bars(equity_data, symbol='PERSIST_TEST')
        equity_repo.load_symbol_data('PERSIST_TEST')
        
        session1_ops = len(tracker1.operations)
        session1_nodes = len(tracker1.nodes)
        print(f"Session 1 - Operations: {session1_ops}, Nodes: {session1_nodes}")
        
        # Second session - load persisted lineage
        print("\\nSession 2: Loading persisted lineage...")
        lineage_path = Path(temp_dir) / "lineage_metadata"
        metadata_repo = LineageMetadataRepository(lineage_path)
        
        # Load operations and nodes
        loaded_operations = metadata_repo.load_operations()
        loaded_nodes = metadata_repo.load_nodes()
        
        print(f"Session 2 - Loaded Operations: {len(loaded_operations)}")
        print(f"Session 2 - Loaded Nodes: {len(loaded_nodes)}")
        
        if loaded_operations:
            op = loaded_operations[0]
            print(f"Sample operation: {op.operation_type.value} at {op.timestamp}")
            print(f"  Duration: {op.duration_ms}ms")
            print(f"  Inputs: {len(op.inputs)}, Outputs: {len(op.outputs)}")
        
        # Show storage stats
        storage_stats = metadata_repo.get_storage_stats()
        print(f"\\nStorage stats: {storage_stats}")


def main():
    """Run all lineage demonstrations."""
    print("DATA LINEAGE TRACKING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    try:
        demonstrate_basic_tracking()
        demonstrate_lineage_analysis()
        demonstrate_impact_analysis() 
        demonstrate_visualization()
        demonstrate_persistence()
        
        print("\\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("\\nThe lineage tracking system provides:")
        print("- Automatic tracking of data operations")
        print("- Comprehensive lineage analysis")
        print("- Impact analysis for data changes")  
        print("- Multiple visualization formats")
        print("- Persistent metadata storage")
        
    except Exception as e:
        print(f"\\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()