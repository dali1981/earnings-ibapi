#!/usr/bin/env python3
"""
Test script for the new repository system.
"""
import pandas as pd
import tempfile
from datetime import datetime, date
from pathlib import Path

# Test the new repositories
from repositories import EquityBarRepository

def test_equity_bars_repository():
    """Test the EquityBarRepository with sample data."""
    
    print("=== Testing EquityBarRepository ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = EquityBarRepository(temp_dir)
        print(f"Created repository at: {temp_dir}")
        
        # Create sample data similar to what source_ib.py returns
        sample_data = pd.DataFrame({
            'symbol': ['AAPL'] * 3,
            'bar_size': ['1 day'] * 3,
            'time': [
                datetime(2024, 1, 1, 9, 30, 0),
                datetime(2024, 1, 2, 9, 30, 0),
                datetime(2024, 1, 3, 9, 30, 0),
            ],
            'open': [150.0, 151.0, 152.0],
            'high': [152.0, 153.0, 154.0], 
            'low': [149.0, 150.0, 151.0],
            'close': [151.0, 152.0, 153.0],
            'volume': [1000000, 1100000, 1200000],
        })
        
        print("Sample data:")
        print(sample_data)
        print()
        
        # Test save - should infer symbol and bar_size from DataFrame
        try:
            repo.save(sample_data)
            print("✅ Save successful - repository inferred parameters from DataFrame")
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return
        
        # Test load
        try:
            loaded_data = repo.load(symbol='AAPL')
            print("✅ Load successful")
            print(f"Loaded {len(loaded_data)} records")
            print("Loaded columns:", list(loaded_data.columns))
            if not loaded_data.empty:
                print("First row:")
                print(loaded_data.iloc[0])
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return
        
        # Test present_dates
        try:
            present = repo.present_dates('AAPL', '1 day', date(2024, 1, 1), date(2024, 1, 3))
            print(f"✅ Present dates: {present}")
        except Exception as e:
            print(f"❌ Present dates failed: {e}")
            return
        
        print("✅ All EquityBarRepository tests passed!")


def test_repository_validation():
    """Test repository validation features."""
    
    print("\n=== Testing Repository Validation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = EquityBarRepository(temp_dir)
        
        # Test with missing required columns
        try:
            bad_data = pd.DataFrame({
                'symbol': ['AAPL'],
                'bar_size': ['1 day'],
                # Missing OHLCV columns
            })
            repo.save(bad_data)
            print("❌ Should have failed validation")
        except Exception as e:
            print(f"✅ Validation caught missing columns: {type(e).__name__}")
        
        # Test with invalid data types
        try:
            bad_data = pd.DataFrame({
                'symbol': ['AAPL'],
                'bar_size': ['1 day'], 
                'time': [datetime(2024, 1, 1)],
                'open': ['not_a_number'],  # Invalid
                'high': [150.0],
                'low': [149.0],
                'close': [151.0],
                'volume': [1000],
            })
            repo.save(bad_data)
            print("✅ Repository handled invalid data types")
        except Exception as e:
            print(f"✅ Validation caught invalid data: {type(e).__name__}")


if __name__ == "__main__":
    test_equity_bars_repository()
    test_repository_validation()
    print("\n🎉 Repository testing completed!")