"""
Database and repository mock factory for comprehensive testing.
"""

import json
import random
import tempfile
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class MockParquetRepository:
    """Mock implementation of Parquet-based repositories for testing."""

    def __init__(self, base_path: Path, repository_type: str = "generic"):
        self.base_path = Path(base_path)
        self.repository_type = repository_type
        self.data_store = {}  # In-memory storage for test data
        self.metadata_store = {}
        self.operation_log = []  # Track all operations for testing
        self.error_simulation = {}
        self.performance_simulation = {}

    def save(
        self,
        data: pd.DataFrame,
        symbol: str,
        *args,
        **kwargs
    ) -> bool:
        """Mock save operation with configurable behavior."""
        operation = {
            'type': 'save',
            'symbol': symbol,
            'rows': len(data),
            'timestamp': datetime.now(),
            'args': args,
            'kwargs': kwargs
        }
        self.operation_log.append(operation)

        # Simulate performance delays
        if symbol in self.performance_simulation:
            import time
            time.sleep(self.performance_simulation[symbol])

        # Simulate errors
        if symbol in self.error_simulation:
            error_config = self.error_simulation[symbol]
            if random.random() < error_config.get('rate', 0.1):
                raise Exception(error_config.get('message', 'Simulated save error'))

        # Store data in memory
        key = self._generate_key(symbol, *args, **kwargs)
        self.data_store[key] = data.copy()
        
        # Update metadata
        self.metadata_store[symbol] = {
            'last_updated': datetime.now(),
            'row_count': len(data),
            'columns': list(data.columns)
        }

        return True

    def load(
        self,
        symbol: str,
        *args,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Mock load operation."""
        operation = {
            'type': 'load',
            'symbol': symbol,
            'timestamp': datetime.now(),
            'args': args,
            'kwargs': kwargs
        }
        self.operation_log.append(operation)

        # Simulate errors
        if symbol in self.error_simulation:
            error_config = self.error_simulation[symbol]
            if random.random() < error_config.get('rate', 0.1):
                raise Exception(error_config.get('message', 'Simulated load error'))

        key = self._generate_key(symbol, *args, **kwargs)
        return self.data_store.get(key)

    def present_dates(
        self,
        symbol: str,
        bar_size: str,
        start_date: date,
        end_date: date
    ) -> List[date]:
        """Mock present_dates operation."""
        operation = {
            'type': 'present_dates',
            'symbol': symbol,
            'bar_size': bar_size,
            'start_date': start_date,
            'end_date': end_date,
            'timestamp': datetime.now()
        }
        self.operation_log.append(operation)

        # Generate mock present dates
        if symbol not in self.data_store:
            return []

        # Simulate some missing dates for testing
        all_dates = pd.date_range(start_date, end_date, freq='D').date.tolist()
        
        # Randomly remove some dates to simulate gaps
        missing_rate = 0.1  # 10% missing
        present_dates = [
            d for d in all_dates 
            if random.random() > missing_rate
        ]
        
        return present_dates

    def get_symbols(self) -> List[str]:
        """Get all symbols in the repository."""
        symbols = set()
        for key in self.data_store.keys():
            symbol = key.split('|')[0]  # Assuming key format: symbol|...
            symbols.add(symbol)
        return list(symbols)

    def get_date_range(self, symbol: str) -> Optional[Tuple[date, date]]:
        """Get date range for a symbol."""
        if symbol not in self.metadata_store:
            return None
        
        # Mock date range
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        return start_date, end_date

    def _generate_key(self, symbol: str, *args, **kwargs) -> str:
        """Generate a unique key for data storage."""
        parts = [symbol]
        parts.extend(str(arg) for arg in args)
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "|".join(parts)

    def set_error_simulation(self, symbol: str, rate: float = 0.1, message: str = "Simulated error"):
        """Configure error simulation for a symbol."""
        self.error_simulation[symbol] = {'rate': rate, 'message': message}

    def set_performance_simulation(self, symbol: str, delay: float = 1.0):
        """Configure performance simulation (delays) for a symbol."""
        self.performance_simulation[symbol] = delay

    def clear_error_simulation(self, symbol: str = None):
        """Clear error simulation."""
        if symbol:
            self.error_simulation.pop(symbol, None)
        else:
            self.error_simulation.clear()

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get all operations performed on this repository."""
        return self.operation_log.copy()

    def clear_operation_log(self):
        """Clear the operation log."""
        self.operation_log.clear()

    def get_data_keys(self) -> List[str]:
        """Get all data keys stored."""
        return list(self.data_store.keys())

    def clear_all_data(self):
        """Clear all stored data."""
        self.data_store.clear()
        self.metadata_store.clear()
        self.operation_log.clear()


class DatabaseMockFactory:
    """Factory for creating various database and repository mocks."""

    def __init__(self, temp_dir: Path = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.repositories = {}
        self.global_error_rate = 0.0
        self.global_delay = 0.0

    def create_equity_repository_mock(
        self,
        preload_symbols: List[str] = None,
        error_rate: float = 0.0
    ) -> MockParquetRepository:
        """Create a mock equity bar repository."""
        repo = MockParquetRepository(
            self.temp_dir / "equity_bars",
            "equity_bars"
        )

        if preload_symbols:
            self._preload_equity_data(repo, preload_symbols)

        if error_rate > 0:
            for symbol in (preload_symbols or []):
                repo.set_error_simulation(symbol, error_rate)

        self.repositories['equity_bars'] = repo
        return repo

    def create_option_repository_mock(
        self,
        preload_symbols: List[str] = None,
        error_rate: float = 0.0
    ) -> MockParquetRepository:
        """Create a mock option bar repository."""
        repo = MockParquetRepository(
            self.temp_dir / "option_bars",
            "option_bars"
        )

        if preload_symbols:
            self._preload_option_data(repo, preload_symbols)

        if error_rate > 0:
            for symbol in (preload_symbols or []):
                repo.set_error_simulation(symbol, error_rate)

        self.repositories['option_bars'] = repo
        return repo

    def create_option_chains_repository_mock(
        self,
        preload_symbols: List[str] = None,
        error_rate: float = 0.0
    ) -> MockParquetRepository:
        """Create a mock option chains repository."""
        repo = MockParquetRepository(
            self.temp_dir / "option_chains",
            "option_chains"
        )

        if preload_symbols:
            self._preload_option_chains_data(repo, preload_symbols)

        if error_rate > 0:
            for symbol in (preload_symbols or []):
                repo.set_error_simulation(symbol, error_rate)

        self.repositories['option_chains'] = repo
        return repo

    def create_earnings_repository_mock(
        self,
        preload_symbols: List[str] = None,
        error_rate: float = 0.0
    ) -> MockParquetRepository:
        """Create a mock earnings repository."""
        repo = MockParquetRepository(
            self.temp_dir / "earnings",
            "earnings"
        )

        if preload_symbols:
            self._preload_earnings_data(repo, preload_symbols)

        if error_rate > 0:
            for symbol in (preload_symbols or []):
                repo.set_error_simulation(symbol, error_rate)

        self.repositories['earnings'] = repo
        return repo

    def create_contract_descriptions_repository_mock(
        self,
        preload_symbols: List[str] = None,
        error_rate: float = 0.0
    ) -> MockParquetRepository:
        """Create a mock contract descriptions repository."""
        repo = MockParquetRepository(
            self.temp_dir / "contract_descriptions",
            "contract_descriptions"
        )

        if preload_symbols:
            self._preload_contract_descriptions_data(repo, preload_symbols)

        if error_rate > 0:
            for symbol in (preload_symbols or []):
                repo.set_error_simulation(symbol, error_rate)

        self.repositories['contract_descriptions'] = repo
        return repo

    def create_all_repositories(
        self,
        preload_symbols: List[str] = None,
        error_rate: float = 0.0
    ) -> Dict[str, MockParquetRepository]:
        """Create all repository mocks."""
        return {
            'equity_bars': self.create_equity_repository_mock(preload_symbols, error_rate),
            'option_bars': self.create_option_repository_mock(preload_symbols, error_rate),
            'option_chains': self.create_option_chains_repository_mock(preload_symbols, error_rate),
            'earnings': self.create_earnings_repository_mock(preload_symbols, error_rate),
            'contract_descriptions': self.create_contract_descriptions_repository_mock(preload_symbols, error_rate)
        }

    def _preload_equity_data(self, repo: MockParquetRepository, symbols: List[str]):
        """Preload equity data into repository."""
        from tests.conftest import MarketDataGenerator
        generator = MarketDataGenerator()
        
        for symbol in symbols:
            data = generator.generate_equity_bars(symbol, 30, "1 day")
            repo.save(data, symbol, "1 day", "TRADES")

    def _preload_option_data(self, repo: MockParquetRepository, symbols: List[str]):
        """Preload option bar data into repository."""
        from tests.conftest import MarketDataGenerator
        generator = MarketDataGenerator()
        
        for symbol in symbols:
            # Generate option data for multiple strikes
            strikes = [140, 145, 150, 155, 160]
            for strike in strikes:
                for right in ["C", "P"]:
                    contract_key = f"{symbol}_{strike}_{right}"
                    data = pd.DataFrame({
                        'time': pd.date_range('2024-01-01 09:30', periods=100, freq='1min'),
                        'open': [5.0 + i * 0.1 for i in range(100)],
                        'high': [5.1 + i * 0.1 for i in range(100)],
                        'low': [4.9 + i * 0.1 for i in range(100)],
                        'close': [5.0 + i * 0.1 + 0.05 for i in range(100)],
                        'volume': [random.randint(10, 1000) for _ in range(100)]
                    })
                    repo.save(data, contract_key, "1 min")

    def _preload_option_chains_data(self, repo: MockParquetRepository, symbols: List[str]):
        """Preload option chains data into repository."""
        from tests.conftest import MarketDataGenerator
        generator = MarketDataGenerator()
        
        for symbol in symbols:
            expiries = [date.today() + timedelta(days=30), date.today() + timedelta(days=60)]
            data = generator.generate_option_chain(symbol, expiries)
            repo.save(data, symbol, datetime.now())

    def _preload_earnings_data(self, repo: MockParquetRepository, symbols: List[str]):
        """Preload earnings data into repository."""
        from tests.conftest import MarketDataGenerator
        generator = MarketDataGenerator()
        
        data = generator.generate_earnings_calendar(symbols)
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol]
            repo.save(symbol_data, symbol)

    def _preload_contract_descriptions_data(self, repo: MockParquetRepository, symbols: List[str]):
        """Preload contract descriptions data into repository."""
        for symbol in symbols:
            data = pd.DataFrame([{
                'symbol': symbol,
                'secType': 'STK',
                'exchange': 'SMART',
                'currency': 'USD',
                'conId': random.randint(100000, 999999),
                'primaryExchange': 'NASDAQ' if symbol in ['AAPL', 'MSFT', 'GOOGL'] else 'NYSE',
                'marketName': f'{symbol} Stock',
                'longName': f'{symbol} Corporation',
                'category': 'Stock',
                'subcategory': 'Common Stock'
            }])
            repo.save(data, symbol)

    def simulate_concurrent_access(
        self,
        repo: MockParquetRepository,
        symbols: List[str],
        operations: int = 100,
        threads: int = 5
    ) -> List[Dict[str, Any]]:
        """Simulate concurrent repository access for testing."""
        results = []
        lock = threading.Lock()

        def worker():
            for _ in range(operations // threads):
                symbol = random.choice(symbols)
                operation = random.choice(['save', 'load', 'present_dates'])
                
                try:
                    if operation == 'save':
                        data = pd.DataFrame({
                            'date': ['20240101'],
                            'open': [150.0],
                            'high': [151.0],
                            'low': [149.0],
                            'close': [150.5],
                            'volume': [1000000]
                        })
                        result = repo.save(data, symbol, "1 day", "TRADES")
                        status = 'success'
                    elif operation == 'load':
                        result = repo.load(symbol, "1 day", "TRADES")
                        status = 'success'
                    else:  # present_dates
                        result = repo.present_dates(
                            symbol, "1 day", 
                            date.today() - timedelta(days=30),
                            date.today()
                        )
                        status = 'success'
                except Exception as e:
                    result = str(e)
                    status = 'error'

                with lock:
                    results.append({
                        'thread': threading.current_thread().name,
                        'operation': operation,
                        'symbol': symbol,
                        'status': status,
                        'result': result,
                        'timestamp': datetime.now()
                    })

        threads_list = []
        for i in range(threads):
            thread = threading.Thread(target=worker, name=f'worker_{i}')
            threads_list.append(thread)
            thread.start()

        for thread in threads_list:
            thread.join()

        return results

    def get_all_operations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all operations from all repositories."""
        operations = {}
        for repo_name, repo in self.repositories.items():
            operations[repo_name] = repo.get_operation_log()
        return operations

    def clear_all_data(self):
        """Clear all data from all repositories."""
        for repo in self.repositories.values():
            repo.clear_all_data()

    def set_global_error_rate(self, rate: float):
        """Set error rate for all repositories."""
        self.global_error_rate = rate
        for repo in self.repositories.values():
            for symbol in repo.get_symbols():
                repo.set_error_simulation(symbol, rate)

    def set_global_performance_delay(self, delay: float):
        """Set performance delay for all repositories."""
        self.global_delay = delay
        for repo in self.repositories.values():
            for symbol in repo.get_symbols():
                repo.set_performance_simulation(symbol, delay)