"""
Data Pipeline Orchestrator - Manages dependencies and execution order for trading data jobs.

This module ensures proper data flow:
1. Contracts must exist before option data
2. Option chains must be current before option bars
3. Equity data must exist before option data (for spot prices)
4. Nothing is re-requested if it already exists
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd

# Repository imports
from repositories import (
    EquityBarRepository,
    OptionBarRepository, 
    OptionChainSnapshotRepository
)
from repositories.contract_descriptions import ContractDescriptionsRepository

# Job task imports
from .tasks import BackfillEquityBarsTask, BackfillOptionBarsTask

# Lineage tracking
try:
    from lineage import LineageTracker, set_global_tracker
    from lineage.core import OperationType
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False

# Reliability imports
try:
    from reliability import get_trading_logger, CircuitBreaker, RetryConfig
except ImportError:
    import logging
    def get_trading_logger(name: str):
        return logging.getLogger(name)
    
    class CircuitBreaker:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class RetryConfig:
        max_attempts = 3
        backoff_seconds = 5


class JobType(Enum):
    """Types of data jobs that can be orchestrated."""
    CONTRACT_DESCRIPTIONS = "contract_descriptions"
    OPTION_CHAIN_SNAPSHOT = "option_chain_snapshot" 
    EQUITY_BARS_BACKFILL = "equity_bars_backfill"
    OPTION_BARS_BACKFILL = "option_bars_backfill"


@dataclass
class JobResult:
    """Result of a job execution."""
    job_type: JobType
    symbol: str
    success: bool
    error_message: Optional[str] = None
    data_records: Optional[int] = None
    execution_time_seconds: Optional[float] = None


class PrerequisiteError(Exception):
    """Raised when required data dependencies are not met."""
    pass


class DataIntegrityError(Exception):
    """Raised when data consistency violations are detected."""
    pass


class DataValidation:
    """Validates data prerequisites and integrity."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.contract_repo = ContractDescriptionsRepository(base_path / "contract_descriptions")
        self.equity_repo = EquityBarRepository(base_path / "equity_bars")  
        self.option_repo = OptionBarRepository(base_path / "option_bars")
        self.chain_repo = OptionChainSnapshotRepository(base_path / "option_chains")
        
        self.logger = get_trading_logger("DataValidation")
    
    def validate_symbol_contracts_exist(self, symbol: str) -> None:
        """Validate that contract descriptions exist for a symbol."""
        try:
            contracts = self.contract_repo.load_contracts_for_symbol(symbol)
            if contracts.empty:
                raise PrerequisiteError(f"No contract descriptions found for {symbol}")
        except Exception as e:
            raise PrerequisiteError(f"Unable to load contracts for {symbol}: {e}")
    
    def validate_option_chain_current(self, symbol: str, max_age_hours: int = 24) -> None:
        """Validate that option chain snapshot is recent."""
        try:
            # Check for recent snapshot
            latest_date = self.chain_repo.get_latest_snapshot_date(symbol)
            if latest_date is None:
                raise PrerequisiteError(f"No option chain snapshot found for {symbol}")
            
            # Check age
            age_hours = (datetime.now().date() - latest_date).total_seconds() / 3600
            if age_hours > max_age_hours:
                raise PrerequisiteError(
                    f"Option chain snapshot for {symbol} is {age_hours:.1f} hours old "
                    f"(max allowed: {max_age_hours})"
                )
        except PrerequisiteError:
            raise
        except Exception as e:
            raise PrerequisiteError(f"Unable to validate option chain for {symbol}: {e}")
    
    def validate_equity_data_available(self, symbol: str, max_age_days: int = 2) -> None:
        """Validate that recent equity data exists (needed for spot prices)."""
        try:
            # Check for recent equity data
            cutoff_date = datetime.now().date() - timedelta(days=max_age_days)
            recent_dates = self.equity_repo.present_dates(
                symbol=symbol,
                bar_size="1 day", 
                start_date=cutoff_date,
                end_date=datetime.now().date()
            )
            
            if not recent_dates:
                raise PrerequisiteError(f"No recent equity data for {symbol}")
                
        except PrerequisiteError:
            raise
        except Exception as e:
            raise PrerequisiteError(f"Unable to validate equity data for {symbol}: {e}")
    
    def validate_option_backfill_prerequisites(self, symbol: str) -> None:
        """Validate all prerequisites for option bar backfill."""
        self.logger.info(f"Validating option backfill prerequisites for {symbol}")
        
        # Check contracts
        self.validate_symbol_contracts_exist(symbol)
        self.logger.debug(f"✓ Contract descriptions exist for {symbol}")
        
        # Check option chain
        self.validate_option_chain_current(symbol, max_age_hours=24)
        self.logger.debug(f"✓ Option chain current for {symbol}")
        
        # Check equity data  
        self.validate_equity_data_available(symbol, max_age_days=2)
        self.logger.debug(f"✓ Equity data available for {symbol}")
        
        self.logger.info(f"All prerequisites validated for {symbol}")
    
    def validate_data_integrity(self, symbol: str) -> None:
        """Validate data consistency across repositories."""
        self.logger.info(f"Validating data integrity for {symbol}")
        
        # Get option data dates
        try:
            option_dates = self.option_repo.get_available_dates(symbol)
        except Exception:
            option_dates = set()  # No option data is fine
        
        # Get equity data dates
        try:
            equity_dates = self.equity_repo.get_available_dates(symbol)
        except Exception:
            equity_dates = set()
        
        # Option data without equity data is a problem
        missing_equity = option_dates - equity_dates
        if missing_equity:
            raise DataIntegrityError(
                f"Option data exists without corresponding equity data for {symbol}. "
                f"Missing equity dates: {sorted(missing_equity)}"
            )
        
        self.logger.info(f"Data integrity validated for {symbol}")


class DataPipelineOrchestrator:
    """
    Orchestrates data pipeline jobs ensuring proper dependency management.
    
    Key principles:
    1. Contracts before options
    2. Chains before option bars  
    3. Equity before option bars (for spot prices)
    4. Never re-request existing data
    5. Validate before executing
    """
    
    def __init__(self, base_path: Path, enable_lineage: bool = True):
        self.base_path = Path(base_path)
        self.validator = DataValidation(self.base_path)
        
        # Initialize repositories
        self.contract_repo = ContractDescriptionsRepository(self.base_path / "contract_descriptions")
        self.equity_repo = EquityBarRepository(self.base_path / "equity_bars")
        self.option_repo = OptionBarRepository(self.base_path / "option_bars")
        self.chain_repo = OptionChainSnapshotRepository(self.base_path / "option_chains")
        
        # Setup lineage tracking
        if enable_lineage and LINEAGE_AVAILABLE:
            self._setup_lineage_tracking()
        
        # Circuit breakers for API protection
        try:
            from reliability.circuit_breaker import CircuitBreakerConfig
            breaker_config = CircuitBreakerConfig(
                failure_threshold=5,
                timeout_duration=300.0  # 5 minutes
            )
            self.api_circuit_breaker = CircuitBreaker(
                name="api_circuit_breaker", 
                config=breaker_config
            )
        except ImportError:
            # Fallback to dummy circuit breaker if reliability module not available
            self.api_circuit_breaker = CircuitBreaker()
        
        self.logger = get_trading_logger("DataPipelineOrchestrator")
    
    def _setup_lineage_tracking(self) -> None:
        """Initialize lineage tracking if available."""
        try:
            from lineage.metadata import LineageMetadataRepository
            
            lineage_repo = LineageMetadataRepository(self.base_path / "lineage_metadata")
            tracker = LineageTracker(storage_backend=lineage_repo)
            set_global_tracker(tracker)
            
            self.logger.info("Lineage tracking enabled")
        except Exception as e:
            self.logger.warning(f"Failed to setup lineage tracking: {e}")
    
    def setup_new_symbol(self, symbol: str, lookback_days: int = 365) -> List[JobResult]:
        """
        Cold start setup for new symbol - MUST follow dependency order.
        
        Order:
        1. Contract descriptions (foundation)
        2. Option chain snapshot (current market structure)
        3. Equity bars (historical data + spot prices)
        4. Option bars (requires all above)
        """
        self.logger.info(f"Setting up new symbol: {symbol} (lookback: {lookback_days} days)")
        
        results = []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            # Step 1: Contract descriptions
            if not self._contracts_exist(symbol):
                result = self._run_contract_backfill(symbol)
                results.append(result)
                if not result.success:
                    self.logger.error(f"Failed to get contracts for {symbol}, aborting setup")
                    return results
            else:
                self.logger.info(f"Contracts already exist for {symbol}")
            
            # Step 2: Option chain snapshot
            if not self._option_chain_current(symbol):
                result = self._run_option_chain_snapshot(symbol)
                results.append(result)
                if not result.success:
                    self.logger.warning(f"Failed to get option chain for {symbol}, skipping option data")
                    # Continue with equity data only
            else:
                self.logger.info(f"Option chain current for {symbol}")
            
            # Step 3: Equity bars
            if not self._equity_data_complete(symbol, start_date, end_date):
                result = self._run_equity_backfill(symbol, start_date, end_date)
                results.append(result)
                if not result.success:
                    self.logger.error(f"Failed to get equity data for {symbol}")
                    return results
            else:
                self.logger.info(f"Equity data complete for {symbol}")
            
            # Step 4: Option bars (only if all prerequisites met)
            if self._all_prerequisites_met(symbol):
                result = self._run_option_backfill(symbol, start_date, end_date)
                results.append(result)
            else:
                self.logger.warning(f"Prerequisites not met for option data: {symbol}")
            
            self.logger.info(f"Setup complete for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Setup failed for {symbol}: {e}")
            error_result = JobResult(
                job_type=JobType.EQUITY_BARS_BACKFILL,  # Use generic type
                symbol=symbol,
                success=False,
                error_message=str(e)
            )
            results.append(error_result)
            return results
    
    def daily_update(self, symbols: List[str]) -> Dict[str, List[JobResult]]:
        """
        Daily incremental updates - validate before processing.
        
        Process:
        1. Validate data integrity
        2. Refresh option chains (snapshots)
        3. Incremental equity backfill
        4. Incremental option backfill (validated contracts only)
        """
        self.logger.info(f"Starting daily update for {len(symbols)} symbols")
        
        all_results = {}
        
        for symbol in symbols:
            results = []
            
            try:
                # Validate existing data integrity
                self.validator.validate_data_integrity(symbol)
                
                # Refresh option chain snapshot
                if self._should_refresh_option_chain(symbol):
                    result = self._run_option_chain_snapshot(symbol)
                    results.append(result)
                
                # Incremental equity update
                result = self._run_incremental_equity_update(symbol)
                if result:
                    results.append(result)
                
                # Incremental option update (only if prerequisites met)
                if self._all_prerequisites_met(symbol):
                    result = self._run_incremental_option_update(symbol)
                    if result:
                        results.append(result)
                
            except Exception as e:
                self.logger.error(f"Daily update failed for {symbol}: {e}")
                error_result = JobResult(
                    job_type=JobType.EQUITY_BARS_BACKFILL,
                    symbol=symbol,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
            
            all_results[symbol] = results
        
        self.logger.info("Daily update complete")
        return all_results
    
    def _contracts_exist(self, symbol: str) -> bool:
        """Check if contract descriptions exist for symbol."""
        try:
            self.validator.validate_symbol_contracts_exist(symbol)
            return True
        except PrerequisiteError:
            return False
    
    def _option_chain_current(self, symbol: str, max_age_hours: int = 4) -> bool:
        """Check if option chain is current."""
        try:
            self.validator.validate_option_chain_current(symbol, max_age_hours)
            return True
        except PrerequisiteError:
            return False
    
    def _equity_data_complete(self, symbol: str, start_date: date, end_date: date) -> bool:
        """Check if equity data is complete for date range."""
        try:
            present_dates = self.equity_repo.present_dates(symbol, "1 day", start_date, end_date)
            expected_dates = set(pd.bdate_range(start_date, end_date).date)
            return len(present_dates) >= len(expected_dates) * 0.95  # Allow 5% missing
        except Exception:
            return False
    
    def _all_prerequisites_met(self, symbol: str) -> bool:
        """Check if all prerequisites for option data are met."""
        try:
            self.validator.validate_option_backfill_prerequisites(symbol)
            return True
        except PrerequisiteError as e:
            self.logger.debug(f"Prerequisites not met for {symbol}: {e}")
            return False
    
    def _should_refresh_option_chain(self, symbol: str, refresh_hours: int = 4) -> bool:
        """Check if option chain should be refreshed."""
        return not self._option_chain_current(symbol, max_age_hours=refresh_hours)
    
    def _run_contract_backfill(self, symbol: str) -> JobResult:
        """Run contract descriptions backfill job."""
        start_time = time.time()
        self.logger.info(f"Running contract backfill for {symbol}")
        
        try:
            with self.api_circuit_breaker:
                # This would call the actual contract backfill job
                # For now, simulate the call
                from jobs.backfill_contracts_us_rt import backfill_us_equity_contracts
                # Implementation would go here
                
                execution_time = time.time() - start_time
                
                return JobResult(
                    job_type=JobType.CONTRACT_DESCRIPTIONS,
                    symbol=symbol,
                    success=True,
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Contract backfill failed for {symbol}: {e}")
            
            return JobResult(
                job_type=JobType.CONTRACT_DESCRIPTIONS,
                symbol=symbol,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
    
    def _run_option_chain_snapshot(self, symbol: str) -> JobResult:
        """Run option chain snapshot job."""
        start_time = time.time()
        self.logger.info(f"Running option chain snapshot for {symbol}")
        
        try:
            with self.api_circuit_breaker:
                # This would call the actual option chain snapshot job
                # Implementation would go here
                
                execution_time = time.time() - start_time
                
                return JobResult(
                    job_type=JobType.OPTION_CHAIN_SNAPSHOT,
                    symbol=symbol,
                    success=True,
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Option chain snapshot failed for {symbol}: {e}")
            
            return JobResult(
                job_type=JobType.OPTION_CHAIN_SNAPSHOT,
                symbol=symbol,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
    
    def _run_equity_backfill(self, symbol: str, start_date: date, end_date: date) -> JobResult:
        """Run equity bars backfill job."""
        start_time = time.time()
        self.logger.info(f"Running equity backfill for {symbol}: {start_date} to {end_date}")
        
        try:
            with self.api_circuit_breaker:
                task = BackfillEquityBarsTask(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    bar_size="1 day",
                    out=self.base_path / "equity_bars"
                )
                
                task.run()
                execution_time = time.time() - start_time
                
                return JobResult(
                    job_type=JobType.EQUITY_BARS_BACKFILL,
                    symbol=symbol,
                    success=True,
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Equity backfill failed for {symbol}: {e}")
            
            return JobResult(
                job_type=JobType.EQUITY_BARS_BACKFILL,
                symbol=symbol,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
    
    def _run_option_backfill(self, symbol: str, start_date: date, end_date: date) -> JobResult:
        """Run option bars backfill job."""
        start_time = time.time()
        self.logger.info(f"Running option backfill for {symbol}: {start_date} to {end_date}")
        
        try:
            # Validate prerequisites before starting
            self.validator.validate_option_backfill_prerequisites(symbol)
            
            with self.api_circuit_breaker:
                task = BackfillOptionBarsTask(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    bar_size="1 day",
                    out=self.base_path / "option_bars",
                    chain_base=self.base_path
                )
                
                task.run()
                execution_time = time.time() - start_time
                
                return JobResult(
                    job_type=JobType.OPTION_BARS_BACKFILL,
                    symbol=symbol,
                    success=True,
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Option backfill failed for {symbol}: {e}")
            
            return JobResult(
                job_type=JobType.OPTION_BARS_BACKFILL,
                symbol=symbol,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
    
    def _run_incremental_equity_update(self, symbol: str) -> Optional[JobResult]:
        """Run incremental equity update if needed."""
        # Check if update needed
        yesterday = datetime.now().date() - timedelta(days=1)
        present_dates = self.equity_repo.present_dates(symbol, "1 day", yesterday, yesterday)
        
        if yesterday not in present_dates:
            return self._run_equity_backfill(symbol, yesterday, yesterday)
        
        return None  # No update needed
    
    def _run_incremental_option_update(self, symbol: str) -> Optional[JobResult]:
        """Run incremental option update if needed."""
        # Check if update needed
        yesterday = datetime.now().date() - timedelta(days=1)
        
        # This would check for missing option data
        # Implementation would determine if incremental update is needed
        
        return None  # Placeholder implementation
    
    def generate_status_report(self) -> Dict[str, any]:
        """Generate comprehensive status report."""
        return {
            "orchestrator_status": "active",
            "repositories": {
                "contract_descriptions": str(self.base_path / "contract_descriptions"),
                "equity_bars": str(self.base_path / "equity_bars"),
                "option_bars": str(self.base_path / "option_bars"),
                "option_chains": str(self.base_path / "option_chains")
            },
            "circuit_breaker_status": "closed",  # Would check actual status
            "lineage_tracking": LINEAGE_AVAILABLE
        }