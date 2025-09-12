# REFACTORING PLAN - Trading Project / Earnings IBAPI

## 🚀 IMPLEMENTATION STATUS

| Phase | Status | Completion Date | Key Achievements |
|-------|--------|-----------------|------------------|
| **Phase 1: Critical Fixes** | ✅ **COMPLETED** | 2024-12-26 | Fixed Jedi import, consolidated exception handling |
| **Phase 2: Code Cleanup** | ✅ **COMPLETED** | 2024-12-26 | Root directory cleanup, archive structure created |
| **Phase 3: Architecture** | ⏳ **PENDING** | TBD | Shared libraries, repository enhancements planned |
| **Phase 4: Long-term** | ⏳ **PENDING** | TBD | Performance optimization, testing improvements |

### 🎯 Current Progress: **Phase 2 Complete (50% of planned work)**

**✅ Completed Tasks:**
- [x] Fix Jedi import issue (Phase 1.1)
- [x] Consolidate exception handling (Phase 1.2) 
- [x] Clean root directory (Phase 2.1)
- [x] Create archive structure (Phase 2.1)
- [x] Update import references (Phase 2.1)
- [x] Create historical tasks foundation (Phase 2.2)

**📋 Next Priority:** Phase 3 - Architecture Improvements
- Create shared libraries (`common/exceptions.py`, `common/schemas/`, etc.)
- Repository pattern enhancement with protocols and mixins
- Configuration consolidation

---

## Executive Summary

This document provides a detailed implementation plan for the refactoring recommendations outlined in `CODE_REVIEW.md`. The plan is structured in four phases with specific implementation steps, test cases, edge cases, and rollback strategies to ensure safe execution without disrupting production operations.

## Implementation Strategy

### Core Principles
1. **Incremental Changes**: Make small, testable changes to minimize risk
2. **Backward Compatibility**: Maintain existing APIs during transitions
3. **Test-Driven**: Write tests before refactoring where possible
4. **Rollback Ready**: Ensure each change can be easily reverted
5. **Documentation**: Update docs concurrently with changes

### Risk Mitigation
- All changes tested in development environment first
- Feature flags for new implementations
- Comprehensive test coverage before deployment
- Staged rollouts with monitoring

---

## PHASE 1: CRITICAL FIXES ✅ **COMPLETED**

**Status:** ✅ Completed on 2024-12-26  
**Branch:** `refactoring/phase1-critical-fixes`  
**Impact:** Critical stability and maintainability improvements

### 🔴 Task 1.1: Fix Jedi Import Issue

**File**: `common/repositories/base.py:16`
**Impact**: HIGH - Critical dependency fix
**Time Estimate**: 30 minutes

#### Implementation Steps:
```python
# BEFORE (line 16):
from jedi.inference.value.iterable import Sequence

# AFTER:
from typing import Sequence
```

#### Test Cases:
1. **Import Test**:
   ```python
   def test_sequence_import():
       from common.repositories.base import BaseRepository
       # Should not raise ImportError
       assert BaseRepository is not None
   ```

2. **Type Annotation Test**:
   ```python
   def test_sequence_type_usage():
       from common.repositories.base import BaseRepository
       repo = TestRepository("test_data")
       # Verify Sequence type works correctly in all usages
   ```

#### Edge Cases:
- **Development vs Production**: Jedi may be installed in dev but not prod
- **Type Checker Compatibility**: Ensure mypy/pyright still pass
- **Python Version Compatibility**: Test on Python 3.8+ (typing.Sequence available)

#### Verification:
```bash
# Run full test suite
make test-unit
# Check import specifically
python -c "from common.repositories.base import BaseRepository; print('✓ Import successful')"
# Type checking
mypy common/repositories/base.py
```

#### Rollback Plan:
- Simple revert of single line change
- No dependencies on this change

---

### 🔴 Task 1.2: Consolidate Exception Handling

**Files**: 5 repository files with duplicate exception patterns
**Impact**: MEDIUM - Code deduplication and maintainability
**Time Estimate**: 4 hours

#### Current Duplication Pattern:
```python
# Found in: base.py, option_chains.py, equity_bars.py, contract_descriptions.py, option_bars.py
try:
    from common.reliability import DataValidationException
except ImportError:
    class DataValidationException(ValueError):
        def __init__(self, message: str, field: str = None, **kwargs):
            super().__init__(message)
            self.field = field
            self.details = kwargs
```

#### Implementation Steps:

**Step 1.2.1**: Create `common/exceptions.py`
```python
"""
Common exception classes for the trading platform.
Centralizes all custom exceptions to avoid duplication.
"""

class DataValidationException(ValueError):
    """Raised when data validation fails in repositories."""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message)
        self.field = field
        self.details = kwargs
        
    def __repr__(self) -> str:
        details = f", field={self.field}" if self.field else ""
        return f"DataValidationException('{self.args[0]}'{details})"

class RepositoryException(Exception):
    """Base exception for repository operations."""
    pass

class DataIntegrityException(RepositoryException):
    """Raised when data integrity constraints are violated."""
    pass

class ConfigurationException(Exception):
    """Raised when configuration is invalid or missing."""
    pass

class APIException(Exception):
    """Base exception for API-related errors."""
    pass

class IBAPIException(APIException):
    """Interactive Brokers API specific exceptions."""
    
    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.error_code = error_code
```

**Step 1.2.2**: Update Repository Files
```python
# Replace in all 5 repository files:
# BEFORE:
try:
    from common.reliability import DataValidationException
except ImportError:
    class DataValidationException(ValueError): ...

# AFTER:
from common.exceptions import DataValidationException
```

#### Test Cases:
1. **Exception Creation Test**:
   ```python
   def test_data_validation_exception():
       exc = DataValidationException("Test message", field="symbol")
       assert exc.field == "symbol"
       assert str(exc) == "Test message"
   ```

2. **Import Test**:
   ```python
   def test_exception_imports():
       from common.exceptions import (
           DataValidationException,
           RepositoryException,
           DataIntegrityException
       )
       # All should import successfully
   ```

3. **Repository Integration Test**:
   ```python
   def test_repository_exception_usage():
       repo = EquityBarRepository("test_data")
       with pytest.raises(DataValidationException) as exc_info:
           repo.save(invalid_data, symbol="TEST")
       assert exc_info.value.field is not None
   ```

#### Edge Cases:
- **Circular Imports**: Ensure no dependency cycles
- **Backward Compatibility**: Old exception handling still works during transition
- **Exception Hierarchy**: Maintain proper inheritance chains

#### Migration Strategy:
1. Create `common/exceptions.py` with all exception classes
2. Update imports in repository files one by one
3. Run tests after each file update
4. Remove duplicate exception definitions

#### Verification:
```bash
# Test each repository individually
pytest tests/unit/test_*_repository.py -v
# Check for remaining duplicates
grep -r "class.*Exception" common/repositories/ --exclude="exceptions.py"
# Full integration test
make test-unit
```

#### Rollback Plan:
- Revert repository file imports to original try/except pattern
- Remove `common/exceptions.py`
- All functionality preserved in duplicated definitions

---

## PHASE 2: CODE CLEANUP ✅ **COMPLETED** 

**Status:** ✅ Completed on 2024-12-26  
**Branch:** `refactoring/phase1-critical-fixes`  
**Impact:** Major code organization and maintainability improvements

### 🟡 Task 2.1: Clean Root Directory

**Files**: 25+ root-level Python files
**Impact**: MEDIUM - Organization and maintainability  
**Time Estimate**: 6 hours

#### Current Root-Level Files Analysis:
```
├── atm_vol_term.py           # ATM volatility calculations - ARCHIVE
├── implied_vol.py            # Volatility analysis - ARCHIVE  
├── ib_vol_history.py         # Historical volatility - ARCHIVE
├── live_vol_surface.py       # Real-time volatility - ARCHIVE
├── option_ticks.py           # Options tick data - ARCHIVE
├── streamer.py              # Legacy streaming - DELETE
├── vol_surface.py           # Volatility surface - ARCHIVE
├── subscription_usage.py    # Usage tracking - MOVE TO /ops
├── test_*.py                # Test files - MOVE TO /tests
├── conftest.py              # Test config - MOVE TO /tests  
├── conftest_full.py         # Full test config - MOVE TO /tests
├── main.py                  # Keep - Entry point
├── config.py                # Keep - Configuration
├── sitecustomize.py         # Keep - Logging setup
└── utils.py                 # MOVE TO /common/utils
```

#### Implementation Steps:

**Step 2.1.1**: Create Archive Directory
```bash
mkdir -p archive/volatility
mkdir -p archive/experimental
```

**Step 2.1.2**: Move Files by Category
```bash
# Volatility-related files
mv atm_vol_term.py implied_vol.py ib_vol_history.py archive/volatility/
mv live_vol_surface.py vol_surface.py archive/volatility/

# Test files  
mv test_*.py tests/legacy/
mv conftest.py conftest_full.py tests/

# Utility moves
mv subscription_usage.py ops/utilities/
mv utils.py common/utils/legacy_utils.py
```

**Step 2.1.3**: Update Import References
```bash
# Find all imports of moved files
grep -r "import.*vol_surface\|from.*vol_surface" . --exclude-dir=archive
grep -r "import.*utils\|from utils import" . --exclude="utils.py"
```

#### Test Cases:
1. **Import Resolution Test**:
   ```python
   def test_no_broken_imports():
       """Verify no imports reference moved files."""
       import subprocess
       result = subprocess.run(['python', '-m', 'py_compile'] + glob.glob('*.py'))
       assert result.returncode == 0
   ```

2. **Essential Files Test**:
   ```python
   def test_essential_files_present():
       """Verify critical files remain in root."""
       essential_files = ['main.py', 'config.py', 'sitecustomize.py']
       for file in essential_files:
           assert Path(file).exists()
   ```

3. **Archive Accessibility Test**:
   ```python
   def test_archived_files_accessible():
       """Verify archived files can still be imported if needed."""
       sys.path.append('archive/volatility')
       import vol_surface  # Should work from archive
   ```

#### Edge Cases:
- **Active Import Usage**: Some moved files may still be imported
- **Git History**: Preserve file history during moves
- **Documentation References**: Update any README/docs that reference file locations

#### Migration Strategy:
1. **Analysis Phase**: Identify all import dependencies
2. **Gradual Move**: Move files in dependency order (least dependent first)
3. **Import Updates**: Update any remaining imports
4. **Verification**: Run full test suite
5. **Documentation**: Update CLAUDE.md with new structure

#### Verification:
```bash
# Check for broken imports
python -m py_compile *.py
# Verify tests still pass
make test-unit
# Check documentation references
grep -r "\.py" CLAUDE.md README.md
```

#### Rollback Plan:
```bash
# Simple file moves back to root
mv archive/volatility/*.py .
mv tests/legacy/test_*.py .
mv tests/conftest*.py .
mv ops/utilities/subscription_usage.py .
mv common/utils/legacy_utils.py utils.py
```

---

### 🟡 Task 2.2: Refactor Large Files

**Files**: `ops/tasks/historical_data.py` (976 lines), `common/repositories/base.py` (692 lines), `earnings/fetcher.py` (911 lines)
**Impact**: HIGH - Maintainability and code complexity  
**Time Estimate**: 16 hours

#### Task 2.2.1: Split `ops/tasks/historical_data.py`

**Current Structure Analysis**:
- 9 Celery task functions
- 4 helper functions  
- Mixed concerns: contract handling, data fetching, event processing

**Target Structure**:
```
ops/tasks/
├── __init__.py
├── contract_tasks.py          # Contract-related tasks
├── equity_data_tasks.py       # Equity bar tasks
├── option_data_tasks.py       # Option bar tasks  
├── historical_base.py         # Shared utilities
└── historical_data.py         # Legacy compatibility layer
```

**Implementation Steps**:

**Step 2.2.1.1**: Create Base Utilities
```python
# ops/tasks/historical_base.py
"""Shared utilities for historical data tasks."""

from typing import List, Dict, Any
from celery import shared_task
from common.api.ib_client import IBClient

class HistoricalDataMixin:
    """Common functionality for historical data tasks."""
    
    def validate_request(self, contracts: List[Dict], **kwargs):
        """Validate historical data request parameters."""
        pass
        
    def setup_client(self) -> IBClient:
        """Setup IB client with proper configuration."""
        pass
        
    def handle_rate_limits(self, request_count: int):
        """Handle IB API rate limiting."""
        pass
```

**Step 2.2.1.2**: Extract Contract Tasks
```python
# ops/tasks/contract_tasks.py
"""Celery tasks for contract operations."""

from .historical_base import HistoricalDataMixin

@shared_task(bind=True, base=HistoricalDataMixin)
def search_contracts(self, symbols: List[str], **kwargs):
    """Search for contract details by symbols."""
    pass

@shared_task(bind=True, base=HistoricalDataMixin)  
def fetch_contract_details(self, contracts: List[Dict], **kwargs):
    """Fetch detailed contract information."""
    pass
```

#### Test Cases for File Splitting:
1. **Task Import Test**:
   ```python
   def test_task_imports():
       from ops.tasks.contract_tasks import search_contracts
       from ops.tasks.equity_data_tasks import fetch_equity_bars
       from ops.tasks.option_data_tasks import fetch_option_bars
       # All should import successfully
   ```

2. **Task Execution Test**:
   ```python
   def test_task_execution():
       # Test that split tasks work identically to original
       result = search_contracts.delay(["AAPL", "MSFT"])
       assert result.successful()
   ```

3. **Backward Compatibility Test**:
   ```python
   def test_legacy_imports():
       # Original imports should still work during transition
       from ops.tasks.historical_data import search_contracts
       assert search_contracts is not None
   ```

#### Task 2.2.2: Refactor `common/repositories/base.py`

**Current Issues**:
- 692-line monolithic class
- Complex `load()` method (140+ lines)
- Mixed responsibilities: schema, validation, querying, caching

**Target Structure**:
```python
# common/repositories/base.py - Slimmed down base class
# common/repositories/mixins.py - Reusable functionality  
# common/repositories/query.py - Query building logic
# common/repositories/schema.py - Schema utilities
```

**Implementation Steps**:

**Step 2.2.2.1**: Extract Query Builder
```python
# common/repositories/query.py
"""Query building utilities for PyArrow datasets."""

import pyarrow as pa
import pyarrow.dataset as ds
from typing import List, Tuple, Any

class QueryBuilder:
    """Build PyArrow filter expressions."""
    
    def __init__(self, schema: pa.Schema):
        self.schema = schema
        
    def build_filters(self, filters: List[Tuple[str, str, Any]]) -> pa.Expression:
        """Convert filter tuples to PyArrow expressions."""
        pass
        
    def build_date_range_filter(self, start_date, end_date, date_column='date') -> pa.Expression:
        """Build date range filter expression."""
        pass
```

**Step 2.2.2.2**: Extract Schema Utilities  
```python
# common/repositories/schema.py
"""Schema validation and manipulation utilities."""

import pyarrow as pa
from typing import Dict, Any

class SchemaValidator:
    """Validate data against PyArrow schemas."""
    
    def validate_dataframe(self, df, schema: pa.Schema) -> None:
        """Validate DataFrame against schema."""
        pass
        
    def normalize_types(self, df, schema: pa.Schema):
        """Normalize DataFrame types to match schema.""" 
        pass
```

#### Test Cases for Base Repository Refactoring:
1. **Functionality Preservation Test**:
   ```python
   def test_repository_functionality_preserved():
       # Test before and after refactoring
       old_repo = OldEquityBarRepository("test_data")  
       new_repo = EquityBarRepository("test_data")
       
       # Same data should produce same results
       old_result = old_repo.load(filters=[("symbol", "=", "AAPL")])
       new_result = new_repo.load(filters=[("symbol", "=", "AAPL")])
       
       pd.testing.assert_frame_equal(old_result, new_result)
   ```

2. **Performance Regression Test**:
   ```python
   def test_performance_no_regression():
       import time
       repo = EquityBarRepository("test_data")
       
       start = time.time()
       result = repo.load(filters=[("symbol", "=", "AAPL")])
       duration = time.time() - start
       
       # Should not be significantly slower than baseline
       assert duration < BASELINE_DURATION * 1.1
   ```

#### Edge Cases for Large File Refactoring:
- **Circular Dependencies**: Ensure no import cycles in split modules
- **Task Registration**: Celery tasks must be properly discovered after splitting
- **Memory Usage**: Ensure refactoring doesn't increase memory footprint
- **Thread Safety**: Preserve thread-safety in repository operations

#### Verification Strategy:
```bash
# Run specific test suites
pytest tests/unit/test_historical_data.py -v
pytest tests/unit/test_repositories.py -v
# Performance benchmarking
python -m pytest tests/benchmarks/ --benchmark-only
# Integration testing
pytest tests/integration/test_data_flow.py -v
```

---

### 🟡 Task 2.3: Standardize API Structure

**Current Structure**: Mixed `/api/` and `/services/` directories
**Impact**: MEDIUM - Code organization and developer experience
**Time Estimate**: 8 hours

#### Current API Layout Analysis:
```
api/                           # Internal/IB APIs
├── ib_api/                   # IB Gateway API
├── orchestrator_api/         # Workflow orchestration  
└── option_chain_api/         # Option chain data

services/                     # External/Frontend APIs  
├── calendar_api/             # Earnings calendar
├── equity_api/               # Equity historical data
├── option_history_api/       # Option historical data
└── frontend/                 # Web UI service
```

#### Proposed Standardized Structure:
```
api/
├── internal/                 # Internal system APIs
│   ├── ib/                  # Interactive Brokers integration
│   └── orchestrator/        # Workflow orchestration
├── external/                # External client APIs
│   ├── calendar/            # Earnings calendar API
│   ├── equity/              # Equity data API  
│   ├── options/             # Options data API
│   └── chains/              # Option chains API
└── frontend/                # Web UI and static assets
```

#### Implementation Steps:

**Step 2.3.1**: Create New Structure
```bash
mkdir -p api/internal api/external api/frontend
```

**Step 2.3.2**: Move APIs with Compatibility Layer
```bash
# Move internal APIs
mv api/ib_api api/internal/ib
mv api/orchestrator_api api/internal/orchestrator  
mv api/option_chain_api api/internal/chains

# Move external APIs
mv services/calendar_api api/external/calendar
mv services/equity_api api/external/equity
mv services/option_history_api api/external/options

# Move frontend
mv services/frontend api/frontend/legacy
mv frontend-nextjs api/frontend/nextjs
```

**Step 2.3.3**: Create Import Compatibility Layer
```python
# api/ib_api/__init__.py (compatibility)
"""
Backward compatibility layer for ib_api imports.
Redirects to new location: api/internal/ib
"""
import warnings
from api.internal.ib import *

warnings.warn(
    "api.ib_api is deprecated. Use api.internal.ib instead.",
    DeprecationWarning,
    stacklevel=2
)
```

#### Test Cases:
1. **API Endpoint Test**:
   ```python
   def test_api_endpoints_accessible():
       """Test that all API endpoints still work after restructuring."""
       import requests
       
       endpoints = [
           "http://localhost:9001/health",  # Calendar
           "http://localhost:9002/health",  # Equity  
           "http://localhost:9010/health",  # IB API
       ]
       
       for endpoint in endpoints:
           response = requests.get(endpoint)
           assert response.status_code == 200
   ```

2. **Import Backward Compatibility Test**:
   ```python
   def test_import_compatibility():
       """Test that old imports still work with deprecation warnings."""
       import warnings
       
       with warnings.catch_warnings(record=True) as w:
           from api.ib_api import IBClient  # Old import
           assert len(w) == 1
           assert issubclass(w[0].category, DeprecationWarning)
           
       from api.internal.ib import IBClient  # New import - no warning
   ```

3. **Docker Compose Test**:
   ```python
   def test_docker_services_start():
       """Test that docker-compose still works with new structure."""
       import subprocess
       result = subprocess.run(['docker-compose', 'up', '-d'], 
                             capture_output=True)
       assert result.returncode == 0
   ```

#### Edge Cases:
- **Docker Volume Mounts**: Update docker-compose.yml volume paths
- **Import Statements**: Update all internal imports to new paths
- **Environment Variables**: Update any env vars that reference old paths
- **Documentation**: Update CLAUDE.md with new structure
- **CI/CD**: Update any deployment scripts with new paths

#### Migration Strategy:
1. **Create New Structure**: Set up new directory layout
2. **Move Files**: Gradually move API directories  
3. **Compatibility Layer**: Add backward compatibility imports
4. **Update References**: Update docker-compose, imports, docs
5. **Testing**: Verify all services still work
6. **Cleanup**: Remove compatibility layer after transition period

---

## PHASE 3: ARCHITECTURE IMPROVEMENTS (Month 2)

### 🟢 Task 3.1: Create Shared Libraries

**Impact**: HIGH - Code reuse and consistency
**Time Estimate**: 12 hours

#### Target Structure:
```python
common/
├── exceptions.py             # All custom exceptions
├── schemas/                  # Shared Pydantic models
│   ├── __init__.py
│   ├── base.py              # Base model classes
│   ├── trading.py           # Trading-related models
│   └── api.py               # API response models
├── decorators.py            # Common decorators
├── constants.py             # System-wide constants
└── types.py                # Custom type definitions
```

#### Implementation Steps:

**Step 3.1.1**: Create Constants Module
```python
# common/constants.py
"""System-wide constants for the trading platform."""

from enum import Enum
from typing import Final

# IB API Constants
class SecType(str, Enum):
    """Security type constants."""
    STOCK = "STK"
    OPTION = "OPT"  
    FUTURE = "FUT"
    FOREX = "CASH"

# Data Constants
DEFAULT_BAR_SIZES: Final = ["1 min", "5 mins", "1 hour", "1 day"]
MAX_DAYS_HISTORY: Final = 365
DEFAULT_EXCHANGE: Final = "SMART"

# Repository Constants  
PARQUET_ENGINE: Final = "pyarrow"
MAX_ROWS_PER_PARTITION: Final = 100000

# API Constants
DEFAULT_TIMEOUT: Final = 30.0
MAX_RETRIES: Final = 3
RATE_LIMIT_REQUESTS_PER_SECOND: Final = 2
```

**Step 3.1.2**: Create Shared Schema Base Classes
```python
# common/schemas/base.py
"""Base Pydantic models for consistent validation."""

from datetime import date, datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any

class BaseAPIModel(BaseModel):
    """Base model for API requests/responses."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

class TimestampedModel(BaseAPIModel):
    """Base model with timestamp tracking."""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
class PaginatedResponse(BaseAPIModel):
    """Standard paginated response format."""
    
    items: list
    total: int
    page: int = 1
    size: int = 50
    has_next: bool = False
```

**Step 3.1.3**: Create Trading-Specific Schemas
```python
# common/schemas/trading.py
"""Trading domain Pydantic models."""

from decimal import Decimal
from datetime import date
from pydantic import BaseModel, Field
from .base import BaseAPIModel
from ..constants import SecType

class ContractBase(BaseAPIModel):
    """Base contract information."""
    
    symbol: str = Field(..., min_length=1, max_length=10)
    sec_type: SecType
    exchange: str = Field(default="SMART")
    currency: str = Field(default="USD", max_length=3)

class EquityContract(ContractBase):
    """Equity contract specification."""
    
    sec_type: SecType = Field(default=SecType.STOCK, frozen=True)

class OptionContract(ContractBase):
    """Option contract specification."""
    
    sec_type: SecType = Field(default=SecType.OPTION, frozen=True) 
    expiry: date
    strike: Decimal = Field(..., gt=0)
    right: str = Field(..., regex="^[CP]$")  # Call or Put
```

#### Test Cases for Shared Libraries:
1. **Schema Validation Test**:
   ```python
   def test_contract_validation():
       # Valid contract
       contract = EquityContract(symbol="AAPL")
       assert contract.symbol == "AAPL"
       assert contract.sec_type == SecType.STOCK
       
       # Invalid contract
       with pytest.raises(ValidationError):
           EquityContract(symbol="")  # Empty symbol
   ```

2. **Constants Usage Test**:
   ```python
   def test_constants_accessibility():
       from common.constants import DEFAULT_BAR_SIZES, SecType
       
       assert "1 day" in DEFAULT_BAR_SIZES
       assert SecType.STOCK == "STK"
   ```

3. **Cross-Module Consistency Test**:
   ```python
   def test_schema_consistency():
       """Test that schemas work consistently across modules."""
       from api.external.equity.schemas import EquityBarsRequest
       from common.schemas.trading import EquityContract
       
       # Should be compatible
       contract = EquityContract(symbol="AAPL")
       request = EquityBarsRequest(contract=contract, start="2024-01-01")
   ```

#### Edge Cases:
- **Import Cycles**: Ensure no circular dependencies between shared modules
- **Version Compatibility**: Maintain backward compatibility for existing models
- **Performance**: Ensure shared models don't slow down API responses
- **Validation Errors**: Consistent error formats across all APIs

---

### 🟢 Task 3.2: Repository Pattern Enhancement

**Impact**: HIGH - Type safety and code reuse
**Time Estimate**: 16 hours

#### Current Issues:
- No common interface/protocol for repositories
- Duplicate method implementations across 8 repository classes
- No type safety for repository operations
- Factory pattern needed for dynamic repository creation

#### Target Architecture:
```python
common/repositories/
├── __init__.py
├── base.py                   # Slimmed down base class
├── interfaces.py             # Repository protocols
├── mixins/                   # Reusable functionality
│   ├── __init__.py
│   ├── caching.py           # Caching mixin
│   ├── validation.py        # Validation mixin
│   └── partitioning.py      # Partitioning mixin
├── factory.py               # Repository factory
└── [individual repos]       # Existing repository classes
```

#### Implementation Steps:

**Step 3.2.1**: Create Repository Protocols
```python
# common/repositories/interfaces.py
"""Repository interface protocols for type safety."""

from typing import Protocol, TypeVar, Generic, List, Dict, Any, Optional
from datetime import date
import pandas as pd
import pyarrow as pa

T = TypeVar('T')

class Repository(Protocol[T]):
    """Base repository protocol."""
    
    def save(self, data: pd.DataFrame, **kwargs) -> None:
        """Save data to repository."""
        ...
        
    def load(self, filters: List[tuple] = None, **kwargs) -> pd.DataFrame:
        """Load data from repository."""
        ...
        
    def exists(self, **kwargs) -> bool:
        """Check if data exists."""
        ...
        
    def delete(self, **kwargs) -> None:
        """Delete data from repository."""
        ...

class TimeSeriesRepository(Repository[T], Protocol):
    """Time series data repository protocol."""
    
    def load_date_range(self, start: date, end: date, **kwargs) -> pd.DataFrame:
        """Load data for date range."""
        ...
        
    def get_available_dates(self, **kwargs) -> List[date]:
        """Get available dates.""" 
        ...

class SymbolRepository(Repository[T], Protocol):
    """Symbol-based repository protocol."""
    
    def load_symbol(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data for specific symbol."""
        ...
        
    def get_symbols(self) -> List[str]:
        """Get available symbols."""
        ...
```

**Step 3.2.2**: Create Reusable Mixins
```python
# common/repositories/mixins/caching.py
"""Caching functionality for repositories."""

from functools import wraps
from typing import Dict, Any, Optional
import hashlib
import pickle

class CachingMixin:
    """Adds in-memory caching to repository operations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, Any] = {}
        self.cache_enabled = True
        
    def _cache_key(self, method_name: str, *args, **kwargs) -> str:
        """Generate cache key for method call."""
        key_data = f"{method_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def cached(self, ttl_seconds: int = 300):
        """Decorator to cache method results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.cache_enabled:
                    return func(*args, **kwargs)
                    
                cache_key = self._cache_key(func.__name__, *args, **kwargs)
                
                # Check cache
                if cache_key in self._cache:
                    cached_result, cached_time = self._cache[cache_key]
                    if time.time() - cached_time < ttl_seconds:
                        return cached_result
                
                # Execute and cache
                result = func(*args, **kwargs)
                self._cache[cache_key] = (result, time.time())
                return result
            return wrapper
        return decorator
```

**Step 3.2.3**: Create Repository Factory
```python
# common/repositories/factory.py
"""Factory for creating repository instances."""

from typing import Type, Dict, Any
from .interfaces import Repository
from .equity_bars import EquityBarRepository
from .option_bars import OptionBarRepository
from .option_chains import OptionChainSnapshotRepository

class RepositoryFactory:
    """Factory for creating repository instances."""
    
    _registry: Dict[str, Type[Repository]] = {
        "equity_bars": EquityBarRepository,
        "option_bars": OptionBarRepository, 
        "option_chains": OptionChainSnapshotRepository,
    }
    
    @classmethod
    def create(cls, repository_type: str, data_root: str, **kwargs) -> Repository:
        """Create repository instance by type."""
        if repository_type not in cls._registry:
            raise ValueError(f"Unknown repository type: {repository_type}")
            
        repo_class = cls._registry[repository_type]
        return repo_class(data_root, **kwargs)
    
    @classmethod  
    def register(cls, name: str, repository_class: Type[Repository]):
        """Register new repository type."""
        cls._registry[name] = repository_class
        
    @classmethod
    def list_types(cls) -> List[str]:
        """List available repository types."""
        return list(cls._registry.keys())
```

#### Test Cases for Repository Enhancements:
1. **Protocol Compliance Test**:
   ```python
   def test_repository_protocol_compliance():
       """Test that repositories implement required protocols."""
       repo = EquityBarRepository("test_data")
       
       # Should implement Repository protocol
       assert hasattr(repo, 'save')
       assert hasattr(repo, 'load') 
       assert hasattr(repo, 'exists')
       assert hasattr(repo, 'delete')
       
       # Should implement TimeSeriesRepository protocol
       assert hasattr(repo, 'load_date_range')
       assert hasattr(repo, 'get_available_dates')
   ```

2. **Factory Pattern Test**:
   ```python
   def test_repository_factory():
       """Test repository factory creation."""
       repo = RepositoryFactory.create("equity_bars", "test_data")
       assert isinstance(repo, EquityBarRepository)
       
       # Test unknown type
       with pytest.raises(ValueError):
           RepositoryFactory.create("unknown_type", "test_data")
   ```

3. **Mixin Functionality Test**:
   ```python
   def test_caching_mixin():
       """Test caching mixin functionality."""
       class TestRepo(CachingMixin, BaseRepository):
           @cached(ttl_seconds=60)
           def expensive_operation(self, data):
               return len(data) * 2
               
       repo = TestRepo("test_data")
       
       # First call
       result1 = repo.expensive_operation([1, 2, 3])
       # Second call should use cache
       result2 = repo.expensive_operation([1, 2, 3])
       assert result1 == result2
   ```

#### Edge Cases:
- **Protocol Violations**: Repositories that don't implement required methods
- **Factory Registration**: Duplicate or invalid repository registrations
- **Mixin Conflicts**: Multiple mixins modifying the same methods
- **Type Safety**: Generic type parameters working correctly
- **Performance**: Mixins not significantly impacting performance

---

### 🟢 Task 3.3: Configuration Consolidation

**Current Issues**: Multiple configuration systems (`config.py` + `app_settings/`)
**Impact**: MEDIUM - Developer experience and maintainability
**Time Estimate**: 8 hours

#### Current Configuration Analysis:
```
├── config.py                 # Legacy configuration
├── app_settings/             # Modern pydantic-settings
│   ├── __init__.py
│   └── settings.py
└── [scattered config files]  # Various .ini, .yaml files
```

#### Target Configuration Structure:
```
config/
├── __init__.py
├── base.py                   # Base configuration classes
├── environments/             # Environment-specific configs
│   ├── development.py
│   ├── testing.py
│   └── production.py
├── logging.py               # Logging configuration
└── settings.py              # Main settings entry point
```

#### Implementation Steps:

**Step 3.3.1**: Create Base Configuration Classes
```python
# config/base.py
"""Base configuration classes using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from pathlib import Path

class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        extra="forbid"
    )
    
    data_root: Path = Path("data")
    parquet_engine: str = "pyarrow"
    max_rows_per_partition: int = 100000

class IBAPIConfig(BaseSettings):
    """Interactive Brokers API configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="IBAPI_",
        env_file=".env"
    )
    
    host: str = "127.0.0.1"  
    port: int = 7497
    client_id: int = 1
    timeout: float = 30.0
    max_requests_per_second: int = 2

class APIConfig(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env"
    )
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "INFO"

class AppConfig(BaseSettings):
    """Main application configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    environment: str = "development"
    debug: bool = False
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    ibapi: IBAPIConfig = IBAPIConfig()  
    api: APIConfig = APIConfig()
```

**Step 3.3.2**: Create Environment-Specific Configurations
```python
# config/environments/development.py
"""Development environment configuration."""

from ..base import AppConfig, APIConfig

class DevelopmentConfig(AppConfig):
    """Development-specific configuration."""
    
    environment: str = "development" 
    debug: bool = True
    
    api: APIConfig = APIConfig(
        reload=True,
        log_level="DEBUG"
    )

# config/environments/production.py
"""Production environment configuration."""

from ..base import AppConfig, APIConfig, IBAPIConfig

class ProductionConfig(AppConfig):
    """Production-specific configuration."""
    
    environment: str = "production"
    debug: bool = False
    
    api: APIConfig = APIConfig(
        workers=4,
        log_level="INFO"
    )
    
    ibapi: IBAPIConfig = IBAPIConfig(
        port=7496,  # Live trading port
        max_requests_per_second=1  # More conservative
    )
```

**Step 3.3.3**: Create Configuration Factory
```python
# config/settings.py
"""Configuration factory and settings access."""

import os
from typing import Type
from .base import AppConfig
from .environments.development import DevelopmentConfig
from .environments.testing import TestingConfig
from .environments.production import ProductionConfig

CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig
}

def get_config() -> AppConfig:
    """Get configuration based on environment."""
    env = os.getenv("APP_ENV", "development")
    
    if env not in CONFIG_MAP:
        raise ValueError(f"Unknown environment: {env}")
        
    return CONFIG_MAP[env]()

# Global settings instance
settings = get_config()
```

#### Test Cases for Configuration:
1. **Environment Loading Test**:
   ```python
   def test_environment_config_loading():
       """Test loading different environment configurations."""
       os.environ["APP_ENV"] = "development"
       config = get_config()
       assert config.debug is True
       
       os.environ["APP_ENV"] = "production"  
       config = get_config()
       assert config.debug is False
   ```

2. **Environment Variable Override Test**:
   ```python
   def test_env_var_override():
       """Test environment variable overrides."""
       os.environ["API_PORT"] = "9999"
       config = get_config()
       assert config.api.port == 9999
   ```

3. **Configuration Validation Test**:
   ```python
   def test_config_validation():
       """Test configuration validation."""
       with pytest.raises(ValidationError):
           IBAPIConfig(port="invalid")  # Should be int
           
       with pytest.raises(ValidationError):
           APIConfig(workers=-1)  # Should be positive
   ```

#### Migration Strategy:
1. **Parallel Implementation**: Keep old config alongside new
2. **Gradual Migration**: Update modules one by one
3. **Compatibility Layer**: Provide backward compatibility
4. **Environment Testing**: Test all environments thoroughly
5. **Documentation Update**: Update CLAUDE.md with new patterns

---

## PHASE 4: LONG-TERM IMPROVEMENTS (Month 3)

### 🔵 Task 4.1: Performance Optimization

**Impact**: MEDIUM - System performance and scalability
**Time Estimate**: 20 hours

#### Target Optimizations:
1. **Connection Pooling**: Reuse IB API connections
2. **Query Optimization**: Improve PyArrow query performance
3. **Caching Layer**: Add Redis/Memory caching
4. **Batch Processing**: Optimize bulk data operations

#### Implementation Steps:

**Step 4.1.1**: Connection Pooling
```python
# common/api/connection_pool.py
"""Connection pooling for IB API clients."""

import asyncio
from typing import Dict, Optional
from contextlib import asynccontextmanager
from .ib_client import IBClient

class IBConnectionPool:
    """Pool of IB API connections for reuse."""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self._available: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._active: Dict[str, IBClient] = {}
        
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool."""
        try:
            # Try to get existing connection
            client = await asyncio.wait_for(
                self._available.get(), timeout=1.0
            )
        except asyncio.TimeoutError:
            # Create new connection if pool not full
            if len(self._active) < self.max_connections:
                client = IBClient()
                await client.connect()
            else:
                # Wait for available connection
                client = await self._available.get()
                
        try:
            yield client
        finally:
            # Return connection to pool
            await self._available.put(client)
```

#### Test Cases for Performance:
1. **Connection Pool Test**:
   ```python
   async def test_connection_pooling():
       """Test connection pool functionality."""
       pool = IBConnectionPool(max_connections=2)
       
       # Test concurrent access
       async with pool.get_connection() as client1:
           async with pool.get_connection() as client2:
               assert client1 != client2
               
       # Test connection reuse
       async with pool.get_connection() as client3:
           assert client3 in (client1, client2)
   ```

2. **Performance Benchmark Test**:
   ```python
   def test_query_performance():
       """Benchmark query performance improvements."""
       repo = EquityBarRepository("test_data")
       
       # Baseline performance
       start = time.time()
       result = repo.load(filters=[("symbol", "=", "AAPL")])
       baseline = time.time() - start
       
       # Optimized performance should be better
       start = time.time()
       result_optimized = repo.load_optimized(filters=[("symbol", "=", "AAPL")])
       optimized = time.time() - start
       
       assert optimized < baseline * 0.8  # At least 20% improvement
   ```

---

### 🔵 Task 4.2: Testing Improvements

**Impact**: HIGH - Code quality and reliability
**Time Estimate**: 24 hours

#### Current Test Coverage Analysis:
- Unit tests: ~60% coverage
- Integration tests: Limited
- Contract tests: None
- Performance tests: None

#### Target Testing Structure:
```
tests/
├── unit/                     # Unit tests (80%+ coverage)
├── integration/              # Integration tests  
├── contracts/               # Contract tests between services
├── performance/             # Performance benchmarks
├── fixtures/                # Test data and utilities
└── conftest.py             # Pytest configuration
```

#### Test Cases and Edge Cases Summary:

**Critical Edge Cases to Test:**

1. **Data Integrity Edge Cases**:
   ```python
   def test_malformed_data_handling():
       """Test handling of malformed market data."""
       # Missing columns
       # Invalid data types  
       # Null values in required fields
       # Duplicate timestamps
       # Out-of-order timestamps
   ```

2. **IB API Edge Cases**:
   ```python  
   def test_ib_api_error_scenarios():
       """Test IB API error handling."""
       # Connection timeout
       # Rate limit exceeded
       # Invalid contract specification
       # Market data not available
       # Historical data gaps
   ```

3. **Repository Edge Cases**:
   ```python
   def test_repository_edge_cases():
       """Test repository edge cases."""
       # Empty datasets
       # Large datasets (memory limits)
       # Corrupted Parquet files
       # Concurrent read/write operations
       # Partition boundary conditions
   ```

4. **Event Processing Edge Cases**:
   ```python
   def test_event_processing_edge_cases():
       """Test event processing edge cases."""
       # Duplicate events
       # Out-of-order events
       # Malformed event payloads
       # Consumer checkpoint corruption
       # Event bus file rotation
   ```

---

## ROLLBACK STRATEGIES

### Immediate Rollback (< 5 minutes)
```bash
# Git-based rollback for any changes
git checkout HEAD~1  # Rollback to previous commit
git reset --hard <commit-hash>  # Rollback to specific commit

# Service-specific rollback
docker-compose down && docker-compose up -d  # Restart services
systemctl restart celery-worker  # Restart task queue
```

### Staged Rollback (Per Phase)
```bash
# Phase 1 rollback
git revert <phase1-commit-range>
# Restore original imports
# Remove common/exceptions.py

# Phase 2 rollback  
# Move files back to original locations
mv archive/volatility/*.py .
# Restore original API structure

# Phase 3 rollback
# Remove shared libraries
# Restore original repository classes
# Revert configuration changes

# Phase 4 rollback
# Remove performance optimizations
# Restore original implementations
```

### Data Recovery
```bash
# Backup before major changes
cp -r data/ data_backup_$(date +%Y%m%d_%H%M%S)

# Restore if needed
rm -rf data/ && mv data_backup_* data/
```

---

## MONITORING AND VERIFICATION

### Success Metrics
- [ ] All tests pass after each phase
- [ ] Performance not degraded (< 5% slowdown acceptable) 
- [ ] No broken imports or circular dependencies
- [ ] Documentation updated and accurate
- [ ] Code coverage maintained or improved
- [ ] No production issues during transition

### Continuous Monitoring
```bash
# Run after each major change
make test-unit                    # Unit test coverage
make test-integration            # Integration tests  
python -m py_compile **/*.py     # Import validation
docker-compose up --build       # Service startup test
```

### Production Health Checks
```bash
# API health checks
curl http://localhost:9001/health  # Calendar API
curl http://localhost:9002/health  # Equity API
curl http://localhost:9005/health  # Orchestrator API

# Data integrity checks  
pytest tests/integration/test_data_integrity.py
```

---

## CONCLUSION

This refactoring plan provides a comprehensive, phased approach to improving the trading platform codebase. Each phase builds upon the previous one while maintaining backward compatibility and minimizing disruption to production operations.

The plan prioritizes:
1. **Critical fixes first** (broken imports, security issues)
2. **Incremental improvements** (code organization, duplication)
3. **Architecture enhancements** (patterns, interfaces, shared libraries)
4. **Performance and quality** (optimization, testing, monitoring)

By following this structured approach with comprehensive testing and rollback strategies, the refactoring can be executed safely while significantly improving code quality, maintainability, and developer experience.

**Estimated Total Time**: 8-10 weeks of development effort
**Risk Level**: Low (due to incremental approach and comprehensive testing)
**Expected Outcome**: Codebase grade improvement from B+ to A