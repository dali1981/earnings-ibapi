# Repository Migration Status

## Current State After Migration Attempt

### Repository Modules in Codebase

The codebase has **two separate repository implementations**:

#### 1. **`common/repositories/`** - **Modern/Production Architecture** ✅
- **Location**: `common/repositories/`
- **Used by**: Production services, main operational code
- **API Style**: Method-specific (e.g., `save_daily_bars()`, `save_intraday_bars()`)
- **Base Class**: `BaseRepository` (PyArrow-based, partitioned storage)

#### 2. **`vendors/ibx_repos/`** - **Legacy/Test Architecture** ⚠️  
- **Location**: `vendors/ibx_repos/` (accessed via `ibx_repos/` stub)
- **Used by**: Test fixtures, some legacy operational code
- **API Style**: Generic methods (e.g., `save(df, symbol, bar_size, what_to_show)`)
- **Base Class**: Custom implementation

## Migration Results

### ✅ Successfully Migrated
- **Production Services**: Already using `common/repositories`
- **Main Operations**: Most `ops/jobs/` and `ops/examples/` migrated
- **File Organization**: Unit tests moved to `tests/unit/repositories/`

### ⚠️ API Incompatibility Discovered
The two repository systems have **incompatible APIs**:

**Legacy (`ibx_repos`) API:**
```python
repo.save(df, symbol="AAPL", bar_size="1 day", what_to_show="TRADES")
```

**Modern (`common/repositories`) API:**  
```python
repo.save_daily_bars(df, symbol="AAPL", data_type="TRADES")  # Daily bars
repo.save_intraday_bars(df, symbol="AAPL", bar_size="1 hour", data_type="TRADES")  # Intraday
```

### 🔄 Reverted Items
Due to API incompatibility, these were reverted to `ibx_repos`:
- **Test fixtures** (`conftest.py`, `conftest_full.py`)  
- **Unit tests** in `tests/unit/repositories/`
- **Some operational tasks** that depend on the legacy API

## Current Repository Usage Map

### **Production Code** → `common/repositories` ✅
```
services/*/routers/       → common/repositories
ops/jobs/ (most)          → common/repositories  
ops/scripts/              → common/repositories
ops/examples/ (most)      → common/repositories
```

### **Test Code** → `ibx_repos` ⚠️
```
tests/conftest.py         → ibx_repos (fixtures)
tests/unit/repositories/  → ibx_repos (unit tests)
tests/test_present_dates.py → ibx_repos
```

### **Mixed/Legacy** → Both 📍
```
ops/tasks/                → Both (fallback patterns)
Some operational jobs     → ibx_repos (API dependency)
```

## Recommendations

### Option A: **Accept Current Hybrid State** (Recommended)
- **Keep** production services on `common/repositories`
- **Keep** tests on `ibx_repos` (stable, working fixtures)
- **Document** the two-system architecture clearly
- **Migrate gradually** as needed for specific features

### Option B: **Complete API Migration**
- Create **compatibility layer** in `common/repositories`
- Add generic `save()` method that delegates to specific methods
- Update all tests to use `common/repositories`
- **Risk**: More complex, potential breaking changes

### Option C: **Standardize on Legacy System**
- Revert production code back to `ibx_repos`
- Maintain single repository system
- **Risk**: Miss benefits of modern architecture

## Files Modified in Migration Attempt

### Successfully Migrated ✅
- `ops/examples/peek_*.py`
- `ops/examples/vol_term_repo_first.py`
- `ops/examples/ex_1.py`
- `ops/jobs/check_coverage.py`
- `ops/jobs/contracts_snapshot*.py`
- `ops/jobs/resample_option_minute_to_daily.py`
- `atm_vol_term.py`

### Reverted Due to API Issues 🔄
- `conftest.py` fixtures
- `conftest_full.py` fixtures  
- `tests/unit/repositories/test_*.py`
- `tests/test_present_dates.py`

### Left As-Is (No Common Equivalent) 📍
- Files using `ibx_repos.contracts` (no equivalent in `common/`)
- Files using `ibx_repos` service classes
- Files with complex API dependencies

## Next Steps

1. **Document** the hybrid architecture in main README
2. **Test** that production services still work with `common/repositories`
3. **Consider** adding compatibility methods to `common/repositories` if needed
4. **Gradually** migrate remaining code as requirements evolve

## Repository API Comparison

| Feature | `ibx_repos` | `common/repositories` |
|---------|-------------|----------------------|
| **Save API** | `save(df, symbol, bar_size, ...)` | `save_daily_bars()` / `save_intraday_bars()` |
| **Load API** | `load(symbol, start_date, ...)` | `load(symbol=..., start_date=...)` |
| **Base Class** | Custom | `BaseRepository` (PyArrow) |
| **Partitioning** | Custom | Hive-style partitions |
| **Schema** | Flexible | Strongly typed |
| **Used By** | Tests, Legacy | Production, Services |

The current state provides **production stability** with `common/repositories` while maintaining **test stability** with `ibx_repos`.