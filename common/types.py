"""Custom type definitions for the trading platform."""

from typing import TypeVar, Union, List, Dict, Any, Protocol, runtime_checkable
from datetime import date, datetime
from decimal import Decimal
import pandas as pd
import pyarrow as pa


# Generic types
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Data types
Price = Union[Decimal, float]
Volume = int
Symbol = str
Timestamp = Union[date, datetime, str]
Currency = str

# Collection types
SymbolList = List[Symbol]
DateList = List[date]
PriceList = List[Price]
FilterList = List[tuple]

# Data structure types
BarDict = Dict[str, Any]  # Market bar data as dictionary
ContractDict = Dict[str, Any]  # Contract specification as dictionary
EventDict = Dict[str, Any]  # Event payload as dictionary

# DataFrame types (for better type hints)
BarDataFrame = pd.DataFrame  # DataFrame with OHLCV data
ChainDataFrame = pd.DataFrame  # DataFrame with option chain data
EarningsDataFrame = pd.DataFrame  # DataFrame with earnings data

# PyArrow types
ArrowTable = pa.Table
ArrowSchema = pa.Schema

# Import dataset module separately to handle version differences
try:
    import pyarrow.dataset as ds
    ArrowDataset = ds.Dataset
except (ImportError, AttributeError):
    # Fallback for older pyarrow versions
    ArrowDataset = Any

# Configuration types
ConfigDict = Dict[str, Any]
ConnectionParams = Dict[str, Union[str, int, float, bool]]


# Protocol definitions for interfaces
@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create object from dictionary."""
        ...


@runtime_checkable
class Cacheable(Protocol):
    """Protocol for objects that can be cached."""
    
    def cache_key(self) -> str:
        """Generate unique cache key for this object."""
        ...
    
    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if cached object is stale."""
        ...


@runtime_checkable
class Validatable(Protocol):
    """Protocol for objects that can be validated."""
    
    def validate(self) -> bool:
        """Validate object state."""
        ...
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        ...


@runtime_checkable
class Timestamped(Protocol):
    """Protocol for objects with timestamp information."""
    
    @property
    def timestamp(self) -> datetime:
        """Get object timestamp."""
        ...
    
    def is_recent(self, max_age_seconds: int = 300) -> bool:
        """Check if object is recent."""
        ...


@runtime_checkable
class Identifiable(Protocol):
    """Protocol for objects with unique identifiers."""
    
    @property
    def id(self) -> str:
        """Get unique identifier."""
        ...
    
    def __eq__(self, other: Any) -> bool:
        """Compare objects by ID."""
        ...
    
    def __hash__(self) -> int:
        """Hash object by ID."""
        ...


# Repository protocol types
@runtime_checkable
class Repository(Protocol[T]):
    """Base repository protocol."""
    
    def save(self, data: pd.DataFrame, **kwargs) -> None:
        """Save data to repository."""
        ...
    
    def load(self, filters: FilterList = None, **kwargs) -> pd.DataFrame:
        """Load data from repository."""
        ...
    
    def exists(self, **kwargs) -> bool:
        """Check if data exists."""
        ...
    
    def delete(self, **kwargs) -> None:
        """Delete data from repository."""
        ...


@runtime_checkable
class TimeSeriesRepository(Repository[T], Protocol):
    """Time series data repository protocol."""
    
    def load_date_range(
        self, 
        start_date: date, 
        end_date: date, 
        **kwargs
    ) -> pd.DataFrame:
        """Load data for date range."""
        ...
    
    def get_available_dates(self, **kwargs) -> DateList:
        """Get available dates."""
        ...


@runtime_checkable
class SymbolRepository(Repository[T], Protocol):
    """Symbol-based repository protocol."""
    
    def load_symbol(self, symbol: Symbol, **kwargs) -> pd.DataFrame:
        """Load data for specific symbol."""
        ...
    
    def get_symbols(self) -> SymbolList:
        """Get available symbols."""
        ...


# Event system types
@runtime_checkable
class EventPublisher(Protocol):
    """Protocol for event publishers."""
    
    def publish(self, topic: str, payload: EventDict, **kwargs) -> None:
        """Publish event to topic."""
        ...


@runtime_checkable
class EventConsumer(Protocol):
    """Protocol for event consumers."""
    
    def subscribe(self, topic: str, **kwargs) -> None:
        """Subscribe to topic."""
        ...
    
    def consume(self, max_events: int = None) -> List[EventDict]:
        """Consume events from subscribed topics."""
        ...


# API client types
@runtime_checkable
class APIClient(Protocol):
    """Protocol for API clients."""
    
    def connect(self, **kwargs) -> None:
        """Connect to API."""
        ...
    
    def disconnect(self) -> None:
        """Disconnect from API."""
        ...
    
    def is_connected(self) -> bool:
        """Check connection status."""
        ...


@runtime_checkable
class MarketDataClient(APIClient, Protocol):
    """Protocol for market data clients."""
    
    def get_historical_bars(
        self,
        contract: ContractDict,
        start_date: date,
        end_date: date,
        bar_size: str = "1 day"
    ) -> BarDataFrame:
        """Get historical bar data."""
        ...
    
    def get_option_chain(
        self,
        symbol: Symbol,
        expiry_date: date = None
    ) -> ChainDataFrame:
        """Get option chain data."""
        ...


# Utility types for common patterns
class NamedTuple(Protocol):
    """Protocol for named tuple-like objects."""
    
    def _asdict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...


# Type aliases for common union types
OptionalStr = Union[str, None]
OptionalInt = Union[int, None]
OptionalFloat = Union[float, None]
OptionalDate = Union[date, None]
OptionalDateTime = Union[datetime, None]
OptionalDecimal = Union[Decimal, None]

# JSON-like types
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONDict = Dict[str, JSONValue]
JSONList = List[JSONValue]

# File path types
FilePath = Union[str, 'PathLike[str]']
DirectoryPath = Union[str, 'PathLike[str]']

# Database types
QueryParams = Dict[str, Any]
WhereClause = str
OrderByClause = str

# Error handling types
ErrorCode = str
ErrorMessage = str
ErrorDetails = Dict[str, Any]
ValidationErrors = Dict[str, List[str]]

# Configuration types
EnvVar = str
ConfigKey = str
ConfigValue = Union[str, int, float, bool]
ConfigSection = Dict[ConfigKey, ConfigValue]

# Logging types
LogLevel = str
LogMessage = str
LogContext = Dict[str, Any]