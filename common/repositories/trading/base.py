"""
Base repository class for trading data persistence with PyArrow/Parquet backend.
Extends the existing BaseRepository with trading-specific functionality.
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional, TypeVar, Generic
import pandas as pd
import pyarrow as pa

from ..base import BaseRepository

T = TypeVar('T')


class BaseTradingRepository(BaseRepository, Generic[T]):
    """Base class for trading-specific repositories."""
    
    def __init__(self, data_dir: str, table_name: str):
        # Use existing BaseRepository interface
        super().__init__(data_dir, table_name)
        self.table_name = table_name
    
    @abstractmethod
    def domain_normalize_data(self, data: T) -> Dict[str, Any]:
        """Convert domain object to dictionary for storage."""
        pass
    
    def domain_save(self, data: T, **kwargs) -> None:
        """Save trading domain object with proper partitioning."""
        normalized = self.domain_normalize_data(data)
        df = pd.DataFrame([normalized])
        self.save(df, **kwargs)
    
    def domain_save_batch(self, data_list: List[T], **kwargs) -> None:
        """Save multiple domain objects efficiently."""
        normalized_list = [self.domain_normalize_data(item) for item in data_list]
        df = pd.DataFrame(normalized_list)
        self.save(df, **kwargs)