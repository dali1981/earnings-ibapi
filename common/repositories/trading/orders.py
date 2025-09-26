"""
Order repository for trading order data management.
Handles persistence and retrieval of trading orders using PyArrow/Parquet.
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa

from .base import BaseTradingRepository
from ...schemas.trading import (
    Order, OrderStatus, OrderAction, OrderType, TimeInForce, 
    EquityContract, OptionContract
)
from ...constants import SecType, OptionRight


class OrderRepository(BaseTradingRepository[Order]):
    """Repository for order data management."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "orders")
    
    def _create_schema(self) -> pa.Schema:
        """Define the PyArrow schema for order data."""
        return pa.schema([
            ("order_id", pa.int64()),
            ("client_order_id", pa.string()),
            ("symbol", pa.string()),
            ("sec_type", pa.string()),
            ("action", pa.string()),
            ("quantity", pa.int32()),
            ("order_type", pa.string()),
            ("limit_price", pa.float64()),
            ("stop_price", pa.float64()),
            ("time_in_force", pa.string()),
            ("account", pa.string()),
            ("parent_order_id", pa.int64()),
            ("status", pa.string()),
            ("created_at", pa.timestamp('us')),
            ("updated_at", pa.timestamp('us')),
            ("date", pa.date32()),  # Partition column
        ])
    
    def _get_partition_columns(self, **kwargs) -> List[str]:
        """Get partition columns for the data."""
        return ["date"]
    
    def _normalize_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Normalize and validate DataFrame for storage."""
        # Add required columns if missing
        required_columns = [field.name for field in self._schema]
        for col in required_columns:
            if col not in df.columns:
                if col == "date":
                    df[col] = pd.to_datetime(df["created_at"]).dt.date
                else:
                    df[col] = None
        
        # Ensure proper data types
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
        if "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            
        return df
    
    def domain_normalize_data(self, order: Order) -> Dict[str, Any]:
        """Convert Order object to dictionary for storage."""
        return {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.contract.symbol,
            "sec_type": order.contract.sec_type,
            "action": order.action,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "time_in_force": order.time_in_force,
            "account": order.account,
            "parent_order_id": order.parent_order_id,
            "status": OrderStatus.PENDING_SUBMIT,  # Default status on creation
            "created_at": order.created_at,
            "updated_at": order.created_at,
            "date": order.created_at.date(),
        }
    
    def update_order_status(self, order_id: int, status: OrderStatus, 
                           filled_qty: int = 0, avg_fill_price: Optional[float] = None) -> None:
        """Update order status in the repository."""
        filters = [("order_id", "==", order_id)]
        table = self.load(filters=filters)
        
        if table.num_rows > 0:
            df = table.to_pandas()
            # Update status and other fields
            df.loc[df['order_id'] == order_id, 'status'] = status
            df.loc[df['order_id'] == order_id, 'updated_at'] = datetime.utcnow()
            
            if filled_qty > 0:
                # Add filled_qty column if it doesn't exist
                if 'filled_qty' not in df.columns:
                    df['filled_qty'] = 0
                df.loc[df['order_id'] == order_id, 'filled_qty'] = filled_qty
            
            if avg_fill_price:
                # Add avg_fill_price column if it doesn't exist
                if 'avg_fill_price' not in df.columns:
                    df['avg_fill_price'] = None
                df.loc[df['order_id'] == order_id, 'avg_fill_price'] = avg_fill_price
            
            # Save back using parent save method
            self.save(df)
    
    def get_orders(self, account: Optional[str] = None, 
                  status: Optional[OrderStatus] = None,
                  date_range: Optional[tuple] = None) -> List[Order]:
        """Retrieve orders with optional filters."""
        filters = []
        
        if account:
            filters.append(("account", "==", account))
        if status:
            filters.append(("status", "==", status))
        if date_range:
            start_date, end_date = date_range
            filters.append(("date", "between", (start_date, end_date)))
        
        table = self.load(filters=filters)
        df = table.to_pandas()
        return [self._df_row_to_order(row) for _, row in df.iterrows()]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Retrieve orders for a specific symbol."""
        filters = [("symbol", "==", symbol)]
        table = self.load(filters=filters)
        df = table.to_pandas()
        return [self._df_row_to_order(row) for _, row in df.iterrows()]
    
    def _df_row_to_order(self, row: pd.Series) -> Order:
        """Convert DataFrame row back to Order object."""
        # Reconstruct contract based on sec_type
        if row['sec_type'] == SecType.STOCK:
            contract = EquityContract(
                symbol=row['symbol'],
                exchange="SMART",  # Default
                currency="USD"     # Default
            )
        else:
            # For now, assume equity contracts only
            # Option contract reconstruction would need additional fields
            contract = EquityContract(
                symbol=row['symbol'],
                exchange="SMART",
                currency="USD"
            )
        
        return Order(
            order_id=row['order_id'] if pd.notna(row['order_id']) else None,
            client_order_id=row['client_order_id'],
            contract=contract,
            action=OrderAction(row['action']),
            quantity=row['quantity'],
            order_type=OrderType(row['order_type']),
            limit_price=float(row['limit_price']) if pd.notna(row['limit_price']) else None,
            stop_price=float(row['stop_price']) if pd.notna(row['stop_price']) else None,
            time_in_force=TimeInForce(row['time_in_force']),
            account=row['account'] if pd.notna(row['account']) else None,
            parent_order_id=row['parent_order_id'] if pd.notna(row['parent_order_id']) else None,
            created_at=row['created_at']
        )