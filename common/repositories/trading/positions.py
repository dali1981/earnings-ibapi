"""
Position repository for trading positions.
Manages position data with P&L tracking and portfolio analytics.
"""

import pandas as pd
import pyarrow as pa
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Any, Optional, Union

from .base import BaseTradingRepository
from common.schemas.trading import Position, EquityContract, OptionContract
from common.constants import SecType, OptionRight


class PositionRepository(BaseTradingRepository[Position]):
    """Repository for managing trading positions."""
    
    def __init__(self, data_dir: str):
        """Initialize position repository."""
        super().__init__(data_dir, "positions")
    
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for positions table."""
        return pa.schema([
            ("account", pa.string()),
            ("symbol", pa.string()),
            ("sec_type", pa.string()),
            ("position_size", pa.int32()),
            ("avg_cost", pa.float64()),
            ("market_price", pa.float64()),
            ("unrealized_pnl", pa.float64()),
            ("realized_pnl", pa.float64()),
            ("total_cost", pa.float64()),
            ("market_value", pa.float64()),
            ("strike", pa.float64()),  # Optional, for options
            ("right", pa.string()),   # Optional, for options
            ("expiry", pa.string()),  # Optional, for options (YYYY-MM-DD)
            ("multiplier", pa.int32()), # Optional, for options
            ("updated_at", pa.timestamp('us')),
            ("date", pa.date32()),    # Partition column
        ])
    
    def _get_partition_columns(self) -> List[str]:
        """Get partition columns for positions."""
        return ["date", "account"]
    
    def domain_normalize_data(self, position: Position) -> Dict[str, Any]:
        """Convert Position domain object to normalized dict."""
        now = datetime.utcnow()
        
        # Base position data
        data = {
            "account": position.account,
            "symbol": position.contract.symbol,
            "sec_type": "STK" if isinstance(position.contract, EquityContract) else "OPT",
            "position_size": position.position_size,
            "avg_cost": float(position.avg_cost),
            "market_price": float(position.market_price),
            "realized_pnl": float(position.realized_pnl),
            "updated_at": now,
            "date": now.date()
        }
        
        # Calculate derived fields
        if isinstance(position.contract, OptionContract):
            # Options: multiply by contract multiplier
            multiplier = int(getattr(position.contract, 'multiplier', 100))
            data["total_cost"] = float(position.avg_cost) * abs(position.position_size) * multiplier
            data["market_value"] = float(position.market_price) * position.position_size * multiplier
            data["unrealized_pnl"] = (float(position.market_price) - float(position.avg_cost)) * position.position_size * multiplier
            
            # Option-specific fields
            data["strike"] = float(position.contract.strike)
            data["right"] = "C" if position.contract.right == OptionRight.CALL else "P"
            data["expiry"] = position.contract.expiry.strftime('%Y-%m-%d') if hasattr(position.contract.expiry, 'strftime') else str(position.contract.expiry)
            data["multiplier"] = multiplier
        else:
            # Stocks: direct multiplication
            data["total_cost"] = float(position.avg_cost) * abs(position.position_size)
            data["market_value"] = float(position.market_price) * position.position_size
            data["unrealized_pnl"] = (float(position.market_price) - float(position.avg_cost)) * position.position_size
            
            # Set option fields to null for stocks
            data["strike"] = None
            data["right"] = None
            data["expiry"] = None
            data["multiplier"] = None
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame for storage (required by BaseRepository)."""
        # This method is used by the base save() method
        # For positions, we don't need additional normalization beyond domain_normalize_data
        return data
    
    def get_position_by_contract(
        self, 
        contract: Union[EquityContract, OptionContract], 
        account: str
    ) -> Optional[Position]:
        """Get position for specific contract and account."""
        try:
            # Build filter conditions
            filters = [
                ("account", "=", account),
                ("symbol", "=", contract.symbol),
                ("sec_type", "=", "STK" if isinstance(contract, EquityContract) else "OPT")
            ]
            
            # Add option-specific filters
            if isinstance(contract, OptionContract):
                filters.extend([
                    ("strike", "=", float(contract.strike)),
                    ("right", "=", "C" if contract.right == OptionRight.CALL else "P"),
                    ("expiry", "=", contract.expiry.strftime('%Y-%m-%d') if hasattr(contract.expiry, 'strftime') else str(contract.expiry))
                ])
            
            df = self.read_data(filters=filters, limit=1)
            
            if df.empty:
                return None
            
            return self._dataframe_to_position(df.iloc[0])
            
        except Exception as e:
            self.logger.error(f"Error retrieving position for {contract.symbol}: {e}")
            return None
    
    def get_positions_by_account(self, account: str) -> List[Position]:
        """Get all positions for an account."""
        try:
            df = self.read_data(filters=[("account", "=", account)])
            
            if df.empty:
                return []
            
            return [self._dataframe_to_position(row) for _, row in df.iterrows()]
            
        except Exception as e:
            self.logger.error(f"Error retrieving positions for account {account}: {e}")
            return []
    
    def update_position_prices(
        self, 
        contract: Union[EquityContract, OptionContract],
        new_price: Decimal,
        account: str
    ) -> None:
        """Update position with new market price."""
        try:
            # Get existing position
            existing_position = self.get_position_by_contract(contract, account)
            if not existing_position:
                self.logger.warning(f"Position not found for {contract.symbol} in account {account}")
                return
            
            # Update market price
            existing_position.market_price = new_price
            
            # Save updated position
            self.domain_save(existing_position)
            
        except Exception as e:
            self.logger.error(f"Error updating position prices for {contract.symbol}: {e}")
    
    def close_position(
        self, 
        contract: Union[EquityContract, OptionContract],
        realized_pnl: Decimal,
        account: str
    ) -> None:
        """Close a position (set position_size to 0)."""
        try:
            # Create closed position record
            closed_position_data = {
                "account": account,
                "symbol": contract.symbol,
                "sec_type": "STK" if isinstance(contract, EquityContract) else "OPT",
                "position_size": 0,
                "avg_cost": 0.0,
                "market_price": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": float(realized_pnl),
                "total_cost": 0.0,
                "market_value": 0.0,
                "updated_at": datetime.utcnow(),
                "date": datetime.utcnow().date()
            }
            
            # Add option-specific fields if needed
            if isinstance(contract, OptionContract):
                closed_position_data.update({
                    "strike": float(contract.strike),
                    "right": "C" if contract.right == OptionRight.CALL else "P",
                    "expiry": contract.expiry.strftime('%Y-%m-%d') if hasattr(contract.expiry, 'strftime') else str(contract.expiry),
                    "multiplier": getattr(contract, 'multiplier', 100)
                })
            else:
                closed_position_data.update({
                    "strike": None,
                    "right": None,
                    "expiry": None,
                    "multiplier": None
                })
            
            # Save closed position
            df = pd.DataFrame([closed_position_data])
            self.save(df)
            
        except Exception as e:
            self.logger.error(f"Error closing position for {contract.symbol}: {e}")
    
    def get_portfolio_summary(self, account: str) -> Dict[str, float]:
        """Get portfolio summary statistics for an account."""
        try:
            df = self.read_data(filters=[("account", "=", account)])
            
            if df.empty:
                return {
                    "total_positions": 0,
                    "total_cost": 0.0,
                    "total_market_value": 0.0,
                    "total_unrealized_pnl": 0.0,
                    "total_realized_pnl": 0.0,
                    "total_pnl": 0.0
                }
            
            # Filter out closed positions (position_size == 0) for active portfolio metrics
            active_positions = df[df["position_size"] != 0]
            
            summary = {
                "total_positions": len(active_positions),
                "total_cost": float(active_positions["total_cost"].sum()) if not active_positions.empty else 0.0,
                "total_market_value": float(active_positions["market_value"].sum()) if not active_positions.empty else 0.0,
                "total_unrealized_pnl": float(active_positions["unrealized_pnl"].sum()) if not active_positions.empty else 0.0,
                "total_realized_pnl": float(df["realized_pnl"].sum()),  # Include all realized P&L
                "total_pnl": 0.0
            }
            
            # Calculate total P&L
            summary["total_pnl"] = summary["total_unrealized_pnl"] + summary["total_realized_pnl"]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio summary for account {account}: {e}")
            return {
                "total_positions": 0,
                "total_cost": 0.0,
                "total_market_value": 0.0,
                "total_unrealized_pnl": 0.0,
                "total_realized_pnl": 0.0,
                "total_pnl": 0.0
            }
    
    def _dataframe_to_position(self, row: pd.Series) -> Position:
        """Convert DataFrame row to Position object."""
        # Create contract based on sec_type
        if row["sec_type"] == "STK":
            contract = EquityContract(
                symbol=row["symbol"],
                sec_type=SecType.STOCK
            )
        else:  # OPT
            contract = OptionContract(
                symbol=row["symbol"],
                sec_type=SecType.OPTION,
                strike=Decimal(str(row["strike"])),
                right=OptionRight.CALL if row["right"] == "C" else OptionRight.PUT,
                expiry=pd.to_datetime(row["expiry"]).date() if pd.notna(row["expiry"]) else date.today(),
                multiplier=int(row["multiplier"]) if pd.notna(row["multiplier"]) else 100
            )
        
        return Position(
            contract=contract,
            position_size=int(row["position_size"]),
            avg_cost=Decimal(str(row["avg_cost"])),
            market_price=Decimal(str(row["market_price"])),
            account=row["account"],
            realized_pnl=Decimal(str(row["realized_pnl"]))
        )