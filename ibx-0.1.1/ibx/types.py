from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

Right = Literal["C", "P"]

@dataclass(frozen=True)
class OptionKey:
    underlying: str
    expiry: date
    right: Right
    strike: float

@dataclass(frozen=True)
class ContractMeta:
    conid: int
    local_symbol: str
    trading_class: str
    exchange: str
    currency: str
    multiplier: int
    key: OptionKey