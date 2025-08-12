from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd

SelectionMode = Literal["k_around_atm","moneyness_bands"]

@dataclass
class StrikeSelectionConfig:
    expiries_max: int = 8
    strikes_per_side: int = 6
    moneyness_low: float = 0.9
    moneyness_high: float = 1.1

def _nearest_expiries(chain: pd.DataFrame, n: int) -> pd.DataFrame:
    chain = chain.sort_values(["expiry"])
    uniq = chain["expiry"].drop_duplicates().iloc[:n]
    return chain[chain["expiry"].isin(uniq)]

def select_contracts(chain_snapshot: pd.DataFrame, spot_price: float, cfg: StrikeSelectionConfig, mode: SelectionMode="k_around_atm") -> pd.DataFrame:
    df = chain_snapshot.copy()
    df = _nearest_expiries(df, cfg.expiries_max)
    if mode == "k_around_atm":
        def pick(g: pd.DataFrame) -> pd.DataFrame:
            strikes = sorted(g["strike"].unique())
            if not strikes: return g.head(0)
            center = min(strikes, key=lambda k: abs(k - spot_price))
            allk = sorted(strikes); idx = allk.index(center)
            lo = max(0, idx - cfg.strikes_per_side); hi = min(len(allk), idx + cfg.strikes_per_side + 1)
            keep = set(allk[lo:hi])
            return g[g["strike"].isin(keep)]
        return df.groupby(["expiry","right"], group_keys=False).apply(pick).reset_index(drop=True)
    elif mode == "moneyness_bands":
        lowK = spot_price / cfg.moneyness_high; highK = spot_price / cfg.moneyness_low
        return df[(df["strike"] >= lowK) & (df["strike"] <= highK)].copy()
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
