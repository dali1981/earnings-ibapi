from __future__ import annotations
import pandas as pd
try:
    from ibx.services import MatchingSymbolService  # type: ignore

except Exception:
    from ibx.futures import IBFuture  # type: ignore

class IBSymbolSearch:
    def __init__(self, rt):
        self.service = MatchingSymbolService(rt)

    def search(self, pattern: str, timeout: float = 10.0) -> pd.DataFrame:
        data = self.service.fetch(pattern)
        rows = []
        for d in data or []:
            c = getattr(d, "contract", None)
            if c is None and isinstance(d, dict): c = d.get("contract")
            if c is None: continue
            rows.append({
                "symbol": getattr(c,"symbol",None),
                "conid": getattr(c,"conId",None),
                "secType": getattr(c,"secType",None),
                "currency": getattr(c,"currency",None),
                "primaryExchange": getattr(c,"primaryExchange",None),
                "derivativeSecTypes": d.get("derivativeSecTypes",None),
            })
        return pd.DataFrame(rows)
