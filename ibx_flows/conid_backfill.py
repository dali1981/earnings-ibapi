from __future__ import annotations
import pandas as pd
try:
    from ibx import ContractDetailsService  # type: ignore
except Exception:
    from ibx.services import ContractDetailsService  # type: ignore
try:
    from ibx import make_option  # type: ignore
except Exception:
    from ibx.contracts import make_option  # type: ignore
def resolve_conids(contracts: pd.DataFrame, cds: ContractDetailsService, timeout: float = 20.0) -> pd.DataFrame:
    if contracts is None or contracts.empty:
        return pd.DataFrame(columns=["underlying","expiry","right","strike","conid"])
    uniq = contracts[["underlying","expiry","right","strike"]].drop_duplicates().copy()
    out = []
    for _, r in uniq.iterrows():
        exp_str = pd.Timestamp(r["expiry"]).strftime("%Y%m%d")
        opt = make_option(r["underlying"], exp_str, float(r["strike"]), r["right"])
        try:
            cds_list = cds.fetch(opt, timeout=timeout); conid = cds_list[0].contract.conId if cds_list else None
        except Exception:
            conid = None
        out.append({"underlying": r["underlying"], "expiry": pd.to_datetime(r["expiry"]).date(), "right": r["right"], "strike": float(r["strike"]), "conid": conid})
    return pd.DataFrame(out)
def attach_conids_to_bars(bars: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty: return bars
    key = ["underlying","expiry","right","strike"]
    m = mapping[key + ["conid"]].drop_duplicates()
    out = bars.merge(m, on=key, how="left", suffixes=("","_map"))
    out["conid"] = out["conid"].combine_first(out["conid_map"])
    return out.drop(columns=[c for c in ["conid_map"] if c in out.columns])
