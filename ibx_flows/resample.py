from __future__ import annotations
import pandas as pd
def resample_option_minute_to_daily(minute_bars: pd.DataFrame) -> pd.DataFrame:
    if minute_bars is None or minute_bars.empty:
        return pd.DataFrame(columns=["underlying","expiry","right","strike","conid","bar_size","time","open","high","low","close","volume","wap","trades"])
    df = minute_bars.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date
    if "wap" not in df.columns: df["wap"] = pd.NA
    if "trades" not in df.columns: df["trades"] = pd.NA
    if "conid" not in df.columns: df["conid"] = pd.NA
    keys = ["underlying","expiry","right","strike","date"]
    df = df.sort_values(["underlying","expiry","right","strike","time"])
    def _agg(g: pd.DataFrame) -> pd.Series:
        first = g.iloc[0]; last = g.iloc[-1]
        open_ = first["open"]; high_ = g["high"].max(); low_ = g["low"].min(); close_ = last["close"]
        vol = g["volume"].sum(min_count=1)
        vwap = (g["wap"].fillna(0) * g["volume"].fillna(0)).sum()/vol if vol and vol>0 and g["wap"].notna().any() else pd.NA
        trades = g["trades"].sum(min_count=1) if g["trades"].notna().any() else pd.NA
        conid = g["conid"].dropna().iloc[0] if g["conid"].notna().any() else pd.NA
        return pd.Series({"time": pd.to_datetime(str(first["date"])) + pd.Timedelta(hours=23, minutes=59, seconds=59), "open": open_, "high": high_, "low": low_, "close": close_, "volume": vol, "wap": vwap, "trades": trades, "conid": conid})
    daily = df.groupby(keys, as_index=False).apply(_agg).reset_index(drop=True)
    daily["bar_size"] = "1 day"
    cols = ["underlying","expiry","right","strike","conid","bar_size","time","open","high","low","close","volume","wap","trades"]
    return daily[cols]
