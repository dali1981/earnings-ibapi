# ib_vol_history.py
# Pulls:
#  A) Underlying 30d Historical Vol (HISTORICAL_VOLATILITY) and 30d Option IV (OPTION_IMPLIED_VOLATILITY)
#  B) Option price bars (MIDPOINT), then computes per-bar contract IV via Black–Scholes
import math
import time
import logging
from dataclasses import dataclass
from idlelib.run import install_recursionlimit_wrappers
from typing import List, Dict, Tuple

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
from ibapi.utils import iswrapper

# ---------- Black–Scholes helpers ----------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_price(und: float, K: float, T: float, r: float, q: float, vol: float, right: str) -> float:
    # right: "C" or "P" ; all vols annualized; T in years
    if vol <= 0 or T <= 0 or und <= 0 or K <= 0:
        return max(0.0, (und*math.exp(-q*T) - K*math.exp(-r*T)) if right == "C" else (K*math.exp(-r*T) - und*math.exp(-q*T)))
    sqrtT = math.sqrt(T)
    d1 = (math.log(und/K) + (r - q + 0.5*vol*vol)*T) / (vol*sqrtT)
    d2 = d1 - vol*sqrtT
    if right == "C":
        return und*math.exp(-q*T) * _norm_cdf(d1) - K*math.exp(-r*T) * _norm_cdf(d2)
    else:
        return K*math.exp(-r*T) * _norm_cdf(-d2) - und*math.exp(-q*T) * _norm_cdf(-d1)

def bs_vega(und: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    if vol <= 0 or T <= 0 or und <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(und/K) + (r - q + 0.5*vol*vol)*T) / (vol*sqrtT)
    return und * math.exp(-q*T) * _norm_pdf(d1) * sqrtT

def implied_vol_newton(price: float, und: float, K: float, T: float, r: float, q: float, right: str,
                       v0: float = 0.3, tol: float = 1e-6, max_iter: int = 100) -> float | None:
    v = max(1e-6, v0)
    for _ in range(max_iter):
        px = bs_price(und, K, T, r, q, v, right)
        diff = px - price
        if abs(diff) < tol:
            return max(v, 0.0)
        veg = bs_vega(und, K, T, r, q, v)
        if veg <= 1e-12:
            break
        v = max(1e-6, v - diff / veg)
    return None

# ---------- IB setup ----------
def make_stock(symbol: str, exchange="SMART", currency="USD", primaryExchange=None) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    if primaryExchange:
        c.primaryExchange = primaryExchange
    return c

@dataclass
class OptionSpec:
    symbol: str
    yyyymmdd: str
    strike: float
    right: str  # "C" or "P"

def make_option(spec: OptionSpec) -> Contract:
    c = Contract()
    c.symbol = spec.symbol
    c.secType = "OPT"
    c.exchange = "SMART"
    c.currency = "USD"
    c.lastTradeDateOrContractMonth = spec.yyyymmdd
    c.strike = float(spec.strike)
    c.right = spec.right
    c.multiplier = "100"
    return c

class HistTap(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.log = logging.getLogger("HistTap")
        self._bars: Dict[int, List[Tuple[str, float]]] = {}  # reqId -> [(datetime, close)]
        self._done: Dict[int, bool] = {}

    def error(self, reqId:int, code:int, msg:str, advancedOrderRejectJson:str=""):
        self.log.error(f"ERR code={code} reqId={reqId} msg={msg}")

    def historicalData(self, reqId:int, bar: BarData):
        self._bars.setdefault(reqId, []).append((bar.date, float(bar.close)))

    def historicalDataEnd(self, reqId:int, start:str, end:str):
        self._done[reqId] = True
        self.log.info(f"historicalDataEnd reqId={reqId} start={start} end={end}")

    @iswrapper
    def nextValidId(self, orderId: int):
        self.log.info(f"Next valid ID: {orderId}")
        # This is where you can set up your request IDs if needed
        # self._next_valid_id = orderId

def wait_done(app: HistTap, req_ids: List[int], timeout_sec: float = 30.0):
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if all(app._done.get(rid, False) for rid in req_ids):
            return
        time.sleep(0.1)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = HistTap()
    app.connect("127.0.0.1", 4002, clientId=77)
    time.sleep(1)


    # --- A) Underlying historical 30d HV & 30d OIV (IB-defined) ---
    stk = make_stock("AMD", primaryExchange="NASDAQ")
    rid_hv = 1001
    rid_iv = 1002
    # duration can be "1 Y", "6 M", etc.  whatToShow must be these strings on the UNDERLYING:
    #   - "HISTORICAL_VOLATILITY"      -> IB 30-day historical (realized) vol
    #   - "OPTION_IMPLIED_VOLATILITY"  -> IB 30-day at-the-money implied vol


    for rid, what in [(rid_hv, "HISTORICAL_VOLATILITY"), (rid_iv, "OPTION_IMPLIED_VOLATILITY")]:
        app.reqHistoricalData(
            rid, stk, endDateTime="", durationStr="1 Y",
            barSizeSetting="1 day", whatToShow=what,
            useRTH=0, formatDate=1, keepUpToDate=False, chartOptions=[]
        )

    # --- B) Contract price bars + compute contract IV time-series ---
    bar_size_setting = "1 hour"
    historical_duration = "1 M"  # 6 months of hourly bars

    spec = OptionSpec("AMD", "20250919", 175, "C")
    opt = make_option(spec)
    rid_opt = 2001
    app.reqHistoricalData(
        rid_opt, opt, endDateTime="", durationStr=historical_duration,
        barSizeSetting=bar_size_setting, whatToShow="MIDPOINT",
        useRTH=0, formatDate=1, keepUpToDate=False, chartOptions=[]
    )

    # Also need UNDERLYING prices for the same period (to solve IV)
    rid_stk_px = 2002
    app.reqHistoricalData(
        rid_stk_px, stk, endDateTime="", durationStr=historical_duration,
        barSizeSetting=bar_size_setting, whatToShow="TRADES",
        useRTH=0, formatDate=1, keepUpToDate=False, chartOptions=[]
    )

    app.run()

    wait_done(app, [rid_hv, rid_iv, rid_opt, rid_stk_px], timeout_sec=60.0)

    hv_series = app._bars.get(rid_hv, [])
    iv_series = app._bars.get(rid_iv, [])
    opt_series = app._bars.get(rid_opt, [])
    und_series = app._bars.get(rid_stk_px, [])

    app.disconnect()

    # Align by date to compute per-contract IV
    # Choose simple constants; replace r,q with your own term structures if you have them
    r_annual = 0.03
    q_annual = 0.00

    # Build dicts date->price
    und_map = {d: px for d, px in und_series}
    # Time to expiry in years for each date
    from datetime import datetime, timezone

    def parse_date(s: str) -> datetime:
        # IB returns 'YYYYMMDD' or 'YYYYMMDD  HH:MM:SS' depending on bar size
        return datetime.strptime(s.split(' ')[0], "%Y%m%d").replace(tzinfo=timezone.utc)

    expiry = datetime.strptime(spec.yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)

    iv_points: List[Tuple[str, float | None]] = []
    for d, opt_px in opt_series:
        dt = parse_date(d)
        if dt >= expiry:
            continue
        und_px = und_map.get(d)
        if not und_px:
            iv_points.append((d, None))
            continue
        T_years = max(0.0, (expiry - dt).days / 365.25)
        iv = implied_vol_newton(
            price=opt_px, und=und_px, K=spec.strike, T=T_years, r=r_annual, q=q_annual, right=spec.right,
            v0=0.3
        )
        iv_points.append((d, iv))

    # Print small samples
    print("\nUNDERLYING 30d HV (head):")
    print(hv_series[:5])
    print("\nUNDERLYING 30d IV (head):")
    print(iv_series[:5])
    print(f"\n{spec.symbol} {spec.right}{spec.strike} {spec.yyyymmdd} contract IV (computed) head:")
    print(iv_points[:5])
    print("(None means solver failed or missing underlier price on that date.)")

if __name__ == "__main__":
    main()
