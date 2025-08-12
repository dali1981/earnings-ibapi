import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

import pandas as pd
from ibapi.contract import Contract

from ib_client import IBClient
from repos.equities import EquityBarRepository
from request_sequencer import RequestSequencer
from response_sequencer import ResponseSequencer
from sink import Sink, EquityBarSink
from utils import RateLimiter, make_stock, make_option
from ibx_time import ib_end_datetime
from config import IB_HOST, IB_PORT, IB_CLIENT_IDS, DATA_ROOT


@dataclass
class Config:
    symbol: str
    date: dt.date
    host: str = IB_HOST
    port: int = IB_PORT
    client_id: int = IB_CLIENT_IDS["streamer"]
    days_before: int = 30
    days_after: int = 5
    out_dir: Path = DATA_ROOT

class EarningsStreamer:
    # symbol: str
    # earnings_date: dt.date
    # host: str = "127.0.0.1"
    # port: int = 7497
    # client_id: int = 9
    # exchange: str = "SMART"
    # currency: str = "USD"
    # strikes_width: int = 3
    # minute_window: Tuple[int, int] = (-2, 2)
    # days_before: int = 30
    # days_after: int = 5
    # out_root: Path = Path("./earnings_data")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sequencer = RequestSequencer()
        self._sink = Sink(self._dir())
        self._limiter: RateLimiter = field(default_factory=RateLimiter, init=False, repr=False)
        self._resp = ResponseSequencer()
        self._client = IBClient(self.sequencer, self._resp, self._sink, self._limiter)

        # Repositories
        # opt_repo = BarRepository(base_path='data/bars')
        eq_repo = EquityBarRepository(base_path=self._dir())
        self.equity_sink = EquityBarSink(eq_repo)


    def _dir(self) -> Path:
        p = self.cfg.out_dir / f"{self.cfg.symbol}_{self.cfg.date:%Y%m%d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    import logging, ibapi.utils
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s %(lineno)d %(message)s")
    # ibapi.utils.logger.setLevel(logging.DEBUG)

    # ——— Connect and launch all requests ——————————————

    def run(self):
        self._client.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)


        # schedule requests when API is ready
        # self.sequencer.when_ready(self.send_contract_details)
        # self.sequencer.when_ready(self.send_daily_bars)
        # self.sequencer.when_ready(self.send_minute_bars)
        # self.sequencer.when_ready(self.send_secdef_pre)
        # self.sequencer.when_ready(self.send_secdef_post)
        self.sequencer.when_ready(self.schedule_chain_snapshot)
        # disconnect once all are done
        self.sequencer.register_on_all_done(self._client.disconnect)

        self._client.run()  # starts the socket in a background thread

    def send_contract_details(self, _first_id: int):
        # Step 1: request contract details to obtain conId for underlying
        stk_contract = make_stock(self.cfg.symbol)
        rid_cd_pre = self.sequencer.next("contractDetails_pre")
        self._client.reqContractDetails(rid_cd_pre, stk_contract)

    def send_daily_bars(self, _first_id: int):
        stk = make_stock(self.cfg.symbol)
        # Step 2: request daily bars for the underlying stock
        end = self.cfg.date + dt.timedelta(days=self.cfg.days_after)
        dur = f"{self.cfg.days_before + self.cfg.days_after} D"
        self.request_historical_bars(stk, end, dur, "1 day", "TRADES", False, self.equity_sink)


    def send_minute_bars(self, _first_id: int):
        stk = make_stock(self.cfg.symbol)
        rid = self.sequencer.next("earn1m")
        end_dt = dt.datetime.combine(self.cfg.date, dt.time(23,59,59))
        # FIXED: Remove hyphen=True - IB API requires space separator
        end = ib_end_datetime(end_dt, tz="UTC")
        self._client.reqHistoricalData(rid, stk, end, "1 D", "1 min", "TRADES", 0, 1, False, [])

    def send_secdef_pre(self, _first_id: int):
        rid = self.sequencer.next("secdef_pre")
        self._client.reqSecDefOptParams(rid, self.cfg.symbol, "", "STK", 8314)

    def send_secdef_post(self, _first_id: int):
        rid = self.sequencer.next("secdef_post")
        self._client.reqSecDefOptParams(rid, self.cfg.symbol, "", "STK", 8314)

    # Helpers for within callbacks (e.g., in Sink handlers)
    def schedule_chain_snapshot(self, _first_id: int):
        strikes = [("20250815", 250)]
        # request minute option bars for each strike+expiry combination
        for expiry, strike in strikes:
            opt = make_option(self.cfg.symbol, expiry, strike, "C")
            # self._limiter.acquire()
            rid = self.sequencer.next(f"opt_{expiry}_{strike}")
            # self._client.reqMktData(rid, opt, "100,101,104,106,107,125", False, False, [])
            self._client.reqMktData(rid, opt, "100,101,104,106", False, False, [])

    # — actual requests —
    def send_requests(self, _first_id: int):
        stk = make_stock(self.cfg.symbol)
        # daily bars
        rid = self.sequencer.next("daily");
        end_dt = dt.datetime.combine(self.cfg.date + dt.timedelta(days=self.cfg.days_after), dt.time(0,0,0))
        # FIXED: Remove hyphen=True - IB API requires space separator
        end = ib_end_datetime(end_dt, tz="UTC")
        dur = f"{self.cfg.days_before + self.cfg.days_after} D"
        self._client.reqHistoricalData(rid, stk, end, dur, "1 day", "TRADES", 0, 1, False, [])
        # minute bars earnings day
        rid = self.sequencer.next("earn1m");
        # FIXED: Remove hyphen=True - IB API requires space separator
        end_intraday = ib_end_datetime(dt.datetime.combine(self.cfg.date, dt.time(23,59,59)), tz="UTC")
        self._client.reqHistoricalData(rid, stk, end_intraday, "1 D", "1 min", "TRADES", 0, 1, False, [])
        # option secdef pre / post
        for off, tag in [(-1, "pre"), (1, "post")]:
            rid = self.sequencer.next(f"secdef_{tag}");
            self._client.reqSecDefOptParams(rid, self.cfg.symbol, "", "STK", 8314)

        self.sequencer.register_on_all_done(self._client.disconnect)

    # ——— Bulk request registration ————————————————

    def _stk(self) -> Contract:
        return make_stock(self.symbol, self.exchange, self.currency)

    def request_historical_bars(
            self,
            contract: Contract,
            end_datetime: Optional[dt.datetime | dt.date],
            duration: str,
            bar_size: str,
            what_to_show: str,
            use_rth: bool,
            callback: Callable[[pd.DataFrame], None]
    ) -> int:
        """
        1) Wait for a fresh reqId via RequestSequencer
        2) Register callback and metadata with ResponseSequencer
        3) Invoke IB reqHistoricalData
        Returns the reqId used.
        """

        symbol = getattr(contract, 'symbol', None)
        # Acquire a valid reqId
        req_id = self.sequencer.next(f'hist_{symbol}_{bar_size}_{what_to_show}')
        # Prepare metadata for later enrichment
        meta = {
            'symbol': symbol,
            'bar_size': bar_size,
            'what_to_show': what_to_show,
            'request_time': dt.date.today(),
        }

        # enrich for options so sinks/repos can persist underlying, expiry, strike, right
        sec_type = getattr(contract, 'secType', None)
        if (sec_type or '').upper() in ('OPT', 'FOP'):
            expiry_raw = getattr(contract, 'lastTradeDateOrContractMonth', None)
            # normalize expiry to a date if possible
            expiry = None
            if expiry_raw:
                s = str(expiry_raw)
                try:
                    if len(s) >= 8:
                        expiry = dt.datetime.strptime(s[:8], '%Y%m%d').date()
                    elif len(s) == 6:
                        # yyyymm -> set to first of month
                        expiry = dt.datetime.strptime(s + '01', '%Y%m%d').date()
                except Exception:
                    expiry = None
            meta.update({
                'underlying': getattr(contract, 'symbol', None),
                'expiry': expiry or expiry_raw,
                'strike': getattr(contract, 'strike', None),
                'right': getattr(contract, 'right', None),
            })

        # Register callback and metadata
        self._resp.add(req_id, callback, meta)
        # Format end datetime string
        # FIXED: Remove hyphen=True - IB API requires space separator
        end_str = ib_end_datetime(end_datetime, tz='UTC') if end_datetime else ''
        # Issue the IB request
        self._client.reqHistoricalData(
            req_id,
            contract,
            end_str,
            duration,
            bar_size,
            what_to_show,
            int(use_rth),
            1,  # formatDate,
            False,  # keepUpToDate
            []
        )
        return req_id
