from ibx import IBRuntime
from ibx_repos import (
    EquityBarRepository, OptionBarRepository, OptionChainSnapshotRepository,
    IBXDataService, OptionMeta, FetchSpec
)

symbol = "CMCL"

with IBRuntime(port=4002, client_id=7) as rt:
    eq_repo = EquityBarRepository("data/equity_bars")
    opt_repo = OptionBarRepository("data/option_bars")
    ch_repo  = OptionChainSnapshotRepository("data/option_chains")
    svc = IBXDataService(rt, eq_repo, opt_repo, ch_repo)

    # 1) Equities
    svc.fetch_equity_bars(FetchSpec(symbol=symbol, bar_size="5 mins", duration="2 D"))

    # 2) Chain snapshot (secdef)
    svc.snapshot_chain(symbol)

    # 3) One option
    meta = OptionMeta(underlying=symbol, expiry="20250919", strike=200, right="C", bar_size="1 min")
    svc.fetch_option_bars(meta, duration="1 D")
