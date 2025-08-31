from __future__ import annotations

from app_settings.settings import get_settings
from celery import shared_task
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

# Typed models and logging helpers
from ops.schemas.option_contracts import (
    OptionSelection,
    OptionHistoricalArgs,
    SelectedContracts,
    HistoricalResult,
    SaveContractsResult,
)
from common.utils.logging_extra import (
    set_correlation_id,
    CorrelationIdAdapter,
)


@shared_task(
    pydantic=True,
    bind=True,
    name="option_chains.get_option_contracts_for_bars",
    queue="option_chains",
)
def get_option_contracts_for_bars_task(
    self,
    # Prefer a single selection payload to keep signatures simple
    selection: Optional[OptionSelection] = None,
    symbol: str | None = None,
    snapshot_date: str | None = None,
    expiry_filter: str | None = None,
    max_contracts: int = 50,
    rights: Optional[List[str]] = None,
    strikes_per_side: Optional[int] = None,
    selection_mode: Optional[str] = None,  # "count" | "moneyness" | "delta"
    expiry_count: Optional[int] = None,
    spot_price: Optional[float] = None,
    moneyness_targets: Optional[List[float]] = None,
    delta_targets: Optional[List[float]] = None,
    iv_assumption: Optional[float] = None,
    use_underlying_iv: Optional[bool] = None,  # reserved for future if underlying IV index stored
    root: str | None = None,
) -> dict:
    """
    Get option contracts for historical bars collection from aggregated chains.
    """
    import logging

    logger = logging.getLogger(__name__)

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    # Ensure we are working with a model instance
    assert selection is not None
    # from celery.contrib import rdb; rdb.set_trace()
    # assert isinstance(selection, OptionSelection)
    if  not isinstance(selection, OptionSelection):
        try:
            selection = OptionSelection(**selection)  # type: ignore[arg-type]
        except Exception:
            selection = None

    # Merge selection payload if provided
    if selection:
        symbol = selection.symbol or symbol
        snapshot_date = selection.snapshot_date or snapshot_date
        expiry_filter = selection.expiry_filter or expiry_filter
        max_contracts = selection.max_contracts or max_contracts
        rights = selection.rights or rights
        strikes_per_side = selection.strikes_per_side or strikes_per_side
        selection_mode = selection.selection_mode or selection_mode
        expiry_count = selection.expiry_count or expiry_count
        spot_price = selection.spot_price or spot_price
        moneyness_targets = selection.moneyness_targets or moneyness_targets
        delta_targets = selection.delta_targets or delta_targets
        iv_assumption = selection.iv_assumption or iv_assumption
        use_underlying_iv = selection.use_underlying_iv or use_underlying_iv

    symbol = (symbol or "").upper()
    logger.info(f"📊 CONTRACT LOOKUP: Getting option contracts for {symbol}")

    try:
        from common.repositories.option_chains import OptionChainSnapshotRepository

        repo = OptionChainSnapshotRepository(base)

        # Select snapshot date (as_of) with explicit repo filters
        if snapshot_date:
            as_of = datetime.fromisoformat(snapshot_date).date()
            tbl = repo.load(filters=[("underlying", "==", symbol), ("as_of", "==", as_of)])
        else:
            tbl = repo.load(filters=[("underlying", "==", symbol)])
            if tbl.num_rows > 0:
                latest_scalar = pc.max(tbl.column("as_of"))
                latest_date = latest_scalar.as_py()
                mask = pc.equal(
                    tbl.column("as_of"), pa.scalar(latest_date, type=tbl.schema.field("as_of").type)
                )
                idx = pc.indices_nonzero(mask)
                tbl = tbl.take(idx)
                logger.info(f"📊 Using latest snapshot date: {latest_date}")

        if tbl.num_rows == 0:
            logger.warning(f"📊 No option chain data found for {symbol}")
            return SelectedContracts(symbol=symbol, contracts=[], contract_count=0).model_dump()

        # Collapse lists across rows
        exp_lists = tbl.column("expirations").to_pylist()
        strike_lists = tbl.column("strikes").to_pylist()
        exchanges = (
            tbl.column("exchange").to_pylist() if "exchange" in tbl.column_names else []
        )
        trading_classes = (
            tbl.column("trading_class").to_pylist()
            if "trading_class" in tbl.column_names
            else []
        )
        multipliers = (
            tbl.column("multiplier").to_pylist() if "multiplier" in tbl.column_names else []
        )

        all_exps = sorted({e for lst in exp_lists for e in (lst or [])})
        all_strikes = sorted({float(s) for lst in strike_lists for s in (lst or [])})

        if not all_exps or not all_strikes:
            logger.warning(f"📊 No expirations/strikes present for {symbol}")
            return SelectedContracts(symbol=symbol, contracts=[], contract_count=0).model_dump()

        # Choose expiries
        desired_exp_count = max(1, int(expiry_count or 1))
        chosen_exps: List[str] = []
        if expiry_filter:
            ef = pd.to_datetime(expiry_filter).date().isoformat()
            if ef not in all_exps:
                logger.warning(f"📊 Expiry {ef} not available for {symbol}")
                return {"symbol": symbol, "contracts": [], "contract_count": 0}
            chosen_exps.append(ef)
        for e in all_exps:
            if len(chosen_exps) >= desired_exp_count:
                break
            if e not in chosen_exps:
                chosen_exps.append(e)

        # Determine spot
        S = None
        if spot_price and spot_price > 0:
            S = float(spot_price)
        else:
            try:
                from common.repositories.equity_bars import EquityBarRepository
                e_repo = EquityBarRepository(base)
                tbe = e_repo.load(symbol=symbol)
                if tbe and tbe.num_rows > 0 and "close" in tbe.column_names:
                    latest = pc.max(tbe.column("date")).as_py()
                    mask = pc.equal(tbe.column("date"), pa.scalar(latest, type=tbe.schema.field("date").type))
                    tbe2 = tbe.take(pc.indices_nonzero(mask))
                    if tbe2.num_rows > 0:
                        S = float(tbe2.column("close")[0].as_py())
            except Exception:
                S = None
        if S is None:
            # Fallback to middle strike
            try:
                S = float(all_strikes[len(all_strikes)//2])
                logger.warning(f"📊 Spot unavailable; using median strike {S} for {symbol}")
            except Exception:
                S = float(all_strikes[0])

        # Selection modes
        mode = (selection_mode or "count").lower().strip()
        sel_strikes_by_exp: Dict[str, List[float]] = {}
        import math, statistics
        def _norm_cdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        def _d1(s, k, sigma, T):
            if sigma <= 0 or T <= 0:
                return float('inf') if s>k else float('-inf')
            return (math.log(max(s,1e-12)/max(k,1e-12)) + 0.5*sigma*sigma*T) / (sigma*math.sqrt(T))
        def _T(as_of_iso: str, exp_iso: str) -> float:
            try:
                a = pd.to_datetime(as_of_iso).date()
                e = pd.to_datetime(exp_iso).date()
                return max((e - a).days, 1)/365.0
            except Exception:
                return 30/365.0

        # Use snapshot as_of for T reference
        as_of_val = tbl.column("as_of")[0].as_py()
        as_of_iso = as_of_val.isoformat() if hasattr(as_of_val, 'isoformat') else str(as_of_val)

        if mode == "moneyness":
            mts = [float(x) for x in (moneyness_targets or [1.0]) if x is not None]
            for exp in chosen_exps:
                targets = [S*m for m in mts]
                ordered = sorted(all_strikes)
                chosen: List[float] = []
                for t in targets:
                    k = min(ordered, key=lambda x: abs(x - t))
                    if k not in chosen:
                        chosen.append(k)
                sel_strikes_by_exp[exp] = chosen
        elif mode == "delta":
            sig = float(iv_assumption or 0.25)
            deltas = [abs(float(x)) for x in (delta_targets or [0.5]) if x is not None]
            for exp in chosen_exps:
                T = _T(as_of_iso, exp)
                grid = []
                for k in all_strikes:
                    d1 = _d1(S, k, sig, T)
                    cdelta = _norm_cdf(d1)
                    pdelta = cdelta - 1.0
                    grid.append((k, cdelta, pdelta))
                chosen: List[float] = []
                for target in deltas:
                    kc = min(grid, key=lambda tup: abs(tup[1] - target))[0]
                    kp = min(grid, key=lambda tup: abs(tup[2] + target))[0]
                    for v in (kc, kp):
                        if v not in chosen:
                            chosen.append(v)
                sel_strikes_by_exp[exp] = chosen
        else:  # count around spot
            take_side = int(strikes_per_side or max(1, max_contracts//max(1, (2))))
            for exp in chosen_exps:
                ordered = sorted(all_strikes, key=lambda x: abs(x - S))
                sel = []
                for v in ordered:
                    if v not in sel:
                        sel.append(v)
                    if len(sel) >= take_side:
                        break
                sel_strikes_by_exp[exp] = sel

        ex_default = exchanges[0] if exchanges else "SMART"
        tc_default = trading_classes[0] if trading_classes else symbol
        mult_default = multipliers[0] if multipliers else 100

        contracts: List[Dict[str, Any]] = []
        # Rights selection
        rts = [r.upper() for r in (rights or ["C", "P"]) if r and r.upper() in ("C", "P")]
        if not rts:
            rts = ["C", "P"]

        # Build contracts
        for exp in chosen_exps:
            expiry_yyyymmdd = pd.to_datetime(exp).strftime("%Y%m%d")
            for strike in sel_strikes_by_exp.get(exp, []):
                for right in rts:
                    contracts.append(
                        {
                            "symbol": symbol,
                            "secType": "OPT",
                            "exchange": ex_default or "SMART",
                            "currency": "USD",
                            "lastTradeDateOrContractMonth": expiry_yyyymmdd,
                            "strike": float(strike),
                            "right": right,
                            "multiplier": str(mult_default or 100),
                            "tradingClass": tc_default or symbol,
                        }
                    )

        enrich_map = {}
        # No implicit save/enrichment here; follow explicit enrich/save tasks
        enrich_map = {}

        # Apply enrichment to contracts
        if enrich_map:
            for c in contracts:
                key = (float(c.get("strike")), c.get("right"))
                info = enrich_map.get(key)
                if info and info.get("contract_id") is not None:
                    try:
                        c["conId"] = int(info.get("contract_id"))
                    except Exception:
                        pass
                if info and info.get("trading_class") and not c.get("tradingClass"):
                    c["tradingClass"] = info.get("trading_class")

        logger.info(
            f"📊 CONTRACT LOOKUP: Found {len(contracts)} option contracts for {symbol}"
        )

        # Derive snapshot_date from the first row if available
        snap_date_val = tbl.column("as_of")[0].as_py() if tbl.num_rows else None

        return SelectedContracts(
            symbol=symbol,
            contracts=contracts,
            contract_count=len(contracts),
            snapshot_date=snap_date_val.isoformat() if snap_date_val else None,
            correlation_id=(selection.correlation_id if selection else None),
        ).model_dump()

    except Exception as e:
        logger.error(
            f"📊 CONTRACT LOOKUP: Error getting option contracts for {symbol}: {e}"
        )
        return SelectedContracts(
            symbol=symbol,
            contracts=[],
            contract_count=0,
            snapshot_date=None,
            correlation_id=(selection.correlation_id if selection else None),
        ).model_dump()


@shared_task(pydantic=True,
    bind=True,
    name="option_chains.fetch_option_historical_data",
    queue="option_chains",
)
def fetch_option_historical_data_task(
    self,
    contracts_result: SelectedContracts,
    end_date_time: str,
    duration_str: str,
    bar_size_setting: str,
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    timeout_seconds: float = 20.0,
    root: str | None = None,
) -> dict:
    """Fetch historical data for option contracts and save to option_bars repo."""
    import logging

    logger = logging.getLogger(__name__)

    # Coerce dict payloads into model if needed
    if isinstance(contracts_result, dict):
        contracts_result = SelectedContracts(**contracts_result)
    symbol = contracts_result.symbol or "N/A"
    contracts = contracts_result.contracts or []

    if not contracts:
        logger.warning(
            f"📊 No option contracts available for historical data: {symbol}"
        )
        return HistoricalResult(symbol=symbol, processed=0, saved=0, failed=0, results=[]).model_dump()

    logger.info(
        f"📊 OPTION BARS: Fetching historical data for {len(contracts)} option contracts of {symbol}"
    )

    # Reuse existing historical data processing, but with repository_type="option"
    from ops.tasks.historical_data import fetch_historical_data_from_ib_task

    # Format as resolve_result for compatibility
    resolve_result = {"resolved_contracts": contracts, "missing_symbols": []}

    # Call existing historical data task with option repository type
    result = fetch_historical_data_from_ib_task(
        resolve_result=resolve_result,
        end_date_time=end_date_time,
        duration_str=duration_str,
        bar_size_setting=bar_size_setting,
        what_to_show=what_to_show,
        use_rth=use_rth,
        repository_type="option",
        timeout_seconds=timeout_seconds,
        root=root,
    )

    logger.info(
        f"📊 OPTION BARS: Historical data processing complete for {symbol}: {result}"
    )

    return HistoricalResult(
        symbol=symbol,
        processed=int(result.get("processed", 0)),
        saved=int(result.get("saved", 0)),
        failed=int(result.get("failed", 0)),
        results=result.get("results"),
    ).model_dump()


@shared_task(pydantic=True, bind=True, name="option_chains.process_historical_option_data", queue="option_chains")
def process_historical_option_data_task(self, request: OptionHistoricalArgs) -> dict:
    """
    Complete historical option data workflow for a symbol.
    1. Get option contracts from option_chains repository
    2. Fetch historical bars for option contracts
    3. Save to option_bars repository
    """
    import logging

    # Coerce dict payloads into Pydantic models for robustness
    if isinstance(request, dict):
        request = OptionHistoricalArgs(**request)

    set_correlation_id(request.correlation_id)
    logger = CorrelationIdAdapter(logging.getLogger(__name__), {"cid": request.correlation_id})
    symbol = request.symbol
    logger.info(f"🔄 STARTING: Historical option data processing for {symbol}")

    # Step 1: Get contracts from option chain repo
    logger.info(f"📍 STEP 1: Getting option contracts for {symbol}...")
    selection = OptionSelection(**request.model_dump())
    # Do not pass outer task self when calling another Celery task directly
    contracts_result = get_option_contracts_for_bars_task(selection)

    # contracts_result may be a dict (serialized from previous task)
    if isinstance(contracts_result, dict):
        contract_count = int(contracts_result.get("contract_count", 0))
    else:
        try:
            contract_count = int(getattr(contracts_result, "contract_count", 0))
        except Exception:
            contract_count = 0
    logger.info(f"✅ Got {contract_count} option contracts for {symbol}")

    if contract_count == 0:
        return HistoricalResult(
            symbol=symbol,
            processed=0,
            saved=0,
            failed=0,
            results=[],
            correlation_id=request.correlation_id,
        ).model_dump()

    # Step 2: Fetch and save historical bars
    logger.info(
        f"📍 STEP 2: Fetching historical data for {contract_count} option contracts..."
    )
    result = fetch_option_historical_data_task(
        contracts_result=contracts_result,
        end_date_time=request.end_date_time,
        duration_str=request.duration_str,
        bar_size_setting=request.bar_size_setting,
        what_to_show=request.what_to_show,
        use_rth=request.use_rth,
        timeout_seconds=request.timeout_seconds,
        root=request.root,
    )

    processed_count = int(result.get("processed", 0))
    saved_count = int(result.get("saved", 0))
    failed_count = int(result.get("failed", 0))

    logger.info(
        f"✅ Historical option data processing complete for {symbol}: {processed_count} processed, {saved_count} bars saved, {failed_count} failed"
    )

    return HistoricalResult(
        symbol=symbol,
        processed=processed_count,
        saved=saved_count,
        failed=failed_count,
        results=result.get("results"),
        correlation_id=request.correlation_id,
    ).model_dump()


@shared_task(pydantic=True,
    bind=True,
    name="option_chains.save_selected_option_contracts",
    queue="option_chains",
)
def save_selected_option_contracts_to_repo_task(
    self,
    contracts_result: SelectedContracts,
    root: str | None = None,
) -> dict:
    """
    Save selected option contracts to the contracts repository if conId is present.

    Does not fetch from IB. Intended to persist details that were already known
    (e.g., from prior enrichment) so other flows can read from the contracts repo.
    """
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)

    # Coerce dict payloads into model if needed
    if isinstance(contracts_result, dict):
        contracts_result = SelectedContracts(**contracts_result)
    contracts = contracts_result.contracts or []
    if not contracts:
        return SaveContractsResult(saved=0, skipped=0, missing_conid=0).model_dump()

    from app_settings.settings import get_settings
    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    rows = []
    missing = 0
    for c in contracts:
        conid = c.get("conId")
        if not conid:
            missing += 1
            continue
        # Derive expiry from lastTradeDateOrContractMonth or localSymbol
        expiry_raw = c.get("lastTradeDateOrContractMonth") or c.get("expiry")
        expiry_dt = None
        if expiry_raw:
            s = str(expiry_raw).strip()
            try:
                if len(s) == 8 and s.isdigit():
                    expiry_dt = pd.to_datetime(s, format="%Y%m%d").date()
                else:
                    token = s.split()[0]
                    if len(token) == 8 and token.isdigit():
                        expiry_dt = pd.to_datetime(token, format="%Y%m%d").date()
                    else:
                        expiry_dt = pd.to_datetime(token).date()
            except Exception:
                expiry_dt = None
        if expiry_dt is None:
            import re
            ls = c.get("localSymbol") or c.get("local_symbol")
            if ls:
                m = re.search(r"(\d{6})([CP])(\d{8})", str(ls))
                if m:
                    yymmdd = m.group(1)
                    try:
                        expiry_dt = pd.to_datetime("20" + yymmdd, format="%Y%m%d").date()
                    except Exception:
                        pass
        # Derive right/strike from payload or localSymbol
        right_val = c.get("right")
        strike_val = c.get("strike")
        if (right_val is None or str(right_val).strip().lower() in ("", "none", "nan")) or (
            strike_val is None or str(strike_val).strip().lower() in ("", "none", "nan")
        ):
            ls = c.get("localSymbol") or c.get("local_symbol")
            if ls:
                import re
                m = re.search(r"(\d{6})([CP])(\d{8})", str(ls))
                if m:
                    if not right_val:
                        right_val = m.group(2)
                    if not strike_val:
                        try:
                            strike_val = int(m.group(3)) / 1000.0
                        except Exception:
                            pass
        rows.append(
            {
                "conid": int(conid),
                "symbol": c.get("symbol"),
                "sec_type": "OPT",
                "currency": c.get("currency", "USD"),
                "exchange": c.get("exchange"),
                "primary_exchange": c.get("primaryExchange") or c.get("primary_exchange"),
                "local_symbol": c.get("localSymbol") or c.get("local_symbol"),
                "trading_class": c.get("tradingClass") or c.get("trading_class"),
                "multiplier": int(c.get("multiplier") or 100),
                "strike": float(strike_val) if strike_val is not None else None,
                "right": (str(right_val).upper() if right_val is not None else None),
                "expiry": expiry_dt,
                "expiry_raw": str(expiry_raw) if expiry_raw else None,
                # Optional IB metadata
                "min_tick": (float(c.get("minTick")) if c.get("minTick") is not None else None),
                "valid_exchanges": c.get("validExchanges") or c.get("valid_exchanges"),
                "long_name": c.get("longName") or c.get("long_name"),
                "category": c.get("category"),
                "subcategory": c.get("subcategory") or c.get("subCategory"),
                "industry": c.get("industry"),
                "timezone_id": c.get("timeZoneId") or c.get("timezone_id"),
                "trading_hours": c.get("tradingHours") or c.get("trading_hours"),
                "liquid_hours": c.get("liquidHours") or c.get("liquid_hours"),
                "market_rule_ids": c.get("marketRuleIds") or c.get("market_rule_ids"),
                "contract_month": c.get("contractMonth") or c.get("contract_month"),
                # Snapshot partition column (date) and created_at timestamp (UTC, ns)
                "snapshot": pd.Timestamp.utcnow().date(),
                "created_at": pd.Timestamp.utcnow(),
            }
        )

    if not rows:
        return SaveContractsResult(saved=0, skipped=0, missing_conid=missing).model_dump()

    df = pd.DataFrame(rows)
    # Ensure all repository schema columns exist for option contracts repo
    try:
        from common.repositories.option_contracts import OptionContractsRepository
        schema = OptionContractsRepository(base)._create_schema()
        for f in schema:
            if f.name not in df.columns:
                df[f.name] = None
    except Exception:
        pass
    from common.repositories.option_contracts import OptionContractsRepository
    repo = OptionContractsRepository(base)
    try:
        repo.save(df)
        logger.info(f"Saved {len(df)} option contracts into option_contracts repo")
        return SaveContractsResult(saved=len(df), skipped=0, missing_conid=missing).model_dump()
    except Exception as e:
        logger.error(f"Error saving option contracts: {e}")
        return SaveContractsResult(saved=0, skipped=0, missing_conid=missing, error=str(e)).model_dump()


@shared_task(pydantic=True,
    bind=True,
    name="option_chains.fetch_option_contract_details",
    queue="option_chains",
)
def fetch_option_contract_details_task(
    self,
    contracts_result: SelectedContracts,
    api_base: Optional[str] = None,
    timeout_seconds: float = 15.0,
    root: str | None = None,
) -> dict:
    """
    Fetch option contract details (to get conId/localSymbol) for the selected
    contracts. This uses the IB API service and is NOT called automatically to
    honor the no-fallback policy; orchestrator can invoke it explicitly.
    """
    import logging, httpx
    logger = logging.getLogger(__name__)

    # Coerce dict payloads into model if needed
    if isinstance(contracts_result, dict):
        contracts_result = SelectedContracts(**contracts_result)
    contracts = contracts_result.contracts or []
    if not contracts:
        return SelectedContracts(symbol=contracts_result.symbol, contracts=[], contract_count=0, snapshot_date=contracts_result.snapshot_date, correlation_id=None).model_dump()

    from app_settings.settings import get_settings
    settings = get_settings()
    base_url = api_base or str(settings.api.services.ib_api.url)

    updated = []
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for c in contracts:
            try:
                # Skip if already has conId
                if c.get("conId"):
                    updated.append(c)
                    continue
                resp = client.post(
                    "/ib/contract-details",
                    json={"contract": c, "timeout": float(timeout_seconds)},
                )
                resp.raise_for_status()
                details = resp.json()
                if details:
                    d = details[0]
                    # Extract common fields
                    conid = d.get("contract", {}).get("conId") or d.get("conId")
                    local_symbol = d.get("contract", {}).get("localSymbol") or d.get("localSymbol")
                    trading_class = d.get("contract", {}).get("tradingClass") or d.get("tradingClass")
                    # Optional metadata from details
                    primary_exchange = d.get("contract", {}).get("primaryExchange") or d.get("primaryExchange")
                    min_tick = d.get("minTick")
                    valid_exchanges = d.get("validExchanges")
                    long_name = d.get("longName")
                    category = d.get("category")
                    subcategory = d.get("subcategory") or d.get("subCategory")
                    industry = d.get("industry")
                    time_zone = d.get("timeZoneId")
                    trading_hours = d.get("tradingHours")
                    liquid_hours = d.get("liquidHours")
                    market_rule_ids = d.get("marketRuleIds")
                    contract_month = d.get("contractMonth")
                    last_trade_or_month = d.get("contract", {}).get("lastTradeDateOrContractMonth") or d.get("lastTradeDateOrContractMonth")
                    if conid:
                        c["conId"] = int(conid)
                    if local_symbol:
                        c["localSymbol"] = local_symbol
                    if trading_class and not c.get("tradingClass"):
                        c["tradingClass"] = trading_class
                    # Enrich optional fields if present
                    if primary_exchange:
                        c["primaryExchange"] = primary_exchange
                    if min_tick is not None:
                        c["minTick"] = min_tick
                    if valid_exchanges:
                        c["validExchanges"] = valid_exchanges
                    if long_name:
                        c["longName"] = long_name
                    if category:
                        c["category"] = category
                    if subcategory:
                        c["subcategory"] = subcategory
                    if industry:
                        c["industry"] = industry
                    if time_zone:
                        c["timeZoneId"] = time_zone
                    if trading_hours:
                        c["tradingHours"] = trading_hours
                    if liquid_hours:
                        c["liquidHours"] = liquid_hours
                    if market_rule_ids:
                        c["marketRuleIds"] = market_rule_ids
                    if contract_month:
                        c["contractMonth"] = contract_month
                    if last_trade_or_month:
                        c["lastTradeDateOrContractMonth"] = last_trade_or_month
                updated.append(c)
            except Exception as e:
                logger.error(f"error fetching option contract detail: {e}")
                updated.append(c)

    return SelectedContracts(
        symbol=contracts_result.symbol,
        contracts=updated,
        contract_count=len(updated),
        snapshot_date=contracts_result.snapshot_date,
        correlation_id=None,
    ).model_dump()
