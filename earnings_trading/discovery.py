#!/usr/bin/env python3
"""
Compatibility shim for the historic earnings_trading.discovery module.

This preserves the old import surface by delegating to the new, separated
modules: `earnings.discovery` for pure discovery and `trading.strategy_analyzer`
for trading-related dataclasses/enums and scoring helpers.
"""

from __future__ import annotations

from config import DISCOVERY_CONFIG, EXPORTS_PATH
from dataclasses import dataclass
from datetime import date
from earnings.discovery import EarningsDiscoveryEngine as PureDiscoveryEngine
from earnings.fetcher import EarningsEvent
from pathlib import Path
from trading.strategy_analyzer import (
    OptionsStrategy,
    OptionsStrategyAnalyzer,
    TradingOpportunity,
)
from typing import Dict, List, Optional


# Backwards-compatible aliases
EarningsCandidate = TradingOpportunity


class EarningsDiscoveryEngine:
    """Back-compat discovery + light scoring facade.

    New code should use `earnings.discovery.EarningsDiscoveryEngine` (pure discovery)
    plus `trading.strategy_analyzer.OptionsStrategyAnalyzer` (scoring).
    """

    def __init__(self, base_path: Path | None = None, config: dict | None = None):
        self.base_path = base_path or Path("data")
        self.config = config or DISCOVERY_CONFIG
        self._pure = PureDiscoveryEngine(self.config)
        self._analyzer = OptionsStrategyAnalyzer(self.config)

    # ---- Compatibility helpers used by examples/tests ----
    def _create_candidate_from_earnings(
        self, event: EarningsEvent
    ) -> EarningsCandidate:
        return EarningsCandidate(
            symbol=event.symbol,
            company_name=event.company_name,
            earnings_date=event.earnings_date,
            time=event.time or "unknown",
            days_until_earnings=event.days_until_earnings,
            market_cap=getattr(event, "market_cap", None),
            quality_score=0.0,
        )

    def _passes_basic_filters(self, candidate: EarningsCandidate) -> bool:
        # Timing only; let analyzer apply strategy-specific filters
        d = candidate.days_until_earnings
        if d > 0:
            if d < self.config["discovery_min_days"]:
                return False
            if d > self.config["discovery_max_days"]:
                return False
        return True

    def _score_options_strategies(self, candidate: EarningsCandidate) -> None:
        # Basic timing-weighted scoring using analyzer preferences
        start = self.config["strategy_optimal_start"]
        end = self.config["strategy_optimal_end"]
        tw = self.config["timing_weights"]

        if candidate.days_until_earnings >= end:
            base = tw["optimal"]
        elif candidate.days_until_earnings >= start:
            base = tw["good"]
        elif candidate.days_until_earnings >= 1:
            base = tw["fair"]
        else:
            base = tw["poor"]

        mult = self.config["strategy_multipliers"]
        candidate.calendar_score = (
            base * mult["calendar_spread"]["7_to_14_days"] / 100.0
        )
        candidate.straddle_score = base * mult["straddle"]["5_to_10_days"] / 100.0
        candidate.strangle_score = base * mult["strangle"]["7_to_14_days"] / 100.0

    # ---- Public API maintained for examples ----
    def discover_earnings_opportunities(
        self, days_ahead: int = 21, min_score: float = 50.0
    ) -> List[EarningsCandidate]:
        events = self._pure.earnings_fetcher.get_earnings_range(
            None, days_back=0, days_ahead=days_ahead
        )
        candidates: List[EarningsCandidate] = []
        for ev in events:
            c = self._create_candidate_from_earnings(ev)
            if self._passes_basic_filters(c):
                self._score_options_strategies(c)
                if c.total_score >= min_score:
                    candidates.append(c)
        candidates.sort(key=lambda x: -x.total_score)
        return candidates

    def export_opportunities(self, opportunities: List[EarningsCandidate]) -> Path:
        out = EXPORTS_PATH / f"opportunities_{date.today().isoformat()}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "symbol",
                    "earnings_date",
                    "days_until",
                    "total_score",
                    "best_strategy",
                ]
            )
            for o in opportunities:
                best = o.best_strategy.value if o.best_strategy else ""
                w.writerow(
                    [
                        o.symbol,
                        o.earnings_date.isoformat(),
                        o.days_until_earnings,
                        f"{o.total_score:.1f}",
                        best,
                    ]
                )
        return out

    def get_data_collection_list(
        self, opportunities: List[EarningsCandidate]
    ) -> Dict[str, List[str]]:
        buckets = {"critical": [], "high": [], "medium": [], "low": []}
        for o in opportunities:
            d = o.days_until_earnings
            if d <= 3:
                buckets["critical"].append(o.symbol)
            elif d <= 7:
                buckets["high"].append(o.symbol)
            elif d <= 14:
                buckets["medium"].append(o.symbol)
            else:
                buckets["low"].append(o.symbol)
        return buckets
