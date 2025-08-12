from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from ibx_flows.backfill import BackfillConfig, backfill_equity_bars, backfill_option_bars
from ibx_flows.source_ib import IBSource
# Use new unified repository system
from repositories import EquityBarRepository, OptionBarRepository, OptionChainSnapshotRepository


class Task:
    """Base class for runnable jobs with validation."""

    def validate(self) -> None:  # pragma: no cover - override in subclasses
        """Validate prerequisites for the task."""
        return None

    def execute(self) -> None:  # pragma: no cover - override in subclasses
        raise NotImplementedError

    def run(self) -> None:
        """Validate then execute the task."""
        self.validate()
        self.execute()


@dataclass
class BackfillEquityBarsTask(Task):
    symbol: str
    start: date
    end: date
    bar_size: str = "1 day"
    out: Path = Path("data/equity_bars")

    def validate(self) -> None:
        if self.start > self.end:
            raise ValueError("start date must be on or before end date")

    def execute(self) -> None:
        cfg = BackfillConfig(
            underlying=self.symbol,
            start=self.start,
            end=self.end,
            bar_size=self.bar_size,
        )
        repo = EquityBarRepository(self.out)
        src = IBSource()
        backfill_equity_bars(src, repo, cfg)


@dataclass
class BackfillOptionBarsTask(Task):
    symbol: str
    start: date
    end: date
    bar_size: str = "1 day"
    out: Path = Path("data/option_bars")
    chain_base: Path = Path("data")
    expiries_max: int = 8
    strikes_per_side: int = 6
    selection_mode: str = "k_around_atm"

    _chain_repo: Optional[OptionChainSnapshotRepository] = None

    def validate(self) -> None:
        chain_repo = OptionChainSnapshotRepository(self.chain_base / "option_chains")
        try:
            df = chain_repo.load()
        except Exception as exc:
            raise FileNotFoundError(
                f"No option chain data repository found at {chain_repo.base_path}"
            ) from exc
        if df.empty:
            raise ValueError(
                f"Option chain repository at {chain_repo.base_path} contains no data"
            )
        self._chain_repo = chain_repo

    def execute(self) -> None:
        if self._chain_repo is None:  # pragma: no cover - safety
            raise RuntimeError("validate must be run before execute")
        cfg = BackfillConfig(
            underlying=self.symbol,
            start=self.start,
            end=self.end,
            bar_size=self.bar_size,
            expiries_max=self.expiries_max,
            strikes_per_side=self.strikes_per_side,
            selection_mode=self.selection_mode,
        )
        repo = OptionBarRepository(self.out)
        src = IBSource()
        backfill_option_bars(src, repo, chain_repo=self._chain_repo, eq_repo=None, cfg=cfg)
