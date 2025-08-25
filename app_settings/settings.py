from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    dev = "dev"
    test = "test"
    prod = "prod"


class PathsSettings(BaseModel):
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = Field(default_factory=lambda: Path("./data"))
    equity_bars: Path = Field(default_factory=lambda: Path("./data/equity_bars"))
    option_bars: Path = Field(default_factory=lambda: Path("./data/option_bars"))
    option_chains: Path = Field(default_factory=lambda: Path("./data/option_chains"))
    contracts: Path = Field(default_factory=lambda: Path("./data/contracts"))
    earnings: Path = Field(default_factory=lambda: Path("./data/earnings"))
    exports: Path = Field(default_factory=lambda: Path("./data/exports"))
    cache: Path = Field(default_factory=lambda: Path("./data/cache"))
    logs: Path = Field(default_factory=lambda: Path("./data/logs"))


class IBSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4002
    client_ids: Dict[str, int] = {
        "default": 77,
        "streamer": 17,
        "contract_details_backfill": 8,
        "ib_source": 101,
        "contracts_backfill": 1,
        "fundamental_data": 200,
    }

    @property
    def client_id_default(self) -> int:
        return self.client_ids.get("default", 77)

    @property
    def client_id_contracts(self) -> int:
        return self.client_ids.get("contracts_backfill", self.client_id_default)


class APISettings(BaseModel):
    # Logical service registry
    contracts_service_base: AnyHttpUrl = Field("http://localhost:9010")


class CeleryContractsLimits(BaseModel):
    fetch_rate: str = "10/m"
    queue: str = "contracts"


class CelerySettings(BaseModel):
    broker_url: Optional[str] = None
    result_backend: Optional[str] = None
    contracts: CeleryContractsLimits = CeleryContractsLimits()


class EarningsSettings(BaseModel):
    default_days_ahead: int = 3
    default_days_back: int = 360
    max_days_ahead: int = 60
    collection_hour: int = 6
    data_sources: list[str] = ["nasdaq"]
    primary_source: str = "nasdaq"
    retention_days: int = 365
    cache_hours: int = 6
    api_delays: Dict[str, float] = {"nasdaq": 0.3, "fmp": 0.25, "finnhub": 1.0}


class DiscoverySettings(BaseModel):
    min_market_cap: int = 1_000_000_000
    max_market_cap: int = 2_000_000_000_000
    min_avg_volume: int = 1_000_000
    min_price: float = 10.0
    max_price: float = 1000.0
    discovery_min_days: int = 1
    discovery_max_days: int = 60
    data_collection: Dict[str, int] = {
        "pre_earnings_days": 3,
        "post_earnings_days": 2,
        "lookback_days": 30,
        "iv_history_days": 90,
    }
    strategy_optimal_start: int = 7
    strategy_optimal_end: int = 14
    require_weekly_options: bool = False
    require_liquid_options: bool = False
    min_iv_rank: int = 20
    timing_weights: Dict[str, float] = {
        "optimal": 100.0,
        "good": 80.0,
        "fair": 60.0,
        "poor": 40.0,
    }
    strategy_multipliers: Dict[str, Dict[str, float]] = {
        "calendar_spread": {
            "14_plus_days": 1.0,
            "7_to_14_days": 0.8,
            "under_7_days": 0.5,
        },
        "straddle": {"under_5_days": 1.0, "5_to_10_days": 0.9, "over_10_days": 0.7},
        "strangle": {"under_7_days": 1.0, "7_to_14_days": 0.8, "over_14_days": 0.6},
    }
    min_opportunity_score: float = 50.0
    excellent_threshold: float = 90.0
    good_threshold: float = 75.0


class LoggingSettings(BaseModel):
    def build_config(self, logs_path: Path) -> dict:
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "default"},
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "filename": str(logs_path / "app.log"),
                    "maxBytes": 5 * 1024 * 1024,
                    "backupCount": 5,
                },
            },
            "root": {"level": "INFO", "handlers": ["console", "file"]},
        }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    env: AppEnv = AppEnv.dev
    paths: PathsSettings = PathsSettings()
    ib: IBSettings = IBSettings()
    api: APISettings = APISettings()
    celery: CelerySettings = CelerySettings()
    earnings: EarningsSettings = EarningsSettings()
    discovery: DiscoverySettings = DiscoverySettings()
    logging: LoggingSettings = LoggingSettings()


def _load_env_files(env: AppEnv) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(".env", override=False)
    env_file = f".env.{env}"
    if Path(env_file).exists():
        load_dotenv(env_file, override=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provisional = Settings()
    _load_env_files(provisional.env)
    s = Settings()
    # Ensure directories exist
    s.paths.data_root.mkdir(parents=True, exist_ok=True)
    for p in [
        s.paths.equity_bars,
        s.paths.option_bars,
        s.paths.option_chains,
        s.paths.contracts,
        s.paths.earnings,
        s.paths.exports,
        s.paths.cache,
        s.paths.logs,
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)
    return s

