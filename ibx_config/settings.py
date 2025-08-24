from __future__ import annotations

from functools import lru_cache
from enum import StrEnum
from pathlib import Path
from pydantic import BaseModel, AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    dev = "dev"
    staging = "staging"
    prod = "prod"


class RepoSettings(BaseModel):
    contracts_path: Path = Path("./data/contracts")


class IBSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id_contracts: int = 111


class APISettings(BaseModel):
    # Logical service registry → concrete URLs; environment switches which one is “active”
    contracts_service_base: AnyHttpUrl = Field("http://localhost:9010")


class CeleryContractsLimits(BaseModel):
    fetch_rate: str = "10/m"      # rate limiting for API-hitting task
    queue: str = "contracts"      # dedicated queue if you want


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IBX_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    env: AppEnv = AppEnv.dev
    repo: RepoSettings = RepoSettings()
    ib: IBSettings = IBSettings()
    api: APISettings = APISettings()
    limits: CeleryContractsLimits = CeleryContractsLimits()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Sources order: env vars > .env file(s)
    # pydantic-settings already reads .env in CWD; customize if you keep env-specific files.
    return Settings()

