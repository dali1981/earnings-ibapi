from __future__ import annotations

from functools import lru_cache
from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    dev = "dev"
    test = "test"
    prod = "prod"


class RepoSettings(BaseModel):
    contracts_path: Path = Path("./data/contracts")


class IBSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id_contracts: int = 111


class APISettings(BaseModel):
    contracts_service_base: AnyHttpUrl = Field("http://localhost:9010")


class CeleryContractsLimits(BaseModel):
    fetch_rate: str = "10/m"
    queue: str = "contracts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IBX_",
        env_nested_delimiter="__",
        extra="ignore",
        # Default .env; per-env handled in get_settings() for flexibility
        env_file=".env",
        env_file_encoding="utf-8",
    )

    env: AppEnv = AppEnv.dev
    repo: RepoSettings = RepoSettings()
    ib: IBSettings = IBSettings()
    api: APISettings = APISettings()
    limits: CeleryContractsLimits = CeleryContractsLimits()


def _load_env_files(env: AppEnv) -> None:
    """Optionally load layered env files: .env then .env.<env>.

    Uses python-dotenv if available; otherwise, relies on pydantic-settings default .env loading.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # Base env
    load_dotenv(".env", override=False)
    # Per-env overlay
    env_file = f".env.{env}"
    if Path(env_file).exists():
        load_dotenv(env_file, override=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # First pass to determine env (from existing env or defaults)
    provisional = Settings()
    # Load env files according to selected environment (dev/test/prod)
    _load_env_files(provisional.env)
    # Final settings parse after env files applied
    s = Settings()
    # Apply environment-specific defaults if desired
    if s.env == AppEnv.test:
        # Example: reduce rates, use in-memory broker, etc. (keep as-is unless specified)
        pass
    elif s.env == AppEnv.prod:
        # Example: tighten limits, different service base, etc. (keep as-is unless specified)
        pass
    return s

