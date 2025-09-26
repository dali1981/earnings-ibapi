"""Repository factory for creating repository instances."""

from app_settings.settings import get_settings
from pathlib import Path
from typing import Union


def create_repository(repository_type: str, base_path: Union[str, Path, None] = None):
    """Create a repository instance based on type.

    Args:
        repository_type: Type of repository ("equity" or "option")
        base_path: Base path for data storage (defaults to settings.paths.data_root)

    Returns:
        Repository instance

    Raises:
        ValueError: If repository_type is not supported
    """
    if base_path is None:
        settings = get_settings()
        base_path = settings.paths.data_root
    else:
        base_path = Path(base_path)

    if repository_type == "equity":
        try:
            from common.repositories.equity_bars import EquityBarRepository
        except ImportError:
            from repositories.equity_bars import EquityBarRepository
        return EquityBarRepository(base_path)
    elif repository_type == "option":
        try:
            from common.repositories.option_bars import OptionBarRepository
        except ImportError:
            from repositories.option_bars import OptionBarRepository
        return OptionBarRepository(base_path)
    else:
        raise ValueError(f"Unknown repository type: {repository_type}")
