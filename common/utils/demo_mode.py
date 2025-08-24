from __future__ import annotations

from pathlib import Path

import config


def demo_base() -> Path:
    """Base path for demo data to avoid mixing with production datasets."""
    return config.EXPORTS_PATH / "demo"


def dataset_topic(dataset: str, demo: bool) -> str:
    """Return dataset topic, namespaced when in demo mode."""
    return f"{dataset}.demo" if demo else dataset
