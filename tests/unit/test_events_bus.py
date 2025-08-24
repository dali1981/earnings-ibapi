import json
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.events]


def test_file_event_bus_publish_and_consume(tmp_path: Path, monkeypatch):
    # Point CACHE_PATH to a tmp dir
    import config as config_mod

    cache = tmp_path / "cache"
    monkeypatch.setattr(config_mod, "CACHE_PATH", cache, raising=False)

    from common.utils.events.bus import FileEventBus
    from common.utils.events.consumer import JSONLConsumer

    bus = FileEventBus()
    bus.publish("data.available", topic="equity_bars", payload={"symbol": "AAPL", "job_id": "job-1"})

    # Assert log file and ready flag created
    events_file = cache / "events" / "equity_bars.jsonl"
    ready_flag = cache / "ready" / "equity_bars.flag"
    assert events_file.exists(), "events JSONL not created"
    assert ready_flag.exists(), "ready flag not touched"

    # Read event via consumer and assert correlation_id propagation
    cons = JSONLConsumer("equity_bars", cache_root=cache)
    items = cons.read(service="test")
    assert items and items[0]["type"] == "data.available"
    payload = items[0]["payload"]
    # correlation_id should be set from job_id per bus enrichment
    assert payload.get("correlation_id") == "job-1"

