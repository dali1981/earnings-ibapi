import json
from pathlib import Path

import pytest


@pytest.mark.unit
def test_jsonl_consumer_checkpoint_and_read(tmp_path):
    from common.utils.events.consumer import JSONLConsumer

    events_dir = tmp_path / "cache" / "events"
    state_root = tmp_path / "cache" / "state"
    events_dir.mkdir(parents=True, exist_ok=True)
    topic_file = events_dir / "equity_bars.jsonl"

    # Write two events
    def write_evt(payload, ts):
        rec = {"type": "data.available", "topic": "equity_bars", "payload": payload, "ts": ts}
        with topic_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    write_evt({"symbol": "AAA"}, 1.0)
    write_evt({"symbol": "BBB"}, 2.0)

    cons = JSONLConsumer("equity_bars", cache_root=tmp_path / "cache", state_root=state_root)

    # First read returns both
    out1 = cons.read(service="svc")
    assert [e["payload"]["symbol"] for e in out1] == ["AAA", "BBB"]

    # Second read returns none
    out2 = cons.read(service="svc")
    assert out2 == []

    # Append one more and ensure we only get new one
    write_evt({"symbol": "CCC"}, 3.0)
    out3 = cons.read(service="svc")
    assert [e["payload"]["symbol"] for e in out3] == ["CCC"]
