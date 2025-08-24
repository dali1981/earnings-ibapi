import json
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


def test_file_notifier_appends_jsonl(tmp_path, monkeypatch):
    # Redirect CACHE_PATH to tmp
    import config as config_mod

    cache = tmp_path / "cache"
    monkeypatch.setattr(config_mod, "CACHE_PATH", cache, raising=False)

    from common.utils.notifications import FileNotifier

    n = FileNotifier()
    n.notify("Backfill completed", severity="success", category="jobs", context={"job": "equity_bars", "rows": 123})

    out = cache / "notifications.jsonl"
    assert out.exists(), "notifications jsonl should be created"
    line = out.read_text(encoding="utf-8").strip().splitlines()[0]
    obj = json.loads(line)
    assert obj["message"] == "Backfill completed"
    assert obj["severity"] == "success"
    assert obj["category"] == "jobs"
    assert obj["context"]["rows"] == 123


def test_multi_notifier_fanout(monkeypatch, caplog, tmp_path):
    # Ensure file writes go to tmp
    import config as config_mod

    cache = tmp_path / "cache"
    monkeypatch.setattr(config_mod, "CACHE_PATH", cache, raising=False)

    from common.utils.notifications import FileNotifier, LoggerNotifier, MultiNotifier

    file_n = FileNotifier()
    log_n = LoggerNotifier()
    m = MultiNotifier([file_n, log_n])

    with caplog.at_level("INFO"):
        m.notify("Data available", category="events", context={"topic": "equity_bars", "symbol": "AAPL"})

    # File notification present
    out = cache / "notifications.jsonl"
    assert out.exists()
    obj = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[0])
    assert obj["category"] == "events"
    assert obj["context"]["topic"] == "equity_bars"

    # Logger emitted an info-level record containing the message
    assert any("Data available" in rec.message for rec in caplog.records)

