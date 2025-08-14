import pandas as pd
import pytest
from datetime import date, timedelta


@pytest.mark.unit
def test_backfill_equity_bars_writes_and_notifies(tmp_path, monkeypatch):
    from repositories.equity_bars import EquityBarRepository
    from ibx_flows.backfill import BackfillConfig, backfill_equity_bars

    # Fake source implementing get_equity_bars
    class FakeSource:
        def get_equity_bars(self, symbol, start, end, bar_size):
            # Produce 3 daily rows with required columns
            rows = []
            cur = start
            while cur <= end:
                rows.append(
                    {
                        "date": cur,
                        "symbol": symbol,
                        "bar_size": bar_size,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.5,
                        "close": 100.5,
                        "volume": 1000,
                    }
                )
                cur += timedelta(days=1)
            return pd.DataFrame(rows)

    # Capture bus used inside backfill
    class CapturingBus:
        events = []

        def __init__(self, *args, **kwargs):
            pass

        def publish(self, event_type: str, topic: str, payload=None):
            CapturingBus.events.append((event_type, topic, payload or {}))

        def notify_data_available(self, dataset: str, **meta):
            self.publish("data.available", dataset, meta)

    # Patch the FileEventBus in module under test
    import ibx_flows.backfill as backfill_mod

    monkeypatch.setattr(backfill_mod, "FileEventBus", CapturingBus, raising=True)
    CapturingBus.events.clear()

    # Arrange
    base = tmp_path
    repo = EquityBarRepository(base)
    cfg = BackfillConfig(
        underlying="TEST",
        start=date(2024, 1, 1),
        end=date(2024, 1, 3),
        bar_size="1 day",
    )

    # Act
    backfill_equity_bars(FakeSource(), repo, cfg)

    # Assert repository write
    out_dir = base / "equity_bars" / "symbol=TEST"
    files = list(out_dir.rglob("*.parquet"))
    assert files, f"Expected parquet files under {out_dir}"

    # Assert notification
    assert CapturingBus.events, "Expected a data.available event"
    evt_type, topic, payload = CapturingBus.events[-1]
    assert evt_type == "data.available"
    assert topic == "equity_bars"
    assert payload.get("symbol") == "TEST"
    assert payload.get("rows") and payload["rows"] >= 3
