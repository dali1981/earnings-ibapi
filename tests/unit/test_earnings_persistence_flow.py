import pytest
from datetime import date, timedelta


@pytest.mark.unit
def test_earnings_persistence_writes_and_notifies(tmp_path, monkeypatch):
    from earnings.fetcher import EarningsEvent
    from earnings.services.storage import EarningsPersistence
    import repositories.earnings as earnings_mod

    # Prevent retention cleanup from deleting just-written files
    monkeypatch.setattr(earnings_mod.EarningsRepository, "_cleanup_old_data", lambda self: None, raising=True)

    class CapturingBus:
        def __init__(self):
            self.events = []

        def notify_data_available(self, dataset: str, **meta):
            self.events.append((dataset, meta))

    # Arrange: two simple earnings events
    today = date(2024, 1, 1)
    evts = [
        EarningsEvent(
            symbol="AAA",
            company_name="AAA Inc",
            earnings_date=today + timedelta(days=10),
            source="nasdaq",
        ),
        EarningsEvent(
            symbol="BBB",
            company_name="BBB Corp",
            earnings_date=today + timedelta(days=5),
            source="nasdaq",
        ),
    ]

    bus = CapturingBus()
    p = EarningsPersistence(use_config_paths=False, base_data_root=tmp_path)

    # Act
    result = p.persist_earnings_batch(evts, collection_date=today, metadata={"test": True}, bus=bus, dataset_topic="earnings.test")

    # Assert: status and files written
    assert result["status"] == "success"
    out_dir = tmp_path / "earnings" / f"collection_date={today.isoformat()}" / "source=nasdaq"
    assert list(out_dir.rglob("*.parquet"))
    # Assert: notification captured
    assert bus.events, "Expected an earnings data.available event"
    topic, meta = bus.events[-1]
    assert topic == "earnings.test"
    assert meta.get("total_events") == 2
