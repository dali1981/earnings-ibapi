import pytest


@pytest.mark.unit
def test_processed_index_marks_and_checks(tmp_path):
    from common.utils.events.processed_index import ProcessedIndex

    idx = ProcessedIndex(
        service="svc", topic="equity_bars", state_root=tmp_path / "state"
    )
    key = "AAPL:1 day:2024-01-01:2024-01-31"

    assert not idx.is_processed(key)
    idx.mark_processed(key)
    assert idx.is_processed(key)
