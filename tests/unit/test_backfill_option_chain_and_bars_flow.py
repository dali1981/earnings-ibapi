import pandas as pd
import pytest
from datetime import date, datetime, timedelta


@pytest.mark.unit
def test_backfill_option_chain_snapshot_writes_and_notifies(tmp_path, monkeypatch):
    from repositories.option_chains import OptionChainSnapshotRepository
    from ibx_flows.backfill import BackfillConfig, backfill_option_chain_daily

    class FakeSource:
        def get_option_chain(self, underlying, asof_date):
            rows = []
            for k in range(3):
                rows.append(
                    {
                        "underlying": underlying,
                        "snapshot_date": asof_date,
                        "expiry": asof_date + timedelta(days=30),
                        "strike": 100 + k,
                        "right": "C" if k % 2 == 0 else "P",
                        "bid": 1.0,
                        "ask": 1.2,
                        "open_interest": 10,
                    }
                )
            return pd.DataFrame(rows)

    class CapturingBus:
        events = []

        def __init__(self, *args, **kwargs):
            pass

        def publish(self, event_type: str, topic: str, payload=None):
            CapturingBus.events.append((event_type, topic, payload or {}))

        def notify_data_available(self, dataset: str, **meta):
            self.publish("data.available", dataset, meta)

    import ibx_flows.backfill as backfill_mod

    monkeypatch.setattr(backfill_mod, "FileEventBus", CapturingBus, raising=True)
    CapturingBus.events.clear()

    base = tmp_path
    repo = OptionChainSnapshotRepository(base)
    cfg = BackfillConfig(
        underlying="TEST",
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
    )

    backfill_option_chain_daily(FakeSource(), repo, cfg)

    out_dir = base / "option_chains" / "underlying=TEST"
    assert list(out_dir.rglob("*.parquet"))
    assert any(t == "option_chains" for _, t, _ in CapturingBus.events)


@pytest.mark.unit
def test_backfill_option_bars_writes_and_notifies(tmp_path, monkeypatch):
    from repositories.option_bars import OptionBarRepository
    from ibx_flows.backfill import BackfillConfig, backfill_option_bars

    class FakeSource:
        def get_option_chain(self, underlying, asof_date):
            # minimal chain snapshot used only for selection; will be filtered later
            return pd.DataFrame(
                [
                    {
                        "underlying": underlying,
                        "expiry": date(2024, 1, 31),
                        "right": "C",
                        "strike": 100.0,
                        "conid": 12345,
                    }
                ]
            )

        def get_equity_bars(self, symbol, start, end, bar_size):
            return pd.DataFrame(
                [
                    {
                        "date": start,
                        "symbol": symbol,
                        "bar_size": bar_size,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.5,
                        "close": 100.5,
                        "volume": 1000,
                    }
                ]
            )

        def get_option_bars(self, contract_df, start, end, bar_size):
            # produce a single bar row with required columns
            return pd.DataFrame(
                [
                    {
                        "time": datetime.combine(start, datetime.min.time()),
                        "underlying": contract_df.iloc[0]["underlying"],
                        "expiry": contract_df.iloc[0]["expiry"],
                        "right": contract_df.iloc[0]["right"],
                        "strike": float(contract_df.iloc[0]["strike"]),
                        "bar_size": bar_size,
                        "open": 1.0,
                        "high": 1.2,
                        "low": 0.9,
                        "close": 1.1,
                        "volume": 100,
                    }
                ]
            )

    class CapturingBus:
        events = []

        def __init__(self, *args, **kwargs):
            pass

        def publish(self, event_type: str, topic: str, payload=None):
            CapturingBus.events.append((event_type, topic, payload or {}))

        def notify_data_available(self, dataset: str, **meta):
            self.publish("data.available", dataset, meta)

    import ibx_flows.backfill as backfill_mod

    monkeypatch.setattr(backfill_mod, "FileEventBus", CapturingBus, raising=True)
    CapturingBus.events.clear()

    base = tmp_path
    repo = OptionBarRepository(base)
    cfg = BackfillConfig(
        underlying="TEST",
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        bar_size="1 day",
    )

    backfill_option_bars(FakeSource(), repo, chain_repo=None, eq_repo=None, cfg=cfg)

    out_dir = base / "option_bars" / "underlying=TEST"
    assert list(out_dir.rglob("*.parquet"))
    assert any(t == "option_bars" for _, t, _ in CapturingBus.events)

