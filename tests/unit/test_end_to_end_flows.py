from datetime import date, datetime, timedelta

import pandas as pd
import pytest


@pytest.mark.unit
def test_flow_chain_then_option_bars_with_source_spot(tmp_path, monkeypatch):
    from common.repositories.option_bars import OptionBarRepository
    from common.repositories.option_chains import OptionChainSnapshotRepository
    from ibx_flows.backfill import (
        backfill_option_bars,
        backfill_option_chain_daily,
        BackfillConfig,
    )

    # Fake IB source with chain snapshot, equity spot, and option bars
    class FakeSource:
        def get_option_chain(self, underlying, asof_date):
            # Two contracts, one call one put
            return pd.DataFrame(
                [
                    {
                        "underlying": underlying,
                        "date": asof_date,
                        "expiry": asof_date + timedelta(days=30),
                        "right": "C",
                        "strike": 100.0,
                        "conid": 111,
                    },
                    {
                        "underlying": underlying,
                        "date": asof_date,
                        "expiry": asof_date + timedelta(days=30),
                        "right": "P",
                        "strike": 100.0,
                        "conid": 222,
                    },
                ]
            )

        def get_equity_bars(self, symbol, start, end, bar_size):
            return pd.DataFrame(
                [
                    {
                        "date": end,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.5,
                        "close": 100.5,
                        "volume": 1000,
                    }
                ]
            )

        def get_option_bars(self, contract_df, start, end, bar_size):
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

        def notify_data_available(self, dataset: str, **meta):
            CapturingBus.events.append((dataset, meta))

    import ibx_flows.backfill as backfill_mod

    monkeypatch.setattr(backfill_mod, "FileEventBus", CapturingBus, raising=True)
    CapturingBus.events.clear()

    base = tmp_path
    chain_repo = OptionChainSnapshotRepository(base)
    opt_repo = OptionBarRepository(base)
    today = date.today()
    cfg = BackfillConfig(underlying="TST", start=today, end=today, bar_size="1 day")

    # Act: chain then option bars
    backfill_option_chain_daily(FakeSource(), chain_repo, cfg)
    backfill_option_bars(FakeSource(), opt_repo, chain_repo=None, eq_repo=None, cfg=cfg)

    # Assert files present
    assert list((base / "option_chains").rglob("*.parquet"))
    assert list((base / "option_bars").rglob("*.parquet"))
    # Assert events emitted for both topics
    topics = [t for t, _ in CapturingBus.events]
    assert "option_chains" in topics
    assert "option_bars" in topics


@pytest.mark.unit
def test_option_bars_requires_spot_unless_source_provides_it(tmp_path, monkeypatch):
    from common.repositories.option_bars import OptionBarRepository
    from ibx_flows.backfill import (
        backfill_equity_bars,
        backfill_option_bars,
        BackfillConfig,
    )

    # Source without equity spot -> raises
    class NoEqSource:
        def get_option_chain(self, underlying, asof_date):
            return pd.DataFrame(
                [
                    {
                        "underlying": underlying,
                        "expiry": asof_date + timedelta(days=30),
                        "right": "C",
                        "strike": 100.0,
                        "conid": 111,
                    }
                ]
            )

        def get_equity_bars(self, symbol, start, end, bar_size):
            return pd.DataFrame([])

        def get_option_bars(self, contract_df, start, end, bar_size):
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

    class EqOnlySource(NoEqSource):
        def get_equity_bars(self, symbol, start, end, bar_size):
            return pd.DataFrame(
                [
                    {
                        "date": end,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.5,
                        "close": 100.5,
                        "volume": 1000,
                    }
                ]
            )

    class CapturingBus:
        events = []

        def __init__(self, *args, **kwargs):
            pass

        def notify_data_available(self, dataset: str, **meta):
            CapturingBus.events.append((dataset, meta))

    import ibx_flows.backfill as backfill_mod

    monkeypatch.setattr(backfill_mod, "FileEventBus", CapturingBus, raising=True)
    CapturingBus.events.clear()

    base = tmp_path
    opt_repo = OptionBarRepository(base)
    today = date.today()
    cfg = BackfillConfig(underlying="TST", start=today, end=today, bar_size="1 day")

    # Missing spot raises
    with pytest.raises(RuntimeError):
        backfill_option_bars(
            NoEqSource(), opt_repo, chain_repo=None, eq_repo=None, cfg=cfg
        )

    # If source provides spot, it succeeds without prior equity repo
    backfill_option_bars(
        EqOnlySource(), opt_repo, chain_repo=None, eq_repo=None, cfg=cfg
    )
    assert list((base / "option_bars").rglob("*.parquet"))
