import os
import pytest

pytestmark = [pytest.mark.integration]

if not os.environ.get("RUN_IB_TESTS"):
    pytest.skip("IB tests require a running IB Gateway/TWS; set RUN_IB_TESTS=1 to enable.", allow_module_level=True)

from ibx import IBRuntime, SubscriptionService, make_stock

def test_ib_runtime_snapshot():
    with IBRuntime(port=4002, client_id=42) as rt:
        subs = SubscriptionService(rt)
        aapl = make_stock("AAPL")
        handle = subs.market_data(aapl)
        for i, evt in enumerate(handle):
            assert evt is not None
            if i == 10:
                break
        handle.cancel()
