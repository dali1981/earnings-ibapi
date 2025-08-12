from ibx import IBRuntime, SubscriptionService, make_stock

with IBRuntime(port=4002, client_id=42) as rt:
    subs = SubscriptionService(rt)
    aapl = make_stock("AAPL")
    handle = subs.market_data(aapl)
    for i, evt in enumerate(handle):
        print(evt)
        if i == 10: break
    handle.cancel()
