from api.ib_core import IBRuntime, make_stock
from api.subscriptions import SubscriptionService


def main():
    with IBRuntime(port=4002, client_id=42) as rt:
        subs = SubscriptionService(rt)
        aapl = make_stock("AAPL")

        # Subscribe (callback optional). You can also just iterate the handle.
        handle = subs.market_data(aapl, genericTicks="",
                                  on_event=lambda e: print("EVENT", e["type"], e.get("tick")),
                                  qsize=1000)

        # Consume a few events then cancel
        n = 0
        for evt in handle:
            # evt is a dict with type: tickPrice/tickSize/...
            print(evt)
            n += 1
            if n >= 10:
                break

        handle.cancel()


if __name__ == "__main__":
    main()