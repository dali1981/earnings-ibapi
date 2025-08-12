from api.ib_core import IBRuntime, ContractDetailsService, SecDefService, HistoricalService, make_stock


def main():
    """
    Example usage of the IBRuntime and services to fetch contract details,
    option parameters, and historical bars for a stock.
    """
    with IBRuntime(host="127.0.0.1", port=4002, client_id=7) as rt:
        cds = ContractDetailsService(rt)
        secdef = SecDefService(rt)
        hist = HistoricalService(rt)

        aapl = make_stock("AAPL")
        details = cds.fetch(aapl, timeout=10)
        conid = details[0].contract.conId

        params = secdef.option_params("AAPL", conid, timeout=10)
        bars = hist.bars(aapl, endDateTime="", durationStr="2 D",
                         barSizeSetting="1 hour", whatToShow="TRADES", useRTH=0, timeout=20)

        print(f"Contract Details: {details}")
        print(f"Option Parameters: {params}")
        print(f"Historical Bars: {bars}")



if __name__ == "__main__":
    main()