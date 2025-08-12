import threading, time

class AppRunner:
    def __init__(self, app, host="127.0.0.1", port=7497, client_id=42):
        self.app, self.host, self.port, self.client_id = app, host, port, client_id
        self.t = None
        if not hasattr(app, "connected_evt"):
            app.connected_evt = threading.Event()

    def start(self, wait=10):
        self.app.connect(self.host, self.port, self.client_id)
        # run loop on NON-daemon thread so we can join it
        self.t = threading.Thread(target=self.app.run, name="ib-run", daemon=False)
        self.t.start()
        if not self.app.connected_evt.wait(wait):
            raise TimeoutError("nextValidId not received; connection not ready")

    def stop(self, join_timeout=5):
        # optional: cancel all to be nice with pacing
        try: self.app.reqGlobalCancel()
        except: pass
        # set a 'done' flag (some IB samples check this) and disconnect
        setattr(self.app, "done", True)
        self.app.disconnect()
        if self.t:
            self.t.join(timeout=join_timeout)


if __name__ == "__main__":
    runner = AppRunner(app, client_id=42)
    runner.start()
    # … send reqMktData / reqHistoricalData …
    time.sleep(25)
    runner.stop()