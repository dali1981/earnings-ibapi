import threading, time, logging

from option_ticks import LiveVolTap


def run_for(app, seconds, host="127.0.0.1", port=7497, client_id=42):
    logging.info("Connecting…")
    app.connect(host, port, client_id)

    # mark connection ready
    if not hasattr(app, "connected_evt"):
        import threading as _t
        app.connected_evt = _t.Event()
    # ensure your EWrapper sets connected_evt in nextValidId()

    # timer will call disconnect even while app.run() is blocking
    killer = threading.Timer(seconds, lambda: (setattr(app, "done", True), app.disconnect()))
    killer.daemon = True
    killer.start()

    try:
        app.run()   # BLOCKS here; callbacks fire normally
    finally:
        killer.cancel()
        # second-chance disconnect; safe if already disconnected
        try: app.disconnect()
        except: pass


if __name__ == "__main__":
    # after you define LiveVolTap/HistTap
    app = LiveVolTap()  # or HistTap
    run_for(app, seconds=25)