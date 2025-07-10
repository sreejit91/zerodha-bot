import os
import json
import time
from datetime import datetime
from algo.broker import KiteWrapper
from algo.tick_aggregator import TickAggregator
from kiteconnect import KiteTicker

def main():
    # 1) User inputs
    symbols_input = input("Enter symbols (comma-separated): ")
    symbols       = [s.strip().upper() for s in symbols_input.split(",")]

    secs = input("Enter intervals in seconds [leave blank to skip bars]: ")
    intervals = [int(s) for s in secs.split(",")] if secs.strip() else []

    data_dir = input("Enter data directory (default 'live_tick_data'): ").strip() or "../live_tick_data"
    print(f"[collect_ticks] Using data directory: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    # 2) Prepare raw tick log files
    rawloggers = {
        sym: open(os.path.join(data_dir, f"{sym}_ticks.jsonl"), "a")
        for sym in symbols
    }

    # 3) Initialize KiteWrapper for token resolution
    kw = KiteWrapper()

    # 4) Build one TickAggregator per symbol
    aggs = [TickAggregator(kw, sym, intervals, data_dir) for sym in symbols]

    # 5) WebSocket callbacks
    def on_connect(ws, _):
        tokens = [agg.token for agg in aggs]
        print(f"✅ Connected. Subscribing (FULL mode) to tokens: {tokens}")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_ticks(ws, ticks):
        for t in ticks:
            for sym, agg in zip(symbols, aggs):
                if t["instrument_token"] == agg.token:
                    # write raw JSON line
                    rawloggers[sym].write(json.dumps(t, default=str) + "\n")
                    rawloggers[sym].flush()
                    # feed into bar aggregator
                    agg.on_tick(t)
                    break

    def on_close(ws, code, reason):
        print(f"🔴 Disconnected ({code}/{reason}), attempting reconnect…")

    def on_error(ws, error):
        print(f"⚠️ Socket error: {error!r}, reconnecting…")

    # 6) Create KiteTicker with auto-reconnect
    cfg = kw.cfg
    ws  = KiteTicker(
        cfg.api_key,
        cfg.access_token,
        reconnect=True,
        reconnect_max_tries=5
    )

    ws.on_connect = on_connect
    ws.on_ticks   = on_ticks
    ws.on_close   = on_close
    ws.on_error   = on_error

    # 7) Start the websocket in threaded mode
    print(f"🟢 Starting live collection for {symbols}…")
    ws.connect(threaded=True)

    # 8) Keep main thread alive so reconnects can occur
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Stopping live collector.")
    finally:
        # clean up
        ws.close()
        for f in rawloggers.values():
            f.close()

if __name__ == "__main__":
    main()
