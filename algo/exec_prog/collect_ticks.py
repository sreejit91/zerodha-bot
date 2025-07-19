import os
import json
import time
from datetime import datetime, timedelta, timezone
from algo.broker import KiteWrapper
from algo.tick_aggregator import TickAggregator
from kiteconnect import KiteTicker

def main():
    symbols_input = input("Enter symbols (comma-separated): ")
    symbols = [s.strip().upper() for s in symbols_input.split(",")]
    secs = input("Enter intervals in seconds [leave blank to skip bars]: ")
    intervals = [int(s) for s in secs.split(",")] if secs.strip() else []
    data_dir = input("Enter data directory (default 'live_tick_data'): ").strip() or "../live_tick_data"
    print(f"[collect_ticks] Using data directory: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    kw = KiteWrapper()
    aggs = [TickAggregator(kw, sym, intervals, data_dir) for sym in symbols]

    # Key: (symbol, date), Value: file object
    rawloggers = {}

    IST = timezone(timedelta(hours=5, minutes=30))  # Indian Standard Time

    def get_tick_date(tick):
        ts_str = tick.get("last_trade_time") or tick.get("exchange_timestamp")
        ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
        return ts.date().isoformat(), ts.time()

    def on_connect(ws, _):
        tokens = [agg.token for agg in aggs]
        print(f"✅ Connected. Subscribing (FULL mode) to tokens: {tokens}")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_ticks(ws, ticks):
        for t in ticks:
            for sym, agg in zip(symbols, aggs):
                if t["instrument_token"] == agg.token:
                    date_str, time_obj = get_tick_date(t)
                    # Only collect ticks from 09:15 to 15:30 IST
                    if time_obj < datetime.strptime("09:15", "%H:%M").time() or time_obj > datetime.strptime("15:30", "%H:%M").time():
                        continue
                    key = (sym, date_str)
                    if key not in rawloggers:
                        # Close previous file for this symbol if open (from previous date)
                        for (s, d), f in list(rawloggers.items()):
                            if s == sym and d != date_str:
                                f.close()
                                del rawloggers[(s, d)]
                        fn = f"{sym}_{date_str}_ticks.jsonl"
                        rawloggers[key] = open(os.path.join(data_dir, fn), "a")
                    # Write raw JSON line
                    rawloggers[key].write(json.dumps(t, default=str) + "\n")
                    rawloggers[key].flush()
                    # Feed into bar aggregator
                    agg.on_tick(t)
                    break

    def on_close(ws, code, reason):
        print(f"🔴 Disconnected ({code}/{reason}), attempting reconnect…")

    def on_error(ws, error):
        print(f"⚠️ Socket error: {error!r}, reconnecting…")

    cfg = kw.cfg
    ws = KiteTicker(
        cfg.api_key,
        cfg.access_token,
        reconnect=True,
        reconnect_max_tries=5
    )

    ws.on_connect = on_connect
    ws.on_ticks = on_ticks
    ws.on_close = on_close
    ws.on_error = on_error

    print(f"🟢 Starting live collection for {symbols}…")
    ws.connect(threaded=True)

    try:
        while True:
            now_ist = datetime.now(IST)
            # Stop at or after 3:30 PM IST
            if now_ist.hour > 15 or (now_ist.hour == 15 and now_ist.minute >= 31):
                print("⏰ 3:30 PM reached, stopping collector for the day.")
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Stopping live collector.")
    finally:
        ws.close()
        for f in rawloggers.values():
            f.close()

if __name__ == "__main__":
    main()
