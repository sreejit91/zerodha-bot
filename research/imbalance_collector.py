# research/imbalance_collector.py
"""
Collect best-bid / best-ask imbalance for one instrument and
write every tick to CSV.  Run ONLY in research; it never touches
your live trades DB.
"""
import csv, datetime as dt, pathlib, sys
from kiteconnect import KiteTicker

from algo.broker import KiteWrapper, _resolve_token   # live broker code
from research.config import load_config               # reads project-root config.json

# ── 1) Credentials & token ──────────────────────────────────────────
cfg   = load_config()
wrap  = KiteWrapper(cfg)                              # only for token lookup

TOKEN = _resolve_token(
    wrap.kite,
    tradingsymbol=cfg.tradingsymbol,
    exchange=cfg.exchange,
)

# ── 2) CSV output (append per day) ──────────────────────────────────
today_str = dt.datetime.now().strftime("%Y-%m-%d")
csv_path  = pathlib.Path(f"imb_tape_{today_str}.csv")
new_file  = not csv_path.exists()

outf   = open(csv_path, "a", newline="")
writer = csv.writer(outf)
if new_file:
    writer.writerow(["ts_utc", "imb"])               # header if fresh file

row_cnt = 0

# ── 3) Tick handler ────────────────────────────────────────────────
def on_ticks(ws, ticks):
    global row_cnt
    t = ticks[0]

    # depth may be empty on first tick of the day; guard
    try:
        bid_qty = t["depth"]["buy"][0]["quantity"]
        ask_qty = t["depth"]["sell"][0]["quantity"]
    except (KeyError, IndexError):
        return

    if bid_qty + ask_qty:
        imb = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        ts  = dt.datetime.utcnow().isoformat(timespec="seconds")
        writer.writerow([ts, f"{imb:.6f}"])
        row_cnt += 1

        # console heartbeat
        if row_cnt % 200 == 0:
            print(f"… {row_cnt:>7} ticks captured", flush=True)

        # flush every 500 rows
        if row_cnt % 500 == 0:
            outf.flush()

# ── 4) Connect banner & start socket ───────────────────────────────
def on_connect(ws, _):
    print("✅ WebSocket connected — subscribing…", flush=True)
    ws.subscribe([TOKEN])
    ws.set_mode(ws.MODE_FULL, [TOKEN])

def on_close(ws, code, reason):
    print(f"⚠️  Socket closed ({code}) {reason}", flush=True)
    outf.flush()
    outf.close()
    sys.exit()

ws = KiteTicker(cfg.api_key, cfg.access_token, debug=False)
ws.on_connect = on_connect
ws.on_ticks   = on_ticks
ws.on_close   = on_close

print("⏳ Opening ticker socket …")
ws.connect(threaded=False)       # blocks until Ctrl-C / disconnection
