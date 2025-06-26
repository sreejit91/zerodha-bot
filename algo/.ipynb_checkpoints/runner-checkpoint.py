"""Live paper-trade loop: subscribe to real-time ticks & generate signals."""
from __future__ import annotations
import threading, time, datetime as _dt, logging
import twisted.internet.base, twisted.internet._signals
from zoneinfo import ZoneInfo

# ── Disable Twisted’s own signal-hooks so Ctrl-C works ───────────────────────
twisted.internet.base.installSignalHandlers = lambda *a, **k: None
twisted.internet._signals.install = lambda *a, **k: None

from kiteconnect import KiteTicker
from .broker   import KiteWrapper
from .config   import load_config
from .features import add_indicators
from .model    import load_or_train, predict_last
from .signals  import generate_signal
from .logger   import TradeLogger

def run_live(symbol: str, qty: int, live: bool = True):
    """
    symbol → tradingsymbol (e.g. 'RELIANCE')
    qty    → shares per trade
    live   → True for real orders, False for paper
    """
    # ── Bootstrap & creds
    cfg    = load_config()
    broker = KiteWrapper(cfg)
    logging.debug("Trading %s → token=%s", symbol, cfg.instrument_token)
    logger = TradeLogger()

    # ── Warm-up historical bars & features
    hist = broker.history(days=90, interval="5minute")
    hist = add_indicators(hist)
    logging.debug("Fetched %d bars up to %s", len(hist), hist.index.max())

    model   = load_or_train(hist, retrain=False)

    # ── Prepare live DataFrame buffer (tz-aware)
    live_df = hist.copy()
    live_df.index = live_df.index.tz_convert("Asia/Kolkata")
    lock     = threading.Lock()

    # ── Tick handler ─────────────────────────────────────────────────────────
    def on_ticks(ws, ticks):
        for tick in ticks:
            with lock:
                # Use exchange timestamp if available, else now()
                now = tick.get("exchange_timestamp",
                               _dt.datetime.now(_dt.timezone.utc))
                if now.tzinfo is None:
                    now = now.replace(tzinfo=_dt.timezone.utc)
                # Convert to IST
                now = now.astimezone(ZoneInfo("Asia/Kolkata"))

                price = tick["last_price"]
                live_df.loc[now] = {
                    "open":   price,
                    "high":   price,
                    "low":    price,
                    "close":  price,
                    "volume": tick.get("volume", 0),
                }

                df2  = add_indicators(live_df)
                prob = predict_last(model, df2)

                # debug print
                print(
                    f"{now:%H:%M:%S}  P↑={prob:.2f}  "
                    f"EMA8={df2.ema_8.iat[-1]:.2f}  "
                    f"EMA21={df2.ema_21.iat[-1]:.2f}  "
                    f"Trend={'↑' if df2.ema_8.iat[-1]>df2.ema_21.iat[-1] else '↓'}"
                )

                sig = generate_signal(df2, model)
                if sig:
                    print(f"SIGNAL: {sig} @ {price:.2f}  qty={qty}  paper={not live}")
                    logger.log(symbol, sig, price, qty, live)

    # ── WebSocket setup ──────────────────────────────────────────────────────
    kws = KiteTicker(cfg.api_key, cfg.access_token)
    kws.on_connect = lambda ws, _: ws.subscribe([cfg.instrument_token])
    kws.on_ticks   = on_ticks

    print("🚀 Live paper-trade loop started – Ctrl-C to stop …")
    kws.connect(threaded=True)

    # ── Keep running until Ctrl-C ────────────────────────────────────────────
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        kws.close()               # cleanly shut websocket
        logger.summary()
        print("🏁 Session over |", logger.metrics())