"""
algo/runner.py
Back-test entry points & live (paper) trading loop.
"""

import threading
import logging
import pandas as pd

from algo import KiteWrapper
from algo import add_indicators
from algo import load_or_train, LOOKBACK
from algo import generate_signal
from algo import TradeLogger
from algo import load_config


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def run_live(symbol: str, quantity: int, live: bool = False):
    cfg    = load_config()
    broker = KiteWrapper(cfg)

    logger.info("Initialising live feed for %s (token=%s)…",
                cfg.tradingsymbol, cfg.instrument_token)

    hist  = broker.history(days=90, interval="5minute")
    model = load_or_train(add_indicators(hist))

    live_df = hist.copy()
    if live_df.index.tz is None:
        live_df.index = live_df.index.tz_localize("Asia/Kolkata")
    else:
        live_df.index = live_df.index.tz_convert("Asia/Kolkata")

    lock         = threading.Lock()
    trade_logger = TradeLogger()

    # ── tick callback ───────────────────────────────────────────────
    def on_ticks(ws, ticks):
        if not ticks:
            return
        tick = ticks[0]
        ts = pd.to_datetime(tick["timestamp"], utc=True).tz_convert("Asia/Kolkata")

        with lock:
            live_df.loc[ts] = {
                "open": tick["last_price"],
                "high": tick["last_price"],
                "low":  tick["last_price"],
                "close": tick["last_price"],
                "volume": tick["volume"],
            }
            live_df.sort_index(inplace=True)
            df_trunc = live_df.iloc[-LOOKBACK:]

            sig = generate_signal(add_indicators(df_trunc.copy()), model)
            order = broker.place_order(symbol, quantity, sig, live=live)
            if order:
                trade_logger.log(order)

    kws = broker.ticker
    kws.on_connect = lambda ws, resp: ws.subscribe([cfg.instrument_token])
    kws.on_ticks   = on_ticks

    logger.info("🚀 Live session started (paper=%s).  Ctrl-C to exit …", not live)
    try:
        kws.connect(threaded=False)     # blocks
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt – shutting down.")
    finally:
        logger.info(
            "🏁 Session done | Trades=%d | Total PnL ₹%.2f | Win-rate %.1f%%",
            trade_logger.trades,
            trade_logger.total_pnl,
            trade_logger.win_rate * 100,
        )
