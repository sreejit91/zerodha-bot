"""
run_paper.py
────────────────────────────────────────────────────────────────────────
Live 3-minute paper-trading loop (no real orders).

• Loads the latest algo/model.pkl
• Applies the scalper parameters that back-tested at ~₹1.6 k / 20 sessions
• Uses a 60-bar rolling window so MACD, ATR, RSI, etc. always have data
• Logs every entry/exit and fee-adjusted PnL
"""

import time, datetime as dt
import pandas as pd

from algo.broker   import KiteWrapper
from algo.config   import load_config
from algo.features import add_indicators
from algo.model    import load_or_train, predict_last
from algo.backtester import zerodha_cost       # fee helper

# ─── strategy parameters (same as profitable back-test) ─────────────
SYMBOL       = "RELIANCE"
INTERVAL     = "3minute"
WINDOW       = 60           # bars to fetch each poll (≈ 3 hours)
SL_PCT       = 0.0012       # 0.12 %
TP_PCT       = 0.0027       # 0.27 %
UPPER_PROB   = 0.58
LOWER_PROB   = 0.42
CONTRACTS    = 1            # = ₹1 L notional per trade
# trade only during:
WINDOWS = [(dt.time(9,30), dt.time(13,00)),
           (dt.time(13,30), dt.time(15,0))]
# ────────────────────────────────────────────────────────────────────

cfg = load_config()
kw  = KiteWrapper(cfg)

# ── load latest model pickle ────────────────────────────────────────
hist  = kw.history(days=1, interval=INTERVAL, tradingsymbol=SYMBOL)
model = load_or_train(add_indicators(hist), retrain=False)

in_pos   = 0      # +1 long, –1 short, 0 flat
entry_px = 0.0
pnl      = 0.0

print("🟢 Paper loop started", dt.datetime.now().strftime("%H:%M:%S"))

try:
    while True:
        # 1️⃣  fetch rolling window & feature-engineer
        df = kw.history(days=1, interval=INTERVAL,
                        tradingsymbol=SYMBOL).tail(WINDOW)
        df = add_indicators(df)

        if len(df) < WINDOW:      # just in case market just opened
            time.sleep(10)
            continue

        # 2️⃣  use the most-recent completed bar
        bar     = df.tail(1)
        ts      = bar.index[-1]
        price   = bar["close"].iat[-1]

        # 3️⃣  skip outside trade windows
        if not any(a <= ts.time() <= b for a, b in WINDOWS):
            time.sleep(10)
            continue

        # 4️⃣  ML probability on full window
        prob = predict_last(df, model)

        # 5️⃣  signal with EMA-3 / EMA-8 trend gate
        if prob >= UPPER_PROB and bar["ema_3"].iat[-1] > bar["ema_8"].iat[-1]:
            signal = "BUY"
        elif prob <= LOWER_PROB and bar["ema_3"].iat[-1] < bar["ema_8"].iat[-1]:
            signal = "SELL"
        else:
            signal = None

        # 6️⃣  EXIT logic
        if in_pos:
            pl_pct = (price - entry_px) / entry_px * in_pos
            if pl_pct <= -SL_PCT or pl_pct >= TP_PCT:
                fee = zerodha_cost(price * CONTRACTS,
                                   "SELL" if in_pos == 1 else "BUY")
                pnl += pl_pct * 100_000 * CONTRACTS - fee
                print(f"{ts.time()}  EXIT  {price:.2f}  pnl {pnl:+.0f}")
                in_pos = 0

        # 7️⃣  ENTRY logic
        if in_pos == 0 and signal:
            in_pos   = 1 if signal == "BUY" else -1
            entry_px = price
            fee = zerodha_cost(price * CONTRACTS,
                               "BUY" if in_pos == 1 else "SELL")
            pnl -= fee
            print(f"{ts.time()}  {signal}  {price:.2f}")
        print(
        f"{ts.time()}  "
        f"prob {prob:.2f}  "
        f"{'ema3>8' if bar['ema_3'].iat[-1] > bar['ema_8'].iat[-1] else 'ema3<8'}  "
        f"pos {in_pos}"
        )    # poll well within the 3-min bar

except KeyboardInterrupt:
    print("\n🔴 Paper loop stopped  |  Net PnL:", round(pnl, 2))
