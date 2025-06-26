"""
run_paper_ws.py  –  Live 3-minute paper-trading loop
────────────────────────────────────────────────────────────────────────
• EMA-3/4/8/9/21, ATR, RSI, MACD, OBV, rv20
• Seeds 70 bars so indicators are warm
• SL / TP / TRAIL / TIME exits
• Debug print of ema crossover + position age
Ctrl-C to stop.
"""

from __future__ import annotations
import datetime as dt, collections
import pandas as pd
from kiteconnect import KiteTicker

from algo.config   import load_config
from algo.broker   import KiteWrapper
from algo.features import add_indicators
from algo.model    import load_or_train, predict_last, LOOKBACK
from algo.backtester import zerodha_cost

# ─── parameters ──────────────────────────────────────────────────────
SYMBOL    = "RELIANCE"
TOKEN     = 738561
INTERVAL  = 180          # 3-minute bar
WINDOW    = 70           # deque length  (>= LOOKBACK + warm-up)

SL_PCT    = 0.0012       # initial stop  0.12 %
TP_PCT    = 0.0025       # fixed target  0.20 %
TRAIL_PCT = 0.0025       # distance of trailing stop from price (0.15 %)
HOLD_MAX  = 15           # time-stop after 30 bars  (≈ 90 min)

UPPER     = 0.63         # long gate
LOWER     = 0.37         # short gate
CONTRACTS = 1

# trade windows
WINDOWS   = [(dt.time(9, 15), dt.time(15, 25))]

# ─── initialise ──────────────────────────────────────────────────────
cfg  = load_config()
wrap = KiteWrapper(cfg)

model = load_or_train(
    add_indicators(
        wrap.history(days=1, interval="3minute", tradingsymbol=SYMBOL)
    ),
    retrain=False,
)

bars: "collections.deque[pd.Series]" = collections.deque(maxlen=WINDOW)

# seed deque
for ts, row in wrap.history(days=1, interval="3minute",
                            tradingsymbol=SYMBOL).tail(WINDOW).iterrows():
    s = row.copy()
    s.name = ts.replace(second=0, microsecond=0)
    s["_processed"] = True
    bars.append(s)

# ─── position state ──────────────────────────────────────────────────
in_pos   = 0        # -1 short, 0 flat, +1 long
entry_px = 0.0
sl_px    = 0.0
trade_age = 0       # bars inside current position
pnl      = 0.0

probs: list[float] = []     # diagnostics

# ─── websocket handlers ──────────────────────────────────────────────
def on_ticks(ws, ticks):
    global in_pos, entry_px, sl_px, trade_age, pnl
    try:
        for t in ticks:
            ltp = t["last_price"]
            ts  = (t.get("last_trade_time")
                   or dt.datetime.now(dt.timezone.utc)).astimezone()

            bar_ts      = ts.replace(second=0, microsecond=0)
            current_key = bars[-1].name if bars else None

            # ── bar rollover ────────────────────────────────────────
            if current_key != bar_ts:
                if bars:
                    prev = bars[-1]
                    if "_last" in prev:
                        prev["close"] = prev["_last"]
                        prev["high"]  = max(prev["high"], prev["_last"])
                        prev["low"]   = min(prev["low"],  prev["_last"])
                        del prev["_last"]
                bars.append(pd.Series(dict(open=ltp, high=ltp, low=ltp,
                                           close=ltp, volume=1, _last=ltp),
                                       name=bar_ts))
            else:
                bar = bars[-1]
                bar["_last"] = ltp
                bar["high"]  = max(bar["high"], ltp)
                bar["low"]   = min(bar["low"],  ltp)
                bar["volume"] += 1

            # need a newly closed bar
            if len(bars) < 2 or bars[-2].get("_processed"):
                return

            bar_cl = bars[-2]
            bar_cl["_processed"] = True
            price   = bar_cl["close"]
            ts_bar  = bar_cl.name.time()

            # ── build feature frame ────────────────────────────────
            df_win = (pd.DataFrame(list(bars))
                      .drop(columns=["_processed", "_last"], errors="ignore"))
            df_feat = add_indicators(df_win.copy()).ffill()
            df_feat_closed = df_feat.iloc[:-1]

            if (len(df_feat_closed) < LOOKBACK or
                df_feat_closed.tail(LOOKBACK).isna().any().any()):
                print(ts_bar, "waiting for full LOOKBACK …")
                return

            prob = predict_last(df_feat_closed, model)
            probs.append(prob)
            if len(probs) == 60:      # hourly diagnostics
                print("\nProb summary:",
                      "min", f"{min(probs):.2f}",
                      "mean", f"{sum(probs)/len(probs):.2f}",
                      "95-pct", f"{sorted(probs)[int(0.95*len(probs))]:.2f}",
                      "max", f"{max(probs):.2f}\n")
                probs.clear()

            ema3 = df_feat_closed["ema_3"].iat[-1]
            ema8 = df_feat_closed["ema_8"].iat[-1]

            print(ts_bar, "DEBUG",
                  f"ema3={ema3:.2f}", f"ema8={ema8:.2f}",
                  "longTrend", ema3 > ema8,
                  "shortTrend", ema3 < ema8)

            long_ok  = prob >= UPPER and ema3 > ema8
            short_ok = prob <= LOWER and ema3 < ema8
            if not any(a <= ts_bar <= b for a, b in WINDOWS):
                long_ok = short_ok = False

            # ─── manage open position ───────────────────────────────
            if in_pos:
                trade_age += 1

                # --- trailing stop update --------------------------
                trail_dist = TRAIL_PCT * entry_px
                if in_pos == 1:            # long
                    new_sl = price - trail_dist
                    if new_sl > sl_px:
                        sl_px = new_sl
                else:                      # short
                    new_sl = price + trail_dist
                    if new_sl < sl_px:
                        sl_px = new_sl
                # ---------------------------------------------------

                pl = (price - entry_px) / entry_px * in_pos
                sl_exit   = (price <= sl_px) if in_pos == 1 else (price >= sl_px)
                tp_exit   = pl >= TP_PCT if in_pos == 1 else pl <= -TP_PCT
                time_exit = trade_age >= HOLD_MAX

                if sl_exit or tp_exit or time_exit:
                    reason = ("SL" if sl_exit else
                              "TP" if tp_exit else "TIME")
                    fee = zerodha_cost(price*CONTRACTS,
                                       "SELL" if in_pos == 1 else "BUY")
                    pnl += pl*100_000*CONTRACTS - fee
                    print(ts_bar, f"EXIT {reason}",
                          f"{price:.2f}", "pnl", f"{pnl:+,.0f}",
                          "held", trade_age, "bars")
                    in_pos = 0
                    trade_age = 0

            # ─── check for new entry ────────────────────────────────
            if in_pos == 0:
                if long_ok:
                    in_pos, entry_px = 1, price
                    sl_px = entry_px * (1 - SL_PCT)
                    trade_age = 0
                    pnl -= zerodha_cost(price*CONTRACTS, "BUY")
                    print(ts_bar, "BUY ", f"{entry_px:.2f}")
                elif short_ok:
                    in_pos, entry_px = -1, price
                    sl_px = entry_px * (1 + SL_PCT)
                    trade_age = 0
                    pnl -= zerodha_cost(price*CONTRACTS, "SELL")
                    print(ts_bar, "SELL", f"{entry_px:.2f}")

            # heartbeat
            age_txt = trade_age if in_pos else "-"
            print(ts_bar, "bar close", f"{price:.2f}",
                  "prob", f"{prob:.2f}", "pos", in_pos,
                  "age", age_txt, "SL", f"{sl_px:.2f}" if in_pos else "-")

    except Exception as e:
        print("Handler error:", e)

def on_connect(ws, _):
    print("✅ websocket connected")
    ws.subscribe([TOKEN])
    ws.set_mode(ws.MODE_QUOTE, [TOKEN])

# ─── run ─────────────────────────────────────────────────────────────
print("🟢 Connecting websocket …  Ctrl-C to stop")
ws = KiteTicker(cfg.api_key, cfg.access_token)
ws.on_connect = on_connect
ws.on_ticks   = on_ticks

try:
    ws.connect(threaded=False)
except KeyboardInterrupt:
    print("\n🔴 stopped. Net PnL:", round(pnl, 2))
    ws.close()
