"""
run_paper_ws.py  –  Live 3-minute paper-trading loop  (debug v5)

NEW IN v5
─────────
• Integrates the SQLite-based logger (logger.py):
  ─ logs every OPEN (side, qty, price, SL, TP)
  ─ logs every CLOSE (reason, fee breakdown, gross & net P/L)
• All previous fixes: quantity-scaled TP, detailed console prints, etc.
"""

from __future__ import annotations
import datetime as dt, sys, collections, itertools
import pandas as pd
from kiteconnect import KiteTicker

from algo.config   import load_config
from algo.broker   import KiteWrapper
from algo.features import add_indicators
from algo.model    import load_or_train, predict_last, LOOKBACK
from algo.backtester import calculate_zerodha_fees

# 🚀  TRADE LOGGER  ───────────────────────────────────────────────────
from logger import TradeLogger, OpenTrade, CloseTrade
logger = TradeLogger()                            # durable SQLite log

# ─── parameters ──────────────────────────────────────────────────────
SYMBOL, TOKEN = "HDFCBANK", 341249
INTERVAL      = 180
WINDOW        = LOOKBACK + 10

INITIAL_CAPITAL = 100_000
SL_PCT, TP_PCT, TRAIL_PCT = 0.0015, 0.0045, 0.0020
HOLD_MAX  = 10
UPPER, LOWER = 0.50, 0.40

WINDOWS = [(dt.time(9, 15), dt.time(15, 25))]
is_mkt_hours = lambda t: any(a <= t <= b for a, b in WINDOWS)
now_ist = lambda: dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))

# ─── bootstrap ───────────────────────────────────────────────────────
print(f"🟢 Boot {now_ist():%Y-%m-%d %H:%M:%S} IST", flush=True)

cfg  = load_config()
wrap = KiteWrapper(cfg)

print("⏳ Loading model / warm history …", flush=True)
hist = wrap.history(days=200, interval="3minute", tradingsymbol=SYMBOL)
df   = add_indicators(hist).ffill()
model = load_or_train(df.iloc[:-20], retrain=False)
print("✅ Model ready", flush=True)

# seed bars
bars: collections.deque[pd.Series] = collections.deque(maxlen=WINDOW)
seed = wrap.history(days=1, interval="3minute", tradingsymbol=SYMBOL).tail(WINDOW)
for ts, row in seed.iterrows():
    s = row.copy()
    s.name = ts.replace(second=0, microsecond=0)
    s["_processed"] = True
    s["_last"] = s["close"]
    bars.append(s)

print(f"✅ Warm-up completed with {len(bars)} cached bars; ready to trade.", flush=True)

# ─── trading state ───────────────────────────────────────────────────
equity = INITIAL_CAPITAL
in_pos = 0
entry_px = sl_px = tp_px = 0.0
trade_age = 0
tick_counter = itertools.count(1)

# ─── websocket handlers ──────────────────────────────────────────────
def on_connect(ws, _):
    print("✅ Connected – subscribing now …", flush=True)
    print(f"   Tokens → [{TOKEN}]", flush=True)
    ws.subscribe([TOKEN])
    ws.set_mode(ws.MODE_QUOTE, [TOKEN])

def on_ticks(ws, ticks):
    global equity, in_pos, entry_px, sl_px, tp_px, trade_age

    # heartbeat
    for _ in ticks:
        c = next(tick_counter)
        if c % 100 == 0:
            print(f"📡 heartbeat tick #{c}", flush=True)

    # build bars
    for t in ticks:
        ltp = t["last_price"]
        ts  = (t.get("last_trade_time") or dt.datetime.now(dt.timezone.utc)).astimezone()
        key = ts.replace(second=0, microsecond=0)

        if not bars or bars[-1].name != key:
            if bars and "_last" in bars[-1]:
                prev = bars[-1]
                prev["close"] = prev.pop("_last")
                prev["high"]  = max(prev["high"], prev["close"])
                prev["low"]   = min(prev["low"], prev["close"])
            bars.append(pd.Series({"open": ltp, "high": ltp, "low": ltp,
                                   "close": ltp, "volume": 1, "_last": ltp},
                                  name=key))
        else:
            b = bars[-1]
            b["_last"]  = ltp
            b["high"]   = max(b["high"], ltp)
            b["low"]    = min(b["low"], ltp)
            b["volume"] += 1

    # process closed bar
    if len(bars) < 2 or bars[-2].get("_processed"):
        return
    bar = bars[-2]; bar["_processed"] = True
    price, ts_time = bar["close"], bar.name.time()

    df_win  = pd.DataFrame([b.drop(["_processed", "_last"], errors="ignore") for b in bars])
    df_feat = add_indicators(df_win).ffill()
    df_closed = df_feat.iloc[:-1]
    if len(df_closed) < LOOKBACK:
        return

    prob = predict_last(df_closed, model)
    ema3, ema8 = df_closed["ema_3"].iat[-1], df_closed["ema_8"].iat[-1]
    long_ok  = prob >= UPPER and ema3 > ema8 and is_mkt_hours(ts_time)
    short_ok = prob <= LOWER and ema3 < ema8 and is_mkt_hours(ts_time)

    print(f"{ts_time} prob={prob:.2f} ema3={ema3:.2f} ema8={ema8:.2f} "
          f"long_ok={long_ok} short_ok={short_ok}", flush=True)

    # ── manage open trade ──────────────────────────────────────────
    if in_pos:
        trade_age += 1
        sl_px = max(sl_px, price - entry_px * TRAIL_PCT) if in_pos > 0 else \
                min(sl_px, price + entry_px * TRAIL_PCT)

        pnl = (price - entry_px) * in_pos
        exit_sl = (price <= sl_px) if in_pos > 0 else (price >= sl_px)
        exit_tp = (pnl >=  TP_PCT * abs(entry_px) * abs(in_pos)) if in_pos > 0 else \
                  (pnl <= -TP_PCT * abs(entry_px) * abs(in_pos))
        exit_tm = (trade_age >= HOLD_MAX)

        if exit_sl or exit_tp or exit_tm:
            reason = "SL" if exit_sl else "TP" if exit_tp else "TIME"
            fee_detail = calculate_zerodha_fees(entry_px, price,
                                                abs(in_pos), debug=True)
            fee_total = fee_detail.get("total", sum(fee_detail.values())) \
                        if isinstance(fee_detail, dict) else fee_detail
            net = pnl - fee_total
            equity += net

            print(f"💔 {ts_time} EXIT {reason} @ {price:.2f}  "
                  f"gross={pnl:+.2f}  fees={fee_total:.2f}  net={net:+.2f}\n"
                  f"   fee-breakup → {fee_detail}", flush=True)

            # 📜  log CLOSE
            logger.log_close(
                CloseTrade(
                    timestamp=bar.name,
                    symbol=SYMBOL,
                    qty=abs(in_pos),
                    price=price,
                    reason=reason,
                    fee_detail=fee_detail if isinstance(fee_detail, dict)
                               else {"total": fee_total},
                    gross_pnl=pnl,
                    net_pnl=net,
                )
            )

            in_pos = 0; trade_age = 0

    # ── enter trade? ───────────────────────────────────────────────
    if in_pos == 0:
        qty = int(equity // price)
        if qty >= 1 and long_ok:
            in_pos =  qty
            entry_px = price
            sl_px = entry_px * (1 - SL_PCT)
            tp_px = entry_px * (1 + TP_PCT)
            equity -= calculate_zerodha_fees(entry_px, entry_px, qty)
            print(f"🟢 {ts_time} BUY  {qty} @ {entry_px:.2f}  "
                  f"SL:{sl_px:.2f}  TP:{tp_px:.2f}", flush=True)

            # 📜  log OPEN
            logger.log_open(
                OpenTrade(
                    timestamp=bar.name,
                    symbol=SYMBOL,
                    side="BUY",
                    qty=qty,
                    price=entry_px,
                    sl=sl_px,
                    tp=tp_px,
                )
            )

        elif qty >= 1 and short_ok:
            in_pos = -qty
            entry_px = price
            sl_px = entry_px * (1 + SL_PCT)
            tp_px = entry_px * (1 - TP_PCT)
            equity -= calculate_zerodha_fees(entry_px, entry_px, qty)
            print(f"🔴 {ts_time} SELL {qty} @ {entry_px:.2f}  "
                  f"SL:{sl_px:.2f}  TP:{tp_px:.2f}", flush=True)

            logger.log_open(
                OpenTrade(
                    timestamp=bar.name,
                    symbol=SYMBOL,
                    side="SELL",
                    qty=qty,
                    price=entry_px,
                    sl=sl_px,
                    tp=tp_px,
                )
            )

# ─── main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.stdout and not sys.stdout.isatty():
        print("⚠️  Tip: run with  `python -u run_paper_ws.py` for unbuffered logs.",
              flush=True)

    ws = KiteTicker(cfg.api_key, cfg.access_token)
    ws.on_connect = on_connect
    ws.on_ticks   = on_ticks

    print("🟢 Starting live paper-trade loop (Ctrl-C to stop) …", flush=True)
    try:
        ws.connect(threaded=False)
    except KeyboardInterrupt:
        print(f"\n🔴 Stopped. Final equity ₹{equity:.2f}", flush=True)
        ws.close()
