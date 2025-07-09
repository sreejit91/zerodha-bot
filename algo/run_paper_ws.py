from __future__ import annotations
import datetime as dt, sys, collections, itertools
import pandas as pd
from kiteconnect import KiteTicker

from algo import load_config
from algo import KiteWrapper
from algo import add_indicators
from algo import load_or_train, predict_last, LOOKBACK
from algo import calculate_zerodha_fees

# 🚀  TRADE LOGGER with trade_id support
from logger import TradeLogger, OpenTrade, CloseTrade
logger = TradeLogger()
current_trade_id: int | None = None

# ─── parameters ─────────────────────────────────────────────────────
SYMBOL, TOKEN        = "IDEA", 3677697
INTERVAL             = 180                   # seconds = 3 minutes
WINDOW               = LOOKBACK + 10         # 60 + 10 bars for warm-up
INITIAL_CAPITAL      = 200_000
SL_PCT, TP_PCT       = 0.0030, 0.0065
TRAIL_PCT            = 0.0020
HOLD_MAX             = 10                    # 10 bars × 3 min = 30 min
UPPER, LOWER         = 0.486, 0.596

WINDOWS = [(dt.time(9, 15), dt.time(15, 25))]
is_mkt_hours = lambda t: any(a <= t <= b for a, b in WINDOWS)
now_ist      = lambda: dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))

# ─── bootstrap ───────────────────────────────────────────────────────
print(f"🟢 Boot {now_ist():%Y-%m-%d %H:%M:%S} IST", flush=True)
cfg  = load_config()
wrap = KiteWrapper(cfg)

print("⏳ Loading model / warm history …", flush=True)
hist  = wrap.history(days=200, interval="3minute", tradingsymbol=SYMBOL)
df    = add_indicators(hist).ffill()
model = load_or_train(df.iloc[:-20], retrain=False)
print("✅ Model ready", flush=True)

# ─── seed rolling cache with fallback ────────────────────────────────
bars: collections.deque[pd.Series] = collections.deque(maxlen=WINDOW)
seed = None
for days in (1, 2, 3):
    try:
        tmp = wrap.history(days=days, interval="3minute", tradingsymbol=SYMBOL)
    except RuntimeError:
        continue
    if len(tmp) >= WINDOW:
        seed = tmp.tail(WINDOW)
        break

if seed is None or len(seed) < WINDOW:
    raise RuntimeError(f"Failed to fetch ≥{WINDOW} bars for warm-up (got {len(seed) if seed else 0})")

for ts, row in seed.iterrows():
    s = row.copy()
    s.name          = ts.replace(second=0, microsecond=0)
    s["_processed"] = True
    s["_last"]      = s["close"]
    bars.append(s)

print(f"✅ Warm-up completed with {len(bars)} bars; ready to trade.", flush=True)

# ─── trading state ──────────────────────────────────────────────────
equity      = INITIAL_CAPITAL
in_pos      = 0
entry_px    = sl_px = tp_px = 0.0
trade_age   = 0
tick_counter = itertools.count(1)

# ─── websocket handlers ─────────────────────────────────────────────
def on_connect(ws, _):
    print("✅ Connected – subscribing now …", flush=True)
    print(f"   Tokens → [{TOKEN}]", flush=True)
    ws.subscribe([TOKEN])
    ws.set_mode(ws.MODE_QUOTE, [TOKEN])

def on_ticks(ws, ticks):
    global equity, in_pos, entry_px, sl_px, tp_px, trade_age, current_trade_id

    # heartbeat
    for _ in ticks:
        c = next(tick_counter)
        if c % 100 == 0:
            print(f"📡 heartbeat tick #{c}", flush=True)

    # 1) tick-level SL/TP exit
    for t in ticks:
        ltp = t["last_price"]
        if in_pos and current_trade_id is not None:
            if in_pos > 0:
                # long: stop if LTP <= sl_px, target if LTP >= tp_px
                if ltp <= sl_px:
                    reason = "SL"
                elif ltp >= tp_px:
                    reason = "TP"
                else:
                    reason = None
            else:
                # short: stop if LTP >= sl_px, target if LTP <= tp_px
                if ltp >= sl_px:
                    reason = "SL"
                elif ltp <= tp_px:
                    reason = "TP"
                else:
                    reason = None

            if reason:
                # force exit price to threshold
                exit_price = sl_px if reason == "SL" else tp_px
                fee_total  = calculate_zerodha_fees(entry_px, exit_price, abs(in_pos), debug=True)
                pnl        = (exit_price - entry_px) * in_pos
                net        = pnl - fee_total
                equity    += net

                now_t = now_ist().time()
                print(
                    f"💔 {now_t:%H:%M:%S} EXIT {reason} @ {exit_price:.2f} "
                    f"gross={pnl:+.2f} fees={fee_total:.2f} net={net:+.2f}\n",
                    flush=True
                )
                logger.log_close(
                    CloseTrade(
                        timestamp   = now_ist(),
                        symbol      = SYMBOL,
                        qty         = abs(in_pos),
                        price       = exit_price,
                        reason      = reason,
                        fee_detail  = {"total": fee_total},
                        gross_pnl   = pnl,
                        net_pnl     = net,
                    ),
                    trade_id=current_trade_id
                )
                in_pos, trade_age, current_trade_id = 0, 0, None
                return

    # 2) build/update bars
    for t in ticks:
        ltp = t["last_price"]
        ts  = (t.get("last_trade_time") or dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))).astimezone()
        key = ts.replace(second=0, microsecond=0)

        if not bars or bars[-1].name != key:
            if bars and "_last" in bars[-1]:
                prev = bars[-1]
                prev["close"] = prev.pop("_last")
                prev["high"]  = max(prev["high"], prev["close"])
                prev["low"]   = min(prev["low"],  prev["close"])
            bars.append(pd.Series({
                "open": ltp, "high": ltp, "low": ltp,
                "close": ltp, "volume": 1, "_last": ltp
            }, name=key))
        else:
            b = bars[-1]
            b["_last"] = ltp
            b["high"]  = max(b["high"], ltp)
            b["low"]   = min(b["low"], ltp)
            b["volume"] += 1

    # 3) process closed bar
    if len(bars) < 2 or bars[-2].get("_processed"):
        return
    bar = bars[-2]
    bar["_processed"] = True

    if in_pos and (bar.name.minute % (INTERVAL // 60) == 0):
        trade_age += 1

    price, ts_time = bar["close"], bar.name.time()

    # 4) EOD square-off
    if in_pos and current_trade_id is not None and ts_time >= dt.time(15, 27):
        fee_total = calculate_zerodha_fees(entry_px, price, abs(in_pos), debug=True)
        pnl       = (price - entry_px) * in_pos
        net       = pnl - fee_total
        equity   += net

        print(
            f"💔 {ts_time} EXIT EOD @ {price:.2f} "
            f"gross={pnl:+.2f} fees={fee_total:.2f} net={net:+.2f}\n",
            flush=True
        )
        logger.log_close(
            CloseTrade(
                timestamp   = bar.name,
                symbol      = SYMBOL,
                qty         = abs(in_pos),
                price       = price,
                reason      = "EOD",
                fee_detail  = {"total": fee_total},
                gross_pnl   = pnl,
                net_pnl     = net,
            ),
            trade_id=current_trade_id
        )
        in_pos, trade_age, current_trade_id = 0, 0, None
        return

    # 5) features & signals
    df_win    = pd.DataFrame([b.drop(["_processed","_last"], errors="ignore") for b in bars])
    df_feat   = add_indicators(df_win).ffill()
    df_closed = df_feat.iloc[:-1]
    if len(df_closed) < LOOKBACK:
        return

    prob     = predict_last(df_closed, model)
    ema_fast = df_closed["ema_8"].iat[-1]
    ema_slow = df_closed["ema_21"].iat[-1]
    long_ok  = prob >= UPPER and ema_fast > ema_slow and is_mkt_hours(ts_time)
    short_ok = prob <= LOWER and ema_fast < ema_slow and is_mkt_hours(ts_time)

    print(
        f"{ts_time} prob={prob:.2f} ema8={ema_fast:.2f} ema21={ema_slow:.2f} "
        f"long_ok={long_ok} short_ok={short_ok}",
        flush=True
    )

    # 6) time-based exit
    if in_pos and current_trade_id is not None and trade_age >= HOLD_MAX:
        fee_total = calculate_zerodha_fees(entry_px, price, abs(in_pos), debug=True)
        pnl       = (price - entry_px) * in_pos
        net       = pnl - fee_total
        equity   += net

        print(
            f"💔 {ts_time} EXIT TIME @ {price:.2f} "
            f"gross={pnl:+.2f} fees={fee_total:.2f} net={net:+.2f}\n",
            flush=True
        )
        logger.log_close(
            CloseTrade(
                timestamp   = bar.name,
                symbol      = SYMBOL,
                qty         = abs(in_pos),
                price       = price,
                reason      = "TIME",
                fee_detail  = {"total": fee_total},
                gross_pnl   = pnl,
                net_pnl     = net,
            ),
            trade_id=current_trade_id
        )
        in_pos, trade_age, current_trade_id = 0, 0, None

    # 7) entry logic
    if in_pos == 0:
        qty = int(equity // price)
        if qty >= 1 and long_ok:
            in_pos, entry_px = qty, price
            sl_px, tp_px     = entry_px * (1 - SL_PCT), entry_px * (1 + TP_PCT)
            equity          -= calculate_zerodha_fees(entry_px, entry_px, qty)
            print(f"🟢 {ts_time} BUY {qty} @ {entry_px:.2f} SL:{sl_px:.2f} TP:{tp_px:.2f}", flush=True)
            current_trade_id = logger.log_open(OpenTrade(
                timestamp = bar.name,
                symbol    = SYMBOL,
                side      = "BUY",
                qty       = qty,
                price     = entry_px,
                sl        = sl_px,
                tp        = tp_px,
            ))
        elif qty >= 1 and short_ok:
            in_pos, entry_px = -qty, price
            sl_px, tp_px     = entry_px * (1 + SL_PCT), entry_px * (1 - TP_PCT)
            equity          -= calculate_zerodha_fees(entry_px, entry_px, qty)
            print(f"🔴 {ts_time} SELL {qty} @ {entry_px:.2f} SL:{sl_px:.2f} TP:{tp_px:.2f}", flush=True)
            current_trade_id = logger.log_open(OpenTrade(
                timestamp = bar.name,
                symbol    = SYMBOL,
                side      = "SELL",
                qty       = qty,
                price     = entry_px,
                sl        = sl_px,
                tp        = tp_px,
            ))

if __name__ == "__main__":
    if sys.stdout and not sys.stdout.isatty():
        print("⚠️ Tip: run with `python -u run_paper_ws.py` for unbuffered logs.", flush=True)
    ws = KiteTicker(cfg.api_key, cfg.access_token)
    ws.on_connect = on_connect
    ws.on_ticks   = on_ticks
    print("🟢 Starting live paper-trade loop (Ctrl-C to stop) …", flush=True)
    try:
        ws.connect(threaded=False)
    except KeyboardInterrupt:
        print(f"\n🔴 Stopped. Final equity ₹{equity:.2f}", flush=True)
        ws.close()
