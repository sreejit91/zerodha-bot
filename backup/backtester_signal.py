import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import time


# Zerodha fees constants
BROKERAGE_RATE   = 0.0003
MAX_BROKERAGE    = 20.0
TXN_CHARGES_RATE = 0.0000325
SEBI_CHARGES_RATE= 0.0000005
GST_RATE         = 0.18
STAMP_DUTY_RATE  = 0.00003
STT_RATE         = 0.00025

def calculate_zerodha_fees(entry_price: float, exit_price: float, quantity: int, debug: bool = False) -> float:
    turnover = (entry_price + exit_price) * quantity
    per_leg_brokerage = min(turnover * 0.5 * BROKERAGE_RATE, MAX_BROKERAGE)
    brokerage = per_leg_brokerage * 2
    txn_charges = turnover * TXN_CHARGES_RATE
    sebi_charges = turnover * SEBI_CHARGES_RATE
    stt_ctt = exit_price * quantity * STT_RATE
    gst = (brokerage + txn_charges) * GST_RATE
    stamp_duty = entry_price * quantity * STAMP_DUTY_RATE
    total_fees = brokerage + txn_charges + sebi_charges + stt_ctt + gst + stamp_duty
    if debug:
        print(f"[fees] total={total_fees:.2f} on qty={quantity} from {entry_price:.2f} → {exit_price:.2f}")
    return total_fees

def backtest(
    df: pd.DataFrame,
    capital: float,
    contract_size: int,
    sl_pct: float,
    tp_pct: float,
    trail_pct: float,
    hold_max: int,
    slippage_pct: float = 0.0,
    fill_rate: float = 1.0,
    hold_min: int = 0,
    debug: bool = True
) -> Tuple[pd.DataFrame, dict]:

    df = df.copy().sort_index()
    trades = []
    position = 0
    equity = capital
    entry_price = 0
    entry_index = None
    total_bars = 0
    signal_count = 0

    freq_minutes = None
    if df.index.freq:
        freq_minutes = int(df.index.freq.delta.total_seconds() / 60)
    elif len(df.index) > 1:
        freq_minutes = int(df.index.to_series().diff().dropna().mode()[0].total_seconds() / 60)
    trailing_stop = None
    trailing_active = False

    for idx, (ts, row) in enumerate(df.iterrows()):
        price = row["close"]

        # ✅ Robustly parse ts if not already a Timestamp
        if not isinstance(ts, pd.Timestamp):
            try:
                ts = pd.to_datetime(ts)
            except Exception as e:
                print(f"[ERROR] Failed to parse ts at index {idx}: {ts} ({type(ts)})")
                continue

        # === ENTRY LOGIC ===
        if position == 0:
            # --- Time-of-day binning as float hour ---
            entry_hour = ts.hour + ts.minute / 60
            good_bins = {10, 10.5, 11.5, 12.5, 14.5}  # 10:00, 10:30, 11:30, 12:30, 14:30

            if entry_hour not in good_bins:
                continue

            # --- ENTRY QUALITY FILTERS ---
            if (pd.isna(row["atr"]) or pd.isna(row["adx"]) or pd.isna(row["high_N"]) or pd.isna(row["low_N"])
                    or pd.isna(row["vol_avg20"]) or pd.isna(row.get("atr_median20", None))):
                continue

            if (row["adx"] < 20):
                continue
            if row["atr"] < 0.2:
                continue
            if row["volume"] < row["vol_avg20"] * 1.1:
                continue

            # --- LONG breakout entry ---
            if (row["supertrend_dir"] == 1 and price > row["vwap"] and row["ema_9"] > row["ema_21"]
                    and price > row["high_N"]):
                position = +1
                entry_price = price * (1 + slippage_pct)
                entry_index = ts
                trailing_stop = None
                trailing_active = False
                if debug:
                    print(f"{ts} ENTRY LONG @ {entry_price:.2f}")

            # --- SHORT breakout entry ---
            elif (row["supertrend_dir"] == -1 and price < row["vwap"] and row["ema_9"] < row["ema_21"]
                  and price < row["low_N"]):
                position = -1
                entry_price = price * (1 - slippage_pct)
                entry_index = ts
                trailing_stop = None
                trailing_active = False
                if debug:
                    print(f"{ts} ENTRY SHORT @ {entry_price:.2f}")


        # === EXIT LOGIC ===
        elif position != 0:
            atr = row["atr"]
            if pd.isna(atr):
                continue

            qty = abs(position)
            held_min = (ts - entry_index).total_seconds() // 60 if entry_index else 0
            if hold_min and held_min < hold_min * freq_minutes:
                continue

            exit_reason = None
            exit_price = None

            # --- ATR-based STOP LOSS ---
            stop_loss = entry_price - atr * 1.0 if position > 0 else entry_price + atr * 1.0
            if position > 0 and price <= stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = price
            elif position < 0 and price >= stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = price

            # --- Breakeven stop activation (if not already trailing) ---
            move_from_entry = price - entry_price if position > 0 else entry_price - price
            favorable_threshold = atr * 1.0
            trailing_buffer = atr * 0.5

            if exit_reason is None and not trailing_active:
                if (position > 0 and price >= entry_price + favorable_threshold) or (
                        position < 0 and price <= entry_price - favorable_threshold):
                    trailing_active = True
                    trailing_stop = entry_price  # Breakeven trailing

            # --- Trailing stop activation (if not already trailing or breakeven trailing) ---
            if exit_reason is None and not trailing_active:
                if move_from_entry >= favorable_threshold:
                    trailing_active = True
                    trailing_stop = price - trailing_buffer if position > 0 else price + trailing_buffer

            # --- Trailing stop / Breakeven stop exit check ---
            if trailing_active:
                # Breakeven trailing stop: only triggers at entry price
                if trailing_stop == entry_price:
                    if (position > 0 and price <= trailing_stop) or (position < 0 and price >= trailing_stop):
                        exit_reason = "BREAKEVEN"
                        exit_price = trailing_stop
                # Normal trailing stop
                elif exit_reason is None:
                    if position > 0:
                        trailing_stop = max(trailing_stop, price - trailing_buffer)
                        if price <= trailing_stop:
                            exit_reason = "TRAIL"
                            exit_price = trailing_stop
                    else:
                        trailing_stop = min(trailing_stop, price + trailing_buffer)
                        if price >= trailing_stop:
                            exit_reason = "TRAIL"
                            exit_price = trailing_stop

            # --- Trend-flip exit ---
            if exit_reason is None:
                if position > 0 and row["supertrend_dir"] == -1:
                    exit_reason = "TREND_FLIP"
                    exit_price = price
                elif position < 0 and row["supertrend_dir"] == 1:
                    exit_reason = "TREND_FLIP"
                    exit_price = price

            # --- Take Profit & Time-based exit ---
            target = entry_price + atr * 1.5 if position > 0 else entry_price - atr * 1.5
            if exit_reason is None:
                if (position > 0 and price >= target) or (position < 0 and price <= target):
                    exit_reason = "TP"
                    exit_price = target
                elif hold_max and held_min >= hold_max * freq_minutes:
                    exit_reason = "TIME"
                    exit_price = price

            # --- Finalize trade if an exit condition was triggered ---
            if exit_reason:
                gross_pnl = position * (exit_price - entry_price) * fill_rate
                fees = calculate_zerodha_fees(entry_price, exit_price, qty, debug)
                net_pnl = gross_pnl - fees
                equity += net_pnl

                trades.append({
                    "entry_ts": entry_index,
                    "exit_ts": ts,
                    "side": "BUY" if position > 0 else "SELL",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "fees": fees,
                    "qty": qty,
                    "pnl": net_pnl,
                    "equity": equity,
                })

                if debug:
                    print(f"{ts} EXIT {exit_reason} @ {exit_price:.2f} PnL={net_pnl:.2f} EQ={equity:.2f}")

                # Reset position and trailing
                position = 0
                entry_price = 0
                entry_index = None
                trailing_stop = None
                trailing_active = False

    # Final exit if still in position
    if position != 0:
        ts = df.index[-1]
        price = df.iloc[-1]["close"]
        gross_pnl = position * (price - entry_price) * fill_rate
        fees = calculate_zerodha_fees(entry_price, price, abs(position), debug)
        net_pnl = gross_pnl - fees
        equity += net_pnl
        trades.append({
            "entry_ts": entry_index,
            "exit_ts": ts,
            "side": "BUY" if position > 0 else "SELL",
            "entry_price": entry_price,
            "exit_price": price,
            "exit_reason": "EOD",
            "fees": fees,
            "qty": abs(position),
            "pnl": net_pnl,
            "equity": equity,
        })
        if debug:
            print(f"{ts} EOD EXIT @ {price:.2f} PnL={net_pnl:.2f} EQ={equity:.2f}")

    trades_df = pd.DataFrame(trades).set_index("entry_ts") if trades else pd.DataFrame()

    metrics = {
        "Trades": len(trades_df),
        "WinRate": (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0.0,
        "GrossPnL": (trades_df["pnl"] + trades_df["fees"]).sum() if not trades_df.empty else 0.0,
        "Fees": trades_df["fees"].sum() if not trades_df.empty else 0.0,
        "NetPnL": trades_df["pnl"].sum() if not trades_df.empty else 0.0,
        "EquityFinal": equity,
    }

    return trades_df, metrics
