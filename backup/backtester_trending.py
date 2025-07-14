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
    regime = None
    trailing_stop = None
    trailing_active = False

    # Determine frequency in minutes
    freq_minutes = None
    if df.index.freq:
        freq_minutes = int(df.index.freq.delta.total_seconds() / 60)
    elif len(df.index) > 1:
        freq_minutes = int(df.index.to_series().diff().dropna().mode()[0].total_seconds() / 60)
    else:
        freq_minutes = 1

    for idx, (ts, row) in enumerate(df.iterrows()):
        price = row["close"]

        # ✅ Robustly parse ts if not already a Timestamp
        if not isinstance(ts, pd.Timestamp):
            try:
                ts = pd.to_datetime(ts)
            except Exception as e:
                print(f"[ERROR] Failed to parse ts at index {idx}: {ts} ({type(ts)})")
                continue

        # --- Get previous EMA values for crossover checks ---
        if idx == 0:
            continue  # can't check cross on first bar
        prev_row = df.iloc[idx - 1]
        prev_ema8 = prev_row["ema_8"]
        prev_ema21 = prev_row["ema_21"]

        # === ENTRY LOGIC: Pure EMA crossover ===
        if position == 0:


            # LONG entry: ema_8 crosses above ema_21
            if prev_ema8 <= prev_ema21 and row["ema_8"] > row["ema_21"] and row["supertrend_dir"] == 1 and price > row["supertrend"]:
                position = +1
                entry_price = price
                entry_index = ts
                regime = "EMA_XOVER"
                trailing_stop = None
                trailing_active = False
                if debug:
                    print(f"{ts} ENTRY LONG @ {entry_price:.2f}")

            # SHORT entry: ema_8 crosses below ema_21
            elif prev_ema8 >= prev_ema21 and row["ema_8"] < row["ema_21"] and row["supertrend_dir"] == -1 and price < row["supertrend"]:
                position = -1
                entry_price = price
                entry_index = ts
                regime = "EMA_XOVER"
                trailing_stop = None
                trailing_active = False
                if debug:
                    print(f"{ts} ENTRY SHORT @ {entry_price:.2f}")

        # === EXIT LOGIC: Exit on crossover in opposite direction or ATR stop ===
        elif position != 0:
            exit_reason = None
            exit_price = None

            # --- ATR-based STOP LOSS ---
            atr = row["atr"]
            if pd.isna(atr):
                continue
            stop_loss = entry_price - atr * 1.5 if position > 0 else entry_price + atr * 1.5
            if position > 0 and price <= stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = price
            elif position < 0 and price >= stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = price

            # --- Basic EMA cross-out (opposite cross) ---
            if exit_reason is None:
                if position > 0 and prev_ema8 >= prev_ema21 and row["ema_8"] < row["ema_21"]:
                    exit_reason = "CROSS_EXIT"
                    exit_price = price
                elif position < 0 and prev_ema8 <= prev_ema21 and row["ema_8"] > row["ema_21"]:
                    exit_reason = "CROSS_EXIT"
                    exit_price = price

            # --- Finalize trade if exit triggered ---
            if exit_reason:
                qty = abs(position)
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
                    "regime": regime,
                })

                if debug:
                    print(f"{ts} EXIT {exit_reason} @ {exit_price:.2f} PnL={net_pnl:.2f} EQ={equity:.2f} ({regime})")

                # Reset position and trailing
                position = 0
                entry_price = 0
                entry_index = None
                regime = None
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
            "regime": regime,
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

