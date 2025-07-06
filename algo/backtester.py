import pandas as pd
import numpy as np
from typing import Tuple, Callable, Optional
from algo.model import LOOKBACK

# ─── Zerodha fee constants ──────────────────────────────────────────
BROKERAGE_RATE   = 0.0003     # 0.03 % per leg, max ₹20
MAX_BROKERAGE    = 20.0
TXN_CHARGES_RATE = 0.0000325  # 0.00325 %
SEBI_CHARGES_RATE= 0.0000005  # 0.00005 %
GST_RATE         = 0.18       # 18 % on (brokerage + txn)
STAMP_DUTY_RATE  = 0.00003    # 0.003 %  buy leg only
STT_RATE         = 0.00025    # 0.025 % sell leg only

# ─── Fee function (final) ───────────────────────────────────────────
def calculate_zerodha_fees(entry_price: float,
                           exit_price : float,
                           quantity   : int,
                           debug: bool = False) -> float:
    """
    Zerodha round-trip charges for intraday equities.
    Returns *total fees* (float).  If debug=True it prints a full breakdown.
    """
    turnover           = (entry_price + exit_price) * quantity
    per_leg_brokerage  = min(turnover * 0.5 * BROKERAGE_RATE, MAX_BROKERAGE)
    brokerage          = per_leg_brokerage * 2                          # both legs
    txn_charges        = turnover * TXN_CHARGES_RATE
    sebi_charges       = turnover * SEBI_CHARGES_RATE
    stt_ctt            = exit_price * quantity * STT_RATE               # sell leg only
    gst                = (brokerage + txn_charges) * GST_RATE           # STT not GST-able
    stamp_duty         = entry_price * quantity * STAMP_DUTY_RATE       # buy leg only

    total_fees = brokerage + txn_charges + sebi_charges + stt_ctt + gst + stamp_duty

    if debug:
        print(f"[fees] entry={entry_price:.2f}, exit={exit_price:.2f}, qty={quantity}")
        print(f"[fees]  > turnover={turnover:.2f}, brokerage(total)={brokerage:.2f} "
              f"(per leg={per_leg_brokerage:.2f}), txn={txn_charges:.4f}, "
              f"sebi={sebi_charges:.4f}, stt={stt_ctt:.4f}, gst={gst:.4f}, "
              f"stamp={stamp_duty:.4f}")
        print(f"[fees]  >> total fees={total_fees:.4f}\n")

    return total_fees


def backtest(
    df_raw: pd.DataFrame,
    model,
    capital: float,
    contract_size: int,
    sl_pct: float,
    tp_pct: float,
    trail_pct: float,
    hold_max: int,
    upper: float,
    lower: float,
    predict_fn: Callable,
    slippage_pct: float = 0.0,
    fill_rate: float = 1.0,
    skip_indicator: Optional[Callable] = None,
    debug: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run backtest recording entries/exits, PnL including Zerodha fees, exit reasons,
    guard initial lookback, and print probability distribution for diagnostics.
    Uses full equity to size each entry (qty = int(equity // price)).
    Returns completed trades DataFrame and performance metrics.
    """
    df = df_raw.copy().sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    if not df.index.is_monotonic_increasing and debug:
        print("[backtest] WARNING: index still not monotonic!")
    if debug:
        print(f"[backtest] raw df shape: {df.shape}")
        print(f"[backtest] columns: {list(df.columns)}")
        print(f"[backtest] index type: {type(df.index)}, freq: {df.index.freq}")
        print(df.head())
        if df.empty:
            print("[backtest] WARNING: DataFrame is empty. No bars to process.")

    equity = capital
    position = 0        # signed quantity
    entry_price = 0.0
    entry_index = None
    trades = []
    signal_count = 0
    total_bars = 0
    skip_count = 0
    probs = []

    def record_trade(side, entry_ts, exit_ts, entry_px, exit_px, gross_pnl, fees, qty, net_pnl, equity_after, reason):
        trades.append({
            'entry_ts': entry_ts,
            'exit_ts': exit_ts,
            'side': side,
            'entry_price': entry_px,
            'exit_price': exit_px,
            'exit_reason': reason,
            'fees': fees,
            'qty': qty,
            'pnl': net_pnl,
            'equity': equity_after,
        })

    # infer bar frequency
    freq_minutes = None
    if df.index.freq:
        freq_minutes = int(df.index.freq.delta.total_seconds() / 60)
    elif len(df.index) > 1:
        diffs = df.index.to_series().diff().dropna()
        if not diffs.empty:
            freq_minutes = int(diffs.mode()[0].total_seconds() / 60)

    if debug:
        print(f"[backtest] thresholds -> upper={upper}, lower={lower}")
        print(f"[backtest] required lookback rows: {LOOKBACK}")

    for idx, ts in enumerate(df.index):
        total_bars += 1
        # guard for lookback
        if idx + 1 < LOOKBACK:
            continue

        # ── end-of-day square-off ───────────────────────────────
        if idx > 0:
            prev_ts = df.index[idx-1]
            if prev_ts.date() != ts.date() and position != 0:
                exit_price = df.loc[prev_ts, 'close']
                qty        = abs(position)
                gross_pnl  = position * (exit_price - entry_price) * fill_rate
                fees       = calculate_zerodha_fees(entry_price, exit_price, qty, debug)
                net_pnl    = gross_pnl - fees
                equity    += net_pnl
                record_trade(
                    side        = 'BUY' if position>0 else 'SELL',
                    entry_ts    = entry_index,
                    exit_ts     = prev_ts,
                    entry_px    = entry_price,
                    exit_px     = exit_price,
                    gross_pnl   = gross_pnl,
                    fees        = fees,
                    qty         = qty,
                    net_pnl     = net_pnl,
                    equity_after= equity,
                    reason      = 'EOD'
                )
                position    = 0
                entry_price = 0.0
                entry_index = None
                continue

        window = df.iloc[idx + 1 - LOOKBACK : idx + 1]
        row = df.iloc[idx]
        if skip_indicator and skip_indicator(row):
            skip_count += 1
            continue

        prob = predict_fn(window, model)
        probs.append(prob)
        price = row.get('close', None)
        if price is None:
            continue

        # ENTRY
        if position == 0:
            if prob >= upper or prob <= lower:
                signal_count += 1
                qty = int(equity // price)
                if qty < 1:
                    if debug:
                        print(f"{ts} SKIP ENTRY, insufficient equity for one share @ {price:.2f}")
                    continue
                entry_price = price * (1 + slippage_pct) if prob >= upper else price * (1 - slippage_pct)
                position = qty if prob >= upper else -qty
                entry_index = ts
                side_str = 'BUY' if position > 0 else 'SELL'
                if debug:
                    print(f"{ts} SIGNAL {side_str} @ prob={prob:.3f}")
                    print(f"{ts} ENTRY {side_str} @ qty={qty} price={entry_price:.2f}\n")
                continue

        # EXIT conditions
        if position != 0:
            qty = abs(position)
            stop = entry_price * (1 - sl_pct) if position > 0 else entry_price * (1 + sl_pct)
            target = entry_price * (1 + tp_pct) if position > 0 else entry_price * (1 - tp_pct)
            exit_reason = None
            exit_price = None

            # stop-loss
            if (position > 0 and price <= stop) or (position < 0 and price >= stop):
                exit_reason, exit_price = 'SL', stop
            # take-profit
            elif (position > 0 and price >= target) or (position < 0 and price <= target):
                exit_reason, exit_price = 'TP', target
            # time-based
            elif hold_max and freq_minutes:
                held_min = (ts - entry_index).total_seconds() // 60
                if held_min >= hold_max * freq_minutes:
                    exit_reason, exit_price = 'TIME', price

            if exit_reason:
                gross_pnl = position * (exit_price - entry_price) * fill_rate
                fees = calculate_zerodha_fees(entry_price, exit_price, qty, debug)
                net_pnl = gross_pnl - fees
                equity += net_pnl
                side_str = 'BUY' if position > 0 else 'SELL'
                if debug:
                    print(f"{ts} EXIT {exit_reason} {side_str} @ price={exit_price:.2f} qty={qty} | "
                          f"Gross={gross_pnl:.2f} Fees={fees:.2f} Net={net_pnl:.2f}\n")
                record_trade(
                    side_str, entry_index, ts, entry_price,
                    exit_price, gross_pnl, fees, qty, net_pnl, equity, exit_reason
                )
                position = 0
                entry_price = 0.0
                entry_index = None

    # final end-of-data square-off
    if position != 0:
        last_ts    = df.index[-1]
        exit_price = df.iloc[-1]['close']
        qty        = abs(position)
        gross_pnl  = position * (exit_price - entry_price) * fill_rate
        fees       = calculate_zerodha_fees(entry_price, exit_price, qty, debug)
        net_pnl    = gross_pnl - fees
        equity    += net_pnl
        record_trade(
            'BUY' if position>0 else 'SELL', entry_index, last_ts,
            entry_price, exit_price, gross_pnl, fees, qty,
            net_pnl, equity, 'EOD'
        )
        position = 0

    # diagnostics
    if debug:
        print(f"[backtest] processed bars: {total_bars}")
        print(f"[backtest] skipped bars: {skip_count}")
        if probs:
            print(f"[backtest] prob stats: count={len(probs)}, min={np.min(probs):.3f}, max={np.max(probs):.3f}, mean={np.mean(probs):.3f}")
            print(f"[backtest] signals fired: {signal_count}\n")
        else:
            print("[backtest] no probability values collected; check predict_fn and data.")

    # assemble trades DataFrame
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty and 'entry_ts' in trades_df.columns:
        trades_df.set_index('entry_ts', inplace=True)

    if debug:
        print("[backtest] Full trade log:")
        print(trades_df)

    # performance metrics
    metrics = {
        'Signals': signal_count,
        'Trades': int(len(trades_df)),
        'WinRate': float((trades_df['pnl'] > 0).mean()) if not trades_df.empty else 0.0,
        'GrossPnL': float((trades_df['pnl'] + trades_df['fees']).sum()) if not trades_df.empty else 0.0,
        'Fees': float(trades_df['fees'].sum()) if not trades_df.empty else 0.0,
        'NetPnL': float(trades_df['pnl'].sum()) if not trades_df.empty else 0.0,
        'EquityFinal': float(equity),
    }
    return trades_df, metrics
