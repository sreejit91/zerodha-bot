import pandas as pd
import numpy as np
from typing import Tuple, Callable, Optional

# Zerodha fees calculation — as per your formula
def calculate_zerodha_fees(entry_price: float, exit_price: float, quantity: int, debug: bool = False) -> float:
    turnover = (entry_price + exit_price) * quantity
    per_leg_brokerage = np.minimum(turnover * 0.0003,20)
    brokerage = per_leg_brokerage * 2
    txn_charges = turnover * 0.0000325
    sebi_charges = turnover * 0.0000005
    stt_ctt = exit_price * quantity * 0.00025
    gst = (brokerage + txn_charges) * 0.18
    stamp_duty = entry_price * quantity * 0.00003
    total_fees = brokerage + txn_charges + sebi_charges + stt_ctt + gst + stamp_duty
    if debug:
        print(f"[fees] total={total_fees:.2f} on qty={quantity} from {entry_price:.2f} → {exit_price:.2f}")
    return total_fees

def backtest_ML_switchable(
    df: pd.DataFrame,
    model,
    predict_fn: Callable,             # (window_df, model) → prob
    entry_rule_fn: Callable,          # (row, prob, state) → {'entry': 1, 'side': 'BUY'/'SELL'} or None
    exit_rule_fn: Optional[Callable], # (row, prob, state) → (reason, price) or None
    capital: float,
    contract_size: int,
    lookback: int,
    sl_pct: float = 0.01,
    tp_pct: float = 0.02,
    debug: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    df = df.copy().sort_index()
    trades = []
    equity = capital
    position = 0
    entry_price = 0
    entry_index = None
    entry_prob = None
    regime = None

    # --- ML probabilities (precompute for speed) ---
    probs = [np.nan] * len(df)
    for idx in range(lookback, len(df)):
        window = df.iloc[idx - lookback:idx]
        prob = predict_fn(window, model)
        probs[idx] = prob
    df['ml_prob'] = probs
    print("Non-NaN ml_prob:", np.isfinite(df['ml_prob']).sum(), "out of", len(df))
    print(df[['ml_prob']].describe())

    for idx, (ts, row) in enumerate(df.iterrows()):
        if idx < lookback or not np.isfinite(row['ml_prob']):
            continue
        prob = row['ml_prob']
        state = dict(position=position, equity=equity, entry_price=entry_price, entry_index=entry_index, prob=prob)

        # ENTRY LOGIC (configurable)
        if position == 0:
            entry_signal = entry_rule_fn(row, prob, state)
            if entry_signal:
                side = entry_signal.get('side')
                position = contract_size if side == 'BUY' else -contract_size
                entry_price = row['close']
                entry_index = ts
                entry_prob = prob
                regime = entry_signal.get('regime', None)
                if debug:
                    print(f"{ts} ENTRY {side} @ {entry_price:.2f} prob={prob:.3f}")
                continue

        # EXIT LOGIC (configurable or fallback SL/TP/EOD)
        if position != 0:
            exit_signal = None
            if exit_rule_fn:
                exit_signal = exit_rule_fn(row, prob, dict(position=position, equity=equity, entry_price=entry_price,
                                                           entry_index=entry_index, prob=prob))
            exit_reason, exit_price = None, None
            if exit_signal:
                exit_reason, exit_price = exit_signal
            else:
                # fallback: basic SL/TP/EOD
                if position > 0 and (row['close'] <= entry_price * (1 - sl_pct)):
                    exit_reason, exit_price = 'SL', entry_price * (1 - sl_pct)
                elif position > 0 and (row['close'] >= entry_price * (1 + tp_pct)):
                    exit_reason, exit_price = 'TP', entry_price * (1 + tp_pct)
                elif position < 0 and (row['close'] >= entry_price * (1 + sl_pct)):
                    exit_reason, exit_price = 'SL', entry_price * (1 + sl_pct)
                elif position < 0 and (row['close'] <= entry_price * (1 - tp_pct)):
                    exit_reason, exit_price = 'TP', entry_price * (1 - tp_pct)
                # EOD
                elif idx == len(df) - 1 or (df.index[idx + 1].date() != ts.date()):
                    exit_reason, exit_price = 'EOD', row['close']
            if exit_reason:
                qty = abs(position)
                gross_pnl = position * (exit_price - entry_price)
                fees = calculate_zerodha_fees(entry_price, exit_price, qty, debug)
                net_pnl = gross_pnl - fees
                equity += net_pnl
                trades.append({
                    "entry_ts": entry_index, "exit_ts": ts,
                    "side": "BUY" if position > 0 else "SELL",
                    "entry_price": entry_price, "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "fees": fees, "qty": qty, "gross_pnl": gross_pnl,
                    "pnl": net_pnl, "equity": equity, "regime": regime, "ml_prob": entry_prob
                })
                position, entry_price, entry_index, entry_prob, regime = 0, 0, None, None, None

    trades_df = pd.DataFrame(trades)
    metrics = {
        "Trades": len(trades_df),
        "WinRate": (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0.0,
        "GrossPnL": trades_df["gross_pnl"].sum() if not trades_df.empty else 0.0,
        "Fees": trades_df["fees"].sum() if not trades_df.empty else 0.0,
        "NetPnL": trades_df["pnl"].sum() if not trades_df.empty else 0.0,
        "EquityFinal": equity,
    }
    return trades_df, metrics

# EXAMPLE ENTRY RULE FUNCTION (hybrid ML + indicator)
def hybrid_entry(row, prob, state):
    # Example: Only allow entry if ML probability > 0.7 and ema_15 > ema_50
    if prob > 0.51 and row['ema_8'] > row['ema_21']:
        return {'entry': 1, 'side': 'BUY'}
    if prob < 0.49 and row['ema_8'] < row['ema_21']:
        return {'entry': 1, 'side': 'SELL'}
    return None

# EXAMPLE EXIT RULE FUNCTION (optional, else use fallback)
def default_exit(row, prob, state):
    # You can customize more logic here
    return None

# --- How to call this function is below ---
