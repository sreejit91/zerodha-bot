# ── algo/backtester.py ───────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Any, Tuple, Dict

# These two **must** match add_labels() & model.py
HORIZON        = 24         # bars the label looks ahead
THR_ATR_MULT   = 1.0        # ATR‑multiple for both label & TP/SL

# Brokerage / micro‑structure constants
BROKERAGE_CAP  = 20         # ₹ cap per leg at Zerodha
SLIPPAGE_BPS   = 3          # 0.03 % slip each side

# ─────────────────────────────────────────────────────────────────
def calculate_zerodha_fees(entry, exit, qty):
    """
    Zerodha equity‑intraday cost model.
    Vectorised: works on scalars **and** NumPy / pandas arrays.
    """
    entry_arr = np.asarray(entry, dtype="float64")
    exit_arr  = np.asarray(exit,  dtype="float64")

    turnover       = (entry_arr + exit_arr) * qty
    brokerage_buy  = np.minimum(entry_arr * qty * 0.0003, BROKERAGE_CAP)
    brokerage_sell = np.minimum(exit_arr  * qty * 0.0003, BROKERAGE_CAP)
    brokerage      = brokerage_buy + brokerage_sell
    txn_charges    = turnover * 0.0000325
    sebi_charges   = turnover * 0.0000005
    stt_sell       = exit_arr * qty * 0.00025          # sell only
    gst            = (brokerage + txn_charges) * 0.18
    stamp_buy      = entry_arr * qty * 0.00003         # buy only

    total = brokerage + txn_charges + sebi_charges + stt_sell + gst + stamp_buy

    if np.isscalar(entry):
        return float(total)
    if isinstance(entry, pd.Series):
        return pd.Series(total, index=entry.index)
    return total


def apply_slippage(price: float, side: int, bps: float = SLIPPAGE_BPS) -> float:
    """Slip price by ±bps (basis points).  side: +1 long leg, ‑1 short leg."""
    return price * (1 + side * bps / 1e4)


# ─────────────────────────────────────────────────────────────────
def backtest(
    df: pd.DataFrame,
    model: Any,
    predict_fn: Callable[[pd.DataFrame, Any], Tuple[float, float]],
    *,
    capital: float,
    contract_size: int,
    lookback: int = 60,
    tp_atr_mult: float | None = None,
    sl_atr_mult: float | None = None,
    max_hold_bars: int | None = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    ML‑probability‑driven intraday back‑tester (3‑class aware):

      • EV‑based entry using both p_long and p_short
      • Fractional Kelly sizing
      • TP / SL aligned with labeling horizon & threshold
      • Realistic slippage and Zerodha fee model
    """

    if "atr" not in df:
        raise ValueError("`atr` missing – run add_indicators(df) first")

    # -----------------------------------------------------------------
    tp_atr_mult   = tp_atr_mult   or THR_ATR_MULT
    sl_atr_mult   = sl_atr_mult   or THR_ATR_MULT
    max_hold_bars = max_hold_bars or HORIZON
    df = df.copy()

    # 1) Pre‑compute probabilities for speed
    p_long_arr  = np.full(len(df), np.nan)
    p_short_arr = np.full(len(df), np.nan)
    for i in range(lookback, len(df)):
        p_long_arr[i], p_short_arr[i] = predict_fn(df.iloc[i - lookback : i], model)

    df["p_long"]  = p_long_arr
    df["p_short"] = p_short_arr

    # 2) Walk forward
    trades, equity = [], capital
    position = 0
    entry_price = entry_atr = trailing_sl = None
    entry_idx   = None

    for idx, (ts, row) in enumerate(df.iterrows()):
        p_long, p_short = row["p_long"], row["p_short"]
        price, atr      = row["close"], row["atr"]

        # ── ENTRY ───────────────────────────────────────────────────
        if position == 0 and np.isfinite(p_long) and np.isfinite(p_short):
            ev_long  =  p_long  * tp_atr_mult - p_short * sl_atr_mult
            ev_short =  p_short * tp_atr_mult - p_long  * sl_atr_mult
            side     = 1 if ev_long  > 0 else -1 if ev_short > 0 else 0
            edge     = max(ev_long, ev_short, 0)

            if side != 0:
                size_frac = np.clip(edge / (tp_atr_mult + sl_atr_mult), 0.1, 1.0)
                position  = int(contract_size * size_frac) * side

                entry_price = apply_slippage(price, side)
                entry_atr   = atr if np.isfinite(atr) else 0.0
                entry_idx   = idx

                tp_price    = entry_price + side * tp_atr_mult * entry_atr
                sl_price    = entry_price - side * sl_atr_mult * entry_atr
                trailing_sl = sl_price

                if debug:
                    print(f"{ts} ENTRY {'BUY' if side>0 else 'SELL'} "
                          f"pL={p_long:.3f} pS={p_short:.3f} edge={edge:.3f} "
                          f"size={position}")
                continue

        # ── MANAGE / EXIT ───────────────────────────────────────────
        if position != 0:
            side = 1 if position > 0 else -1

            # Trailing SL candidate
            if np.isfinite(atr):
                candidate = price - side * sl_atr_mult * atr
                trailing_sl = max(trailing_sl, candidate) if side > 0 else min(trailing_sl, candidate)

            exit_reason = None
            if (side > 0 and price >= tp_price) or (side < 0 and price <= tp_price):
                exit_reason = "TP"
            elif (side > 0 and price <= trailing_sl) or (side < 0 and price >= trailing_sl):
                exit_reason = "SL" if trailing_sl == sl_price else "TRAIL_SL"
            elif idx - entry_idx >= max_hold_bars:
                exit_reason = "MAX_HOLD"
            elif idx == len(df) - 1 or df.index[idx + 1].date() != ts.date():
                exit_reason = "EOD"

            if exit_reason:
                exit_price = apply_slippage(price, -side)
                qty        = abs(position)
                gross      = position * (exit_price - entry_price)
                fees       = calculate_zerodha_fees(entry_price, exit_price, qty)
                net        = gross - fees
                equity    += net

                trades.append(
                    {
                        "entry_ts":    df.index[entry_idx],
                        "exit_ts":     ts,
                        "side":        "BUY" if side > 0 else "SELL",
                        "entry_price": entry_price,
                        "exit_price":  exit_price,
                        "atr":         entry_atr,
                        "qty":         qty,
                        "gross_pnl":   gross,
                        "fees":        fees,
                        "pnl":         net,
                        "exit_reason": exit_reason,
                        "p_long":      p_long,
                        "p_short":     p_short,
                        "equity":      equity,
                    }
                )

                # reset state
                position = 0
                entry_price = entry_atr = trailing_sl = None
                entry_idx   = None

    # ── Post‑run metrics ───────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    metrics   = {
        "Trades":      len(trades_df),
        "WinRate":     trades_df.pnl.gt(0).mean() if not trades_df.empty else 0.0,
        "GrossPnL":    trades_df.gross_pnl.sum()  if not trades_df.empty else 0.0,
        "Fees":        trades_df.fees.sum()       if not trades_df.empty else 0.0,
        "NetPnL":      trades_df.pnl.sum()        if not trades_df.empty else 0.0,
        "EquityFinal": equity,
    }
    return trades_df, metrics
# ─────────────────────────────────────────────────────────────────
