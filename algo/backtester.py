"""
algo/backtester.py
────────────────────────────────────────────────────────────────────────
Vectorised intraday back-tester that mirrors the live-trading loop:

• works on any feature-enriched DataFrame (call add_indicators() first)
• supports fixed SL / TP, trailing stop, time-stop
• includes Zerodha all-in cost model (brokerage + taxes)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from .features import FEATURES, add_indicators          # FEATURES list
from .model    import predict_last, LOOKBACK            # model helpers

# ════════════════════════════════════════════════════════════════════
# Zerodha fee helper (embedded – no external import needed)
# ════════════════════════════════════════════════════════════════════
def zerodha_cost(turnover: float, side: str = "BUY") -> float:
    """
    Approximate all-in cost (brokerage + STT + exchange + GST + stamp duty)
    for an equity intraday trade.

    Parameters
    ----------
    turnover : float
        price × quantity for **this leg** of the trade.
    side : {"BUY", "SELL"}
        Cost differs slightly on BUY vs SELL because stamp duty (buy only)
        and STT (sell only).

    Returns
    -------
    float  – positive rupee amount you should subtract from PnL.
    """
    brk   = min(0.0003 * turnover, 20)            # brokerage 0.03 % capped 20
    exch  = 0.0000325 * turnover                  # exchange txn
    sebi  = 0.000001  * turnover                  # SEBI
    gst   = 0.18 * (brk + exch)                   # GST on (brk+exch)
    stamp = 0.00003  * turnover if side == "BUY"  else 0
    stt   = 0.00025  * turnover if side == "SELL" else 0
    return brk + exch + sebi + gst + stamp + stt


# ════════════════════════════════════════════════════════════════════
# Back-test core
# ════════════════════════════════════════════════════════════════════
def backtest(
    df_raw      : pd.DataFrame,
    model,
    capital     : float = 100_000,
    contract_size : int   = 1,
    sl_pct      : float = 0.0012,
    tp_pct      : float = 0.0027,
    trail_pct   : float = 0.0020,
    hold_max    : int   = 30,       # bars
    upper       : float = 0.60,
    lower       : float = 0.40,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate a long/short single-position strategy on *df_raw*.

    df_raw **must already contain** the indicator columns in FEATURES.

    Returns
    -------
    trades : DataFrame   – every fill & exit row-by-row
    metrics: dict        – summary (trades, win-rate, PnL …)
    """
    # Ensure DataFrame is sorted & has the needed return column
    df = df_raw.sort_index().copy()
    if "ret1" not in df.columns:
        df["ret1"] = df["close"].pct_change()

    FEATURE_COLS = FEATURES                         # shorthand

    equity   = capital
    in_pos   = 0            # +1 long, −1 short, 0 flat
    entry_px = 0.0
    sl_px    = tp_px = 0.0
    age      = 0

    rows: List[dict] = []

    for idx, row in df.iterrows():
        price = row["close"]

        # 1) calculate ML probability on last *LOOKBACK* closed bars
        window = df.loc[:idx].tail(LOOKBACK)

        # skip until full window w/o NaNs
        if window[FEATURE_COLS].isna().any().any():
            continue
        prob = predict_last(window, model)

        # 2) trend filter (ema_3 vs ema_8)
        ema3, ema8 = row["ema_3"], row["ema_8"]
        long_ok  = prob >= upper and ema3 > ema8
        short_ok = prob <= lower and ema3 < ema8

        # 3) exit logic first ------------------------------------------------
        exit_reason = None
        if in_pos != 0:
            age += 1
            # trailing SL update
            if in_pos == 1:          # long
                sl_px = max(sl_px, price * (1 - trail_pct))
            else:                    # short
                sl_px = min(sl_px, price * (1 + trail_pct))

            hit_sl  = (price <= sl_px) if in_pos == 1 else (price >= sl_px)
            hit_tp  = (price >= tp_px) if in_pos == 1 else (price <= tp_px)
            time_up = age >= hold_max

            if hit_tp:  exit_reason = "TP"
            elif hit_sl: exit_reason = "SL"
            elif time_up: exit_reason = "TIME"

            if exit_reason:
                turnover = price * contract_size
                fee      = zerodha_cost(turnover, "SELL" if in_pos==1 else "BUY")
                pl_abs   = (price - entry_px) * in_pos * contract_size - fee
                equity  += pl_abs
                rows.append(dict(ts=idx, side="EXIT", price=price,
                                 pnl=pl_abs, equity=equity,
                                 reason=exit_reason))
                in_pos = age = 0

        # 4) entry logic -----------------------------------------------------
        if in_pos == 0:
            if long_ok or short_ok:
                in_pos   = 1 if long_ok else -1
                entry_px = price
                sl_px    = (price * (1 - sl_pct) if in_pos == 1
                            else price * (1 + sl_pct))
                tp_px    = (price * (1 + tp_pct) if in_pos == 1
                            else price * (1 - tp_pct))
                turnover = price * contract_size
                fee      = zerodha_cost(turnover, "BUY" if in_pos==1 else "SELL")
                equity  -= fee
                rows.append(dict(ts=idx,
                                 side="BUY" if in_pos==1 else "SELL",
                                 price=price,
                                 pnl=-fee,
                                 equity=equity,
                                 reason="ENTER"))

    trades = pd.DataFrame(rows).set_index("ts")

    metrics = dict(
        Trades      = trades.side.isin(["BUY", "SELL"]).sum() // 2,
        WinRate     = (trades.reason == "TP").sum() /
                      max(1, trades.reason.isin(["TP", "SL", "TIME"]).sum()),
        PnL         = trades.pnl.sum(),
        EquityFinal = equity,
    )
    return trades, metrics
