"""
algo/signals.py
Translate ML probabilities + trend filter → BUY / SELL / None.
"""

from __future__ import annotations
import pandas as pd

# Graceful fallback if LOOKBACK is ever missing
try:
    from .model import LOOKBACK
except ImportError:
    LOOKBACK = 60

# ────────────────────────────────────────────────────────────────────
def generate_signal(
    df_feat: pd.DataFrame,
    model,
    upper: float = 0.53,   # probability ≥ upper → BUY
    lower: float = 0.47,   # probability ≤ lower → SELL
):
    """
    Return "BUY", "SELL", or None.

    Strategy:
        ML-probability filter  AND  EMA-trend filter.
    """
    df_feat = df_feat.tail(LOOKBACK).copy()

    prob = model.predict_proba(df_feat.iloc[[-1]])[0, 1]

    ema_fast = df_feat["ema_8"].iloc[-1]
    ema_slow = df_feat["ema_21"].iloc[-1]
    trend_up = ema_fast > ema_slow

    if prob >= upper and trend_up:
        return "BUY"
    if prob <= lower and not trend_up:
        return "SELL"
    return None
