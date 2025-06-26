"""Translate features + ML output into a discrete **BUY / SELL / None** signal."""
from __future__ import annotations
import pandas as pd
from .model import LOOKBACK


def generate_signal(
    df_feat: pd.DataFrame,
    model,
    upper: float = 0.53,  # probability threshold to go long
    lower: float = 0.47,  # probability threshold to go short
):
    """Return **BUY**, **SELL**, or **None**.

    Strategy  =  ML probability  ∧  trend filter (EMA‑8 > EMA‑21).
    Adjust *upper*/*lower* to make the system more/less aggressive.
    """
    if len(df_feat) < LOOKBACK:
        return None

    X_last = df_feat[["close", "ema_8", "ema_21", "atr"]].tail(LOOKBACK).values[None]
    prob_up = float(model.predict(X_last, verbose=0)[0, 0])

    if prob_up > upper:
        return "BUY"
    if prob_up < lower:
        return "SELL"
    return None