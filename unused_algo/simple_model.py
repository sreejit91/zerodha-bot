from __future__ import annotations
import pandas as pd

from unused_algo.ind_features import add_indicators


def load_or_train_simple(df_raw: pd.DataFrame, retrain: bool = False) -> None:
    """
    Placeholder for a simple indicator-based model; no training required.
    Returns None as the model is stateless.
    """
    return None


def predict_last_simple(df_raw: pd.DataFrame, model=None) -> float:
    """
    Generate a probability signal based on scalping indicators:
    - Long (1.0) when fast EMA crosses above slow EMA,
      AND volume >1.2×20-bar avg,
      AND RSI2 <30,
      AND price < VWAP
    - Short (0.0) when fast EMA crosses below slow EMA,
      AND volume >1.2×20-bar avg,
      AND RSI2 >70,
      AND price > VWAP
    - Neutral (0.5) otherwise
    """
    df = add_indicators(df_raw.copy())
    # need at least two bars to detect crossover
    if len(df) < 2:
        return 0.5
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 1) EMA crossover
    cross_up   = (last['ema_fast'] > last['ema_slow']) and (prev['ema_fast'] <= prev['ema_slow'])
    cross_down = (last['ema_fast'] < last['ema_slow']) and (prev['ema_fast'] >= prev['ema_slow'])

        # 2) Volume spike threshold (compare to previous bar avg)
    vol_ok = last['volume'] > 1.2 * prev['vol_avg20']

    # 3) RSI2 thresholds
    rsi_ok_long  = last['rsi2'] < 30
    rsi_ok_short = last['rsi2'] > 70

    # 4) VWAP mean reversion
    vwap_ok_long  = last['close'] < last['vwap']
    vwap_ok_short = last['close'] > last['vwap']

    # combine conditions
    if cross_up and vol_ok and rsi_ok_long and vwap_ok_long:
        return 1.0
    if cross_down and vol_ok and rsi_ok_short and vwap_ok_short:
        return 0.0
    return 0.5
