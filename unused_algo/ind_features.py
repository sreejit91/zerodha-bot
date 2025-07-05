from __future__ import annotations
import pandas as pd

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append scalping-specific indicators to the DataFrame:
    - ema_fast: 8-period EWM of close
    - ema_slow: 21-period EWM of close
    - vol_avg20: 20-bar rolling average of volume
    - rsi2: 2-period RSI
    - vwap: cumulative VWAP for the session

    Returns a new DataFrame with these columns added.
    """
    df = df.copy()

    # Fast & slow EMAs (1-minute context assumed)
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()

    # 20-bar average volume
    df['vol_avg20'] = df['volume'].rolling(window=20, min_periods=1).mean()

    # RSI(2)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing: alpha = 1/2
    avg_gain = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi2'] = 100 - (100 / (1 + rs))

    # VWAP: cumulative price*volume / cumulative volume
    pv = df['close'] * df['volume']
    df['cum_pv'] = pv.cumsum()
    df['cum_vol'] = df['volume'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']

    # Clean up temporary columns
    df.drop(columns=['cum_pv', 'cum_vol'], inplace=True)

    return df
