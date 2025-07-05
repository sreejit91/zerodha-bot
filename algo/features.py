import pandas as pd

# Updated feature generation to include ret1 for live WS ticks
FEATURES = ['ret1', 'ema_3', 'ema_8', 'vwap', 'vol_spike']


def add_indicators(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Add intraday technical indicators to the DataFrame.
    Indicators enabled: ret1, EMA-3, EMA-8, VWAP, volume spike.
    Works whether index is naive or tz-aware.
    """
    df = df.copy()

    # 1️⃣ Ensure datetime index, drop tz if present
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

    # 2️⃣ ret1: 1-period return (pct change or diff)
    df['ret1'] = df['close'].pct_change().fillna(0)
    if debug:
        print("ret1 head:", df['ret1'].head().tolist())

    # 3️⃣ EMA-3 & EMA-8 on close
    df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    if debug:
        print("ema_3 head:", df['ema_3'].head().tolist())
        print("ema_8 head:", df['ema_8'].head().tolist())

    # 4️⃣ Manual VWAP, reset each day
    tp = (df['high'] + df['low'] + df['close']) / 3
    tpv = tp * df['volume']
    dates = df.index.date
    df['cum_tpv'] = tpv.groupby(dates).cumsum()
    df['cum_vol'] = df['volume'].groupby(dates).cumsum()
    df['vwap'] = df['cum_tpv'] / df['cum_vol']
    if debug:
        print("vwap head:", df['vwap'].head().tolist())

    # 5️⃣ Volume spike: 1 if volume > 2× rolling-20 average
    df['vol_avg_20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['vol_spike'] = (df['volume'] > 2 * df['vol_avg_20']).astype(int)
    if debug:
        print("vol_spike head:", df['vol_spike'].head().tolist())

    # 6️⃣ Clean up temporary columns
    df.drop(columns=['cum_tpv', 'cum_vol', 'vol_avg_20'], inplace=True)

    return df
