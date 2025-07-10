# algo/features.py
import numpy as np
import pandas as pd

# ╭─ configurable hyper-parameters ───────────────────────────────╮
EMA_SLOW_LEN   = 21
EMA_FAST_LEN   = 8
RSI_LEN        = 14
VOL_MA_LEN     = 20
VOL_SPIKE_FACTOR = 2.0
ST_LEN         = 10
ST_MULT        = 3.0
BB_LEN         = 20
STOCH_LEN      = 14
ADX_LEN        = 14
# ╰───────────────────────────────────────────────────────────────╯

# ----------------------------------------------------------------
ENABLED: list[str] = [
    # classic features
    "ret1", "ret3", "ret5", "ret10",
    f"ema_{EMA_FAST_LEN}", f"ema_{EMA_SLOW_LEN}",
    "vwap",
    "vol_spike", "vol_change",
    "close_change",
    "obv",
    f"rsi_{RSI_LEN}",
    "macd", "macd_signal",
    "supertrend", "supertrend_dir",
    # new features
    "bb_upper", "bb_lower", "bb_width",
    "stoch_k", "stoch_d",
    "williams_r",
    "adx", "atr",
    "zscore_close", "zscore_volume",
    # session features (if intraday)
    # "hour", "dow",
]
FEATURES = ENABLED
# ----------------------------------------------------------------

def _true_range(high, low, close):
    pc = close.shift()
    return pd.concat([high - low,
                      (high - pc).abs(),
                      (low  - pc).abs()], axis=1).max(axis=1)

def _wilder_atr(df, length=ST_LEN):
    tr = _true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1/length, adjust=False,
                  min_periods=length).mean()

def _supertrend(df, length=ST_LEN, mult=ST_MULT):
    atr  = _wilder_atr(df, length)
    hl2  = (df["high"] + df["low"]) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        if df["close"].iat[i-1] > final_upper.iat[i-1]:
            final_upper.iat[i] = upper.iat[i]
        else:
            final_upper.iat[i] = min(upper.iat[i], final_upper.iat[i-1])
        if df["close"].iat[i-1] < final_lower.iat[i-1]:
            final_lower.iat[i] = lower.iat[i]
        else:
            final_lower.iat[i] = max(lower.iat[i], final_lower.iat[i-1])

    st  = pd.Series(np.nan,  index=df.index)
    dir = pd.Series(0,       index=df.index)
    for i in range(length, len(df)):
        if df["close"].iat[i-1] <= st.iat[i-1]:
            if df["close"].iat[i] > final_upper.iat[i]:
                dir.iat[i] = 1
                st.iat[i]  = final_lower.iat[i]
            else:
                dir.iat[i] = -1
                st.iat[i]  = final_upper.iat[i]
        else:
            if df["close"].iat[i] < final_lower.iat[i]:
                dir.iat[i] = -1
                st.iat[i]  = final_upper.iat[i]
            else:
                dir.iat[i] = 1
                st.iat[i]  = final_lower.iat[i]
    return st, dir

def add_indicators(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # returns
    if "ret1" in ENABLED:  df["ret1"] = df["close"].pct_change(1)
    if "ret3" in ENABLED:  df["ret3"] = df["close"].pct_change(3)
    if "ret5" in ENABLED:  df["ret5"] = df["close"].pct_change(5)
    if "ret10" in ENABLED: df["ret10"] = df["close"].pct_change(10)

    # EMAs
    if f"ema_{EMA_FAST_LEN}" in ENABLED:
        df[f"ema_{EMA_FAST_LEN}"] = (
            df["close"].ewm(span=EMA_FAST_LEN, adjust=False, min_periods=EMA_FAST_LEN).mean()
        )
    if f"ema_{EMA_SLOW_LEN}" in ENABLED:
        df[f"ema_{EMA_SLOW_LEN}"] = (
            df["close"].ewm(span=EMA_SLOW_LEN, adjust=False, min_periods=EMA_SLOW_LEN).mean()
        )

    # vwap
    if "vwap" in ENABLED:
        tp  = (df["high"] + df["low"] + df["close"]) / 3.0
        tpv = tp * df["volume"]
        g   = df.index.date
        df["_cum_tpv"] = tpv.groupby(g).cumsum()
        df["_cum_vol"] = df["volume"].groupby(g).cumsum()
        df["vwap"]     = df["_cum_tpv"] / df["_cum_vol"]

    # volume spike
    if "vol_spike" in ENABLED:
        vol_ma = df["volume"].rolling(VOL_MA_LEN, min_periods=1).mean()
        df["vol_spike"] = (df["volume"] > VOL_SPIKE_FACTOR * vol_ma).astype(int)

    # price/volume change
    if "close_change" in ENABLED:
        df["close_change"] = df["close"].diff()
    if "vol_change" in ENABLED:
        df["vol_change"] = df["volume"].pct_change()

    # OBV
    if "obv" in ENABLED:
        df["obv"] = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()

    # RSI
    if f"rsi_{RSI_LEN}" in ENABLED:
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(RSI_LEN).mean()
        dn = (-delta.clip(upper=0)).rolling(RSI_LEN).mean()
        rs = up / dn
        df[f"rsi_{RSI_LEN}"] = 100 - 100 / (1 + rs)

    # MACD
    if {"macd", "macd_signal"} & set(ENABLED):
        ema12 = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema12 - ema26
        if "macd" in ENABLED: df["macd"] = macd_line
        if "macd_signal" in ENABLED: df["macd_signal"] = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()

    # SuperTrend
    if {"supertrend", "supertrend_dir"} & set(ENABLED):
        st, direction = _supertrend(df)
        if "supertrend" in ENABLED: df["supertrend"] = st
        if "supertrend_dir" in ENABLED: df["supertrend_dir"] = direction

    # Bollinger Bands
    if {"bb_upper", "bb_lower", "bb_width"} & set(ENABLED):
        mid = df["close"].rolling(BB_LEN, min_periods=BB_LEN).mean()
        std = df["close"].rolling(BB_LEN, min_periods=BB_LEN).std()
        if "bb_upper" in ENABLED: df["bb_upper"] = mid + 2 * std
        if "bb_lower" in ENABLED: df["bb_lower"] = mid - 2 * std
        if "bb_width" in ENABLED: df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid

    # Stochastic Oscillator
    if {"stoch_k", "stoch_d"} & set(ENABLED):
        lowest_low = df["low"].rolling(STOCH_LEN, min_periods=STOCH_LEN).min()
        highest_high = df["high"].rolling(STOCH_LEN, min_periods=STOCH_LEN).max()
        stoch_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        if "stoch_k" in ENABLED: df["stoch_k"] = stoch_k
        if "stoch_d" in ENABLED: df["stoch_d"] = stoch_k.rolling(3, min_periods=3).mean()

    # Williams %R
    if "williams_r" in ENABLED:
        lowest_low = df["low"].rolling(STOCH_LEN, min_periods=STOCH_LEN).min()
        highest_high = df["high"].rolling(STOCH_LEN, min_periods=STOCH_LEN).max()
        df["williams_r"] = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

    # ATR
    if "atr" in ENABLED:
        df["atr"] = _wilder_atr(df, length=ST_LEN)

    # ADX
    if "adx" in ENABLED:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = _true_range(high, low, close)
        atr = tr.rolling(ADX_LEN, min_periods=ADX_LEN).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/ADX_LEN, adjust=False).mean() / atr)
        minus_di = -100 * (minus_dm.ewm(alpha=1/ADX_LEN, adjust=False).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df["adx"] = dx.ewm(alpha=1/ADX_LEN, adjust=False).mean()

    # Z-score
    if "zscore_close" in ENABLED:
        df["zscore_close"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    if "zscore_volume" in ENABLED:
        df["zscore_volume"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

    # clean up
    df.drop(columns=[c for c in ["_cum_tpv", "_cum_vol"] if c in df], inplace=True)

    if debug:
        print("\nPreview of computed indicators:")
        print(df[FEATURES].head(25))

    return df
