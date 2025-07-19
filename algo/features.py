# algo/features.py
import numpy as np
import pandas as pd
from algo.backtester import calculate_zerodha_fees

# ╭─ Configurable Hyperparameters ─────────────────────────────╮
RSI_LEN           = 2
VOL_MA_LEN        = 20
VOL_SPIKE_FACTOR  = 2.0
ST_LEN            = 10
ST_MULT           = 3.0
BB_LEN            = 20
STOCH_LEN         = 14
ADX_LEN           = 14
VOL_WIN           = 20
ATR_WINDOW        = 20
BREAKOUT_WINDOW   = 10
DONCHIAN_WINDOW   = 20
KELTNER_LEN       = 20
# ╰────────────────────────────────────────────────────────────╯

ENABLED: list[str] = [
    # --- Classic features ---
    "ret1",
     "ret2",
    #"ret5",
    #"up_streak",
    #"down_streak",
    "atr",
    # "ret10",
    # "vol_avg20",
    "ema_5",         # NEW: explicitly add these if you want fast/slow EMAs!
    "ema_20","ema5_ema20_diff",
    # "ema_50",
    # "ema_15",
    "vwap",
    "close_vs_vwap",
    #"range_1",
    #"body_1",
    #"is_vol_spike",
    # "vol_change",
    # "close_change",
    # "obv",
    "rsi_2",
     #"macd", "macd_signal",
     #"supertrend", "supertrend_dir",
    # --- New features ---
    #"above_high_10",
    #"below_low_10",
    "bb_upper",
    "bb_lower",
    #"bb_upper_touch",
    #"bb_lower_touch",
    # "bb_mid",
     "bb_width",
    # "stoch_k",
    # "stoch_d",
    # "williams_r",
    # "adx",
    # "atr_median20",
    # "zscore_close",
    # "zscore_volume",
    # --- Session/lag features ---
    # "close_lag_1", "close_lag_3", "close_lag_5",
    # "volume_lag_1", "volume_lag_5",
    #"volatility_5", "volatility_10",
    "vol_spike",
    # "high_low_range","high_N","low_N","channel_mid"
    # "hour",
    # "dayofweek",
     "hour_sin","hour_cos",
    #"minute_of_day",
    #"fees_pct",
    #"pivot_point","pivot_res1","pivot_sup1",
    #"donchian_high","donchian_low","donchian_breakout",
    #"is_doji","is_hammer","is_bullish_engulfing",
    #"body_range_ratio",
    #"gap",
    #"clv",
    #"roc_5","roc_10",
    #"keltner_upper","keltner_lower",
    #"range_change",
    #"corr_close_vwap"
]
FEATURES = ENABLED

def _true_range(high, low, close):
    pc = close.shift()
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

def _wilder_atr(df, length=ST_LEN):
    tr = _true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def _supertrend(df, length=ST_LEN, mult=ST_MULT):
    atr = _wilder_atr(df, length)
    hl2 = (df["high"] + df["low"]) / 2.0
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
    st = pd.Series(np.nan, index=df.index)
    direction = pd.Series(0, index=df.index)
    for i in range(length, len(df)):
        if df["close"].iat[i-1] <= st.iat[i-1]:
            if df["close"].iat[i] > final_upper.iat[i]:
                direction.iat[i] = 1
                st.iat[i] = final_lower.iat[i]
            else:
                direction.iat[i] = -1
                st.iat[i] = final_upper.iat[i]
        else:
            if df["close"].iat[i] < final_lower.iat[i]:
                direction.iat[i] = -1
                st.iat[i] = final_upper.iat[i]
            else:
                direction.iat[i] = 1
                st.iat[i] = final_lower.iat[i]
    return st, direction



def add_fee_feature(df, contract_size=10):
    # This is a forward-looking feature, so don't leak info! Use previous close, or mid, or a lag
    df = df.copy()
    # We'll assume buy/sell at close price, typical size
    df['fees_estimate'] = calculate_zerodha_fees(df['close'], df['close'], contract_size)
    return df




def add_indicators(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    df = df.copy()
    df = add_fee_feature(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # ATR and ADX
    if "atr" in ENABLED:
        df["atr"] = _wilder_atr(df, length=ST_LEN)
    if "atr_median20" in ENABLED:
        df["atr_median20"] = df["atr"].rolling(window=ATR_WINDOW, min_periods=ATR_WINDOW).median()
    if "adx" in ENABLED:
        high, low, close = df["high"], df["low"], df["close"]
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

    # Lagged prices and volume
    if "close_lag_1" in ENABLED: df["close_lag_1"] = df["close"].shift(1)
    if "close_lag_3" in ENABLED: df["close_lag_3"] = df["close"].shift(3)
    if "close_lag_5" in ENABLED: df["close_lag_5"] = df["close"].shift(5)
    if "volume_lag_1" in ENABLED: df["volume_lag_1"] = df["volume"].shift(1)
    if "volume_lag_5" in ENABLED: df["volume_lag_5"] = df["volume"].shift(5)
    if "vol_avg20" in ENABLED: df["vol_avg20"] = df["volume"].rolling(window=VOL_WIN).mean()
    if "vol_dev20" in ENABLED: df["vol_dev20"] = (df["volume"] - df["vol_avg20"]) / df["vol_avg20"]

    # Volatility
    if "volatility_5" in ENABLED: df["volatility_5"] = df["close"].pct_change().rolling(5).std()
    if "volatility_10" in ENABLED: df["volatility_10"] = df["close"].pct_change().rolling(10).std()

    # High-low and breakouts
    if "high_low_range" in ENABLED: df["high_low_range"] = df["high"] - df["low"]
    if "high_N" in ENABLED: df["high_N"] = df["high"].rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).max().shift(1)
    if "low_N" in ENABLED: df["low_N"] = df["low"].rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).min().shift(1)
    if "channel_mid" in ENABLED: df["channel_mid"] = (df["high_N"] + df["low_N"]) / 2

    # Time features
    if "hour" in ENABLED: df["hour"] = df.index.hour
    if "dayofweek" in ENABLED: df["dayofweek"] = df.index.dayofweek
    if "minute_of_day" in ENABLED: df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    seconds = (df.index.hour * 3600 + df.index.minute * 60 + df.index.second)
    if "hour_sin" in ENABLED: df["hour_sin"] = np.sin(2 * np.pi * seconds / 86400)
    if "hour_cos" in ENABLED: df["hour_cos"] = np.cos(2 * np.pi * seconds / 86400)

    # Returns and streaks
    if "ret1" in ENABLED:  df["ret1"] = df["close"].pct_change(1)
    if "ret2" in ENABLED:  df["ret2"] = df["close"].pct_change(2)
    if "ret5" in ENABLED:  df["ret5"] = df["close"].pct_change(5)
    if "ret10" in ENABLED: df["ret10"] = df["close"].pct_change(10)
    if "up_streak" in ENABLED: df["up_streak"] = df["close"].diff().gt(0).rolling(5).sum()
    if "down_streak" in ENABLED: df["down_streak"] = df["close"].diff().lt(0).rolling(5).sum()

    # Zerodha fees
    if "fees_pct" in ENABLED: df['fees_pct'] = df['fees_estimate'] / df['close']



    # Range/body features
    if "range_1" in ENABLED: df["range_1"] = (df["high"] - df["low"]) / df["close"]
    if "body_1" in ENABLED: df["body_1"] = (df["close"] - df["open"]).abs() / (df["atr"] + 1e-8)
    if "above_high_10" in ENABLED: df["above_high_10"] = (df["close"] > df["high"].rolling(10).max().shift(1)).astype(int)
    if "below_low_10" in ENABLED: df["below_low_10"] = (df["close"] < df["low"].rolling(10).min().shift(1)).astype(int)

    # VWAP
    if "vwap" in ENABLED:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        tpv = tp * df["volume"]
        g = df.index.date
        df["_cum_tpv"] = tpv.groupby(g).cumsum()
        df["_cum_vol"] = df["volume"].groupby(g).cumsum()
        df["vwap"] = df["_cum_tpv"] / df["_cum_vol"]
    if "close_vs_vwap" in ENABLED: df["close_vs_vwap"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-8)
    if "corr_close_vwap" in ENABLED:
        df["corr_close_vwap"] = df["close"].rolling(20).corr(df["vwap"])

    # Volume spike
    if "vol_spike" in ENABLED:
        vol_ma = df["volume"].rolling(VOL_MA_LEN, min_periods=1).mean()
        df["vol_spike"] = (df["volume"] > VOL_SPIKE_FACTOR * vol_ma).astype(int)
    if "is_vol_spike" in ENABLED: df["is_vol_spike"] = (df["vol_spike"] > 2).astype(int)

    # EMAs
    if "ema_5" in ENABLED:
        df["ema_5"] = df["close"].ewm(span=5, adjust=False, min_periods=5).mean()
    if "ema_20" in ENABLED:
        df["ema_20"] = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    if "ema_50" in ENABLED:
        df["ema_50"] = df["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    if "ema_15" in ENABLED:
        df["ema_15"] = df["close"].ewm(span=15, adjust=False, min_periods=15).mean()
    if "ema5_ema20_diff" in ENABLED: df["ema5_ema20_diff"] = df["ema_5"] - df["ema_20"]

    # RSI
    if "rsi_2" in ENABLED:
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(RSI_LEN).mean()
        dn = (-delta.clip(upper=0)).rolling(RSI_LEN).mean()
        rs = up / dn
        df["rsi_2"] = 100 - 100 / (1 + rs)

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
        if "bb_width" in ENABLED: df["bb_width"] = 4 * std / mid
        if "bb_mid" in ENABLED: df["bb_mid"] = (df["bb_upper"] + df["bb_lower"]) / 2
    if "bb_upper_touch" in ENABLED: df["bb_upper_touch"] = (df["close"] >= df["bb_upper"]).astype(int)
    if "bb_lower_touch" in ENABLED: df["bb_lower_touch"] = (df["close"] <= df["bb_lower"]).astype(int)

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

    # OBV
    if "obv" in ENABLED:
        df["obv"] = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()

    # Z-score
    if "zscore_close" in ENABLED:
        df["zscore_close"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    if "zscore_volume" in ENABLED:
        df["zscore_volume"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

    # Pivot Points (Classic)
    if "pivot_point" in ENABLED:
        pp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['pivot_point'] = pp
    if "pivot_res1" in ENABLED:
        pp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['pivot_res1'] = 2 * pp - df['low'].shift(1)
    if "pivot_sup1" in ENABLED:
        pp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['pivot_sup1'] = 2 * pp - df['high'].shift(1)

    # Donchian Channel
    if "donchian_high" in ENABLED:
        df['donchian_high'] = df['high'].rolling(DONCHIAN_WINDOW).max().shift(1)
    if "donchian_low" in ENABLED:
        df['donchian_low'] = df['low'].rolling(DONCHIAN_WINDOW).min().shift(1)
    if "donchian_breakout" in ENABLED:
        high = df['high'].rolling(DONCHIAN_WINDOW).max().shift(1)
        low = df['low'].rolling(DONCHIAN_WINDOW).min().shift(1)
        # 1 if close breaks out above, -1 below, 0 otherwise
        df['donchian_breakout'] = np.where(df['close'] > high, 1, np.where(df['close'] < low, -1, 0))

    # Donchian channel breakout
    if "is_donchian_breakout" in ENABLED:
        high = df['high'].rolling(DONCHIAN_WINDOW).max().shift(1)
        low = df['low'].rolling(DONCHIAN_WINDOW).min().shift(1)
        df["is_donchian_breakout"] = ((df["close"] > high) | (df["close"] < low)).astype(int)

    # Doji: very small body
    if "is_doji" in ENABLED:
        df['is_doji'] = ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8) < 0.1).astype(int)

    # Hammer: lower wick > 2x body, small upper wick
    if "is_hammer" in ENABLED:
        body = (df['close'] - df['open']).abs()
        lower = df['open'].where(df['close'] >= df['open'], df['close']) - df['low']
        upper = df['high'] - df['close'].where(df['close'] >= df['open'], df['open'])
        df['is_hammer'] = ((lower > 2 * body) & (upper < 0.25 * body)).astype(int)

    # Bullish Engulfing: today's body engulfs previous body and closes higher
    if "is_bullish_engulfing" in ENABLED:
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        body_up = (df['close'] > df['open'])
        prev_body_down = (prev_close < prev_open)
        engulf = (df['close'] > prev_open) & (df['open'] < prev_close)
        df['is_bullish_engulfing'] = (body_up & prev_body_down & engulf).astype(int)

    # Body/Range Ratio
    if "body_range_ratio" in ENABLED:
        body = (df['close'] - df['open']).abs()
        rng = (df['high'] - df['low']) + 1e-8
        df['body_range_ratio'] = body / rng

    # Gap Up/Down
    if "gap" in ENABLED:
        df['gap'] = df['open'] - df['close'].shift(1)

    # High/Low Close Position
    if "clv" in ENABLED:
        df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)

    # Rolling Standard Deviation
    if "volatility_20" in ENABLED:
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    if "volatility_50" in ENABLED:
        df['volatility_50'] = df['close'].pct_change().rolling(50).std()

    # Rate of Change
    if "roc_5" in ENABLED:
        df['roc_5'] = df['close'].pct_change(5)
    if "roc_10" in ENABLED:
        df['roc_10'] = df['close'].pct_change(10)

    # Keltner Channel


    if "keltner_upper" in ENABLED:
        ema = df['close'].ewm(span=KELTNER_LEN, adjust=False).mean()
        atr = df['high'].rolling(KELTNER_LEN).max() - df['low'].rolling(KELTNER_LEN).min()
        df['keltner_upper'] = ema + 2 * atr
    if "keltner_lower" in ENABLED:
        ema = df['close'].ewm(span=KELTNER_LEN, adjust=False).mean()
        atr = df['high'].rolling(KELTNER_LEN).max() - df['low'].rolling(KELTNER_LEN).min()
        df['keltner_lower'] = ema - 2 * atr

    # Bar-to-Bar Range Change
    if "range_change" in ENABLED:
        rng = df['high'] - df['low']
        df['range_change'] = rng.pct_change()

    #
    # Clean up temporary columns
    df.drop(columns=[c for c in ["_cum_tpv", "_cum_vol"] if c in df], inplace=True)

    if debug:
        print("\nPreview of computed indicators:")
        print(df[FEATURES].head(25))


    return df




# ─── Label generator aligned with live TP/SL ──────────────────────
HORIZON      = 24      # ≈ one trading day if you use 5‑min bars
THR_ATR_MULT = 1.0     # must match tp/sl multiples below

def add_labels(df: pd.DataFrame,
               horizon: int = HORIZON,
               thr_atr_mult: float = THR_ATR_MULT,
               drop_flat: bool = False) -> pd.DataFrame:
    """
    3‑class horizon label:
        +1  ‑ future_ret > +thr
        -1  ‑ future_ret < -thr
         0  ‑ in‑between
    `drop_flat=True` switches to binary by discarding the 0‑class.
    """
    if "atr" not in df:
        raise ValueError("run add_indicators(df) first")

    df = df.copy()
    df["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1
    thr              = df["atr"] / df["close"] * thr_atr_mult

    df["label"] = np.select(
        [df.future_ret >  thr,
         df.future_ret < -thr],
        [ 1, -1], default=0
    ).astype("int8")

    if drop_flat:
        df = df[df.label != 0]

    return df.drop(columns="future_ret")
