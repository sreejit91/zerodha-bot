# ── Monkey‐patch NumPy so pandas_ta can import ─────────────────────────
import numpy as _np
if not hasattr(_np, "NaN"):
    _np.NaN = _np.nan

import pandas as pd
import pandas_ta as ta

# ─── Tweak this list to turn indicators on/off ─────────────────────────
ENABLED = [
    "ret1", "ema_8", "ema_21",
    "rsi_14", "macd", "vwap",
    "vol_spike", "bollinger",
    "stochastic", "obv", "supertrend",
]

def add_indicators(df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    # ① normalize index
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    elif df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # ② loop over each enabled feature
    for feat in ENABLED:
        if feat == "ret1":
            df["ret1"] = df["close"].pct_change().fillna(0)

        elif feat.startswith("ema_"):
            n = int(feat.split("_")[1])
            df[f"ema_{n}"] = df["close"].ewm(span=n, adjust=False).mean()

        elif feat.startswith("rsi_"):
            n = int(feat.split("_")[1])
            df[f"rsi_{n}"] = ta.rsi(df["close"], length=n)

        elif feat == "macd":
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)

        elif feat == "vwap":
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        elif feat == "vol_spike":
            vol_avg = df["volume"].rolling(20, min_periods=1).mean()
            df["vol_spike"] = (df["volume"] > 2 * vol_avg).astype(int)

        elif feat == "bollinger":
            bb = ta.bbands(df["close"], length=20, std=2)
            df["bb_mid"]   = bb[f"BBM_20_2.0"]
            df["bb_lower"] = bb[f"BBL_20_2.0"]
            df["bb_upper"] = bb[f"BBU_20_2.0"]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        elif feat == "stochastic":
            # raw %K over 14, %D = 3-period SMA of %K
            low  = df["low"].rolling(14, min_periods=1).min()
            high = df["high"].rolling(14, min_periods=1).max()
            k    = 100 * (df["close"] - low) / (high - low)
            df["stoch_k"] = k
            df["stoch_d"] = k.rolling(3, min_periods=1).mean()

        elif feat == "obv":
            df["obv"] = ta.obv(df["close"], df["volume"])

        elif feat == "supertrend":
            st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3)
            df["supertrend"]     = st[f"SUPERT_10_3.0"]
            df["supertrend_dir"] = st[f"SUPERTd_10_3.0"]

        else:
            raise ValueError(f"Unknown feature: {feat}")

        if debug:
            print(f"[feat] computed {feat}")

    return df
