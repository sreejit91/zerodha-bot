"""Feature-engineering utilities (technical indicators, etc.)."""

import numpy as np
# ---- hot-patch for pandas-ta on NumPy ≥1.26 ---------------------------
if not hasattr(np, "NaN"):           # add attribute if it is missing
    np.NaN = np.nan

import pandas as pd
import pandas_ta as ta
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a raw OHLCV DataFrame (must include columns
    'open','high','low','close','volume') and return
    a new DataFrame with technical indicators added.
    """
    df = df.copy()
    # ensure chronological order
    df.sort_index(inplace=True)

    # ── Trend ─────────────────────────────────────────────
    df["ema_8"]  = EMAIndicator(df["close"], 8).ema_indicator()
    df["ema_21"] = EMAIndicator(df["close"], 21).ema_indicator()
    df["atr"]    = AverageTrueRange(df["high"], df["low"], df["close"]) \
                       .average_true_range()

    # ── Returns ───────────────────────────────────────────
    df["ret1"] = df["close"].pct_change().fillna(0)

    # ── Momentum & Volume ─────────────────────────────────
    df["rsi14"] = ta.rsi(df["close"], length=14)

    macd, macds, _ = ta.macd(df["close"])
    df["macd"]  = macd
    df["macds"] = macds

    df["obv"] = ta.obv(df["close"], df["volume"]).ffill()

    # Relative volume: last 20 bars vs last 100 bars
    rv20 = df["volume"].rolling(20).mean() / df["volume"].rolling(100).mean()
    df["rv20"] = rv20.fillna(1.0)

    return df.dropna()

# ────────────────────────────────────────────────────────
# List of feature columns produced by add_indicators()
FEATURES = [
    "close", "ema_8", "ema_21", "atr", "ret1",
    "rsi14", "macd", "macds", "obv", "rv20",
]