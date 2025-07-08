"""
algo/features.py
────────────────
Technical feature engineering for HDFCBANK/IDEA intraday bars.

Added:
- atr14_pc   : 14-period ATR as % of price
- rsi14      : classic RSI(14)
- bb_z       : Bollinger-band z-score (20-period, 2 σ)
- tod_frac   : fraction of the trading day (09:15→15:30 ⇒ 0→1)
- dow_0 … 4  : one-hot day-of-week

The original columns (ret1, ema_8, ema_21, vwap, vol_spike) are kept
unchanged so live code continues to work.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ── feature list used everywhere ────────────────────────────────────
FEATURES = [
    # original
    "ret1", "ema_8", "ema_21", "vwap", "vol_spike",
    # new
    "atr14_pc", "rsi14", "bb_z", "tod_frac",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
]


def add_indicators(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Return a copy of *df* with all FEATURES populated."""
    out = df.copy()

    # 1️⃣ ensure datetime index (UTC-naive) ---------------------------------
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)

    # 2️⃣ 1-bar return ------------------------------------------------------
    out["ret1"] = out["close"].pct_change().fillna(0)
    if debug:
        print("ret1 head:", out["ret1"].head().tolist())

    # 3️⃣ EMA-8 / EMA-21 ----------------------------------------------------
    out["ema_8"]  = out["close"].ewm(span=8, adjust=False).mean()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()

    # 4️⃣ VWAP reset each day ----------------------------------------------
    tp   = (out["high"] + out["low"] + out["close"]) / 3
    tpv  = tp * out["volume"]
    dates = out.index.date
    out["vwap"] = tpv.groupby(dates).cumsum() / out["volume"].groupby(dates).cumsum()

    # 5️⃣ Volume spike flag -------------------------------------------------
    vol_avg = out["volume"].rolling(20, min_periods=1).mean()
    out["vol_spike"] = (out["volume"] > 2 * vol_avg).astype(int)

    # ── NEW FEATURES ──────────────────────────────────────────────────────
    # 6️⃣ ATR-14 (% of price) ----------------------------------------------
    hi_lo = out["high"] - out["low"]
    hi_pc = (out["high"] - out["close"].shift()).abs()
    lo_pc = (out["low"]  - out["close"].shift()).abs()
    tr    = pd.concat([hi_lo, hi_pc, lo_pc], axis=1).max(axis=1)
    out["atr14"]     = tr.rolling(14).mean()
    out["atr14_pc"]  = out["atr14"] / out["close"]

    # 7️⃣ RSI-14 ------------------------------------------------------------
    delta = out["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    rs    = gain / loss
    out["rsi14"] = 100 - 100 / (1 + rs)

    # 8️⃣ Bollinger z-score (20-period) ------------------------------------
    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_z"] = (out["close"] - mid) / std

    # 9️⃣ Time-of-day fraction ---------------------------------------------
    minutes = out.index.hour * 60 + out.index.minute
    minutes = minutes.to_numpy()  # <-- convert to ndarray
    start, end = 9 * 60 + 15, 15 * 60 + 30  # 555 … 930
    out["tod_frac"] = np.clip((minutes - start) / (end - start), 0, 1)

    # 🔟 Day-of-week one-hots ----------------------------------------------
    dow = out.index.dayofweek
    for d in range(5):                                 # Mon=0 … Fri=4
        out[f"dow_{d}"] = (dow == d).astype(int)

    # ── clean temp columns (atr14 kept for debug but not in FEATURES) ─────
    out.drop(columns=["atr14"], inplace=True)

    if debug:
        print("ema_8 head:",  out["ema_8"].head().tolist())
        print("ema_21 head:", out["ema_21"].head().tolist())
        print("vwap head:",   out["vwap"].head().tolist())
        print("vol_spike head:", out["vol_spike"].head().tolist())
        print("atr14_pc head:", out["atr14_pc"].head().tolist())
        print("rsi14 head:", out["rsi14"].head().tolist())
        print("bb_z head:",  out["bb_z"].head().tolist())

    return out
