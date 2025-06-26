def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicator columns **in-place** and return df.
    All new columns are created even on very short DataFrames; pd.NA fills
    rows where the look-back window is not yet complete.
    """
    # ── EMA family ────────────────────────────────────────────────────
    for n in (3, 4, 8, 9, 21):
        df[f"ema_{n}"] = ta.ema(df["close"], length=n)

    # ── ATRs ──────────────────────────────────────────────────────────
    atr14 = _safe(ta.atr, df["high"], df["low"], df["close"], length=14)
    atr20 = _safe(ta.atr, df["high"], df["low"], df["close"], length=20)
    df["atr"]    = atr14 if atr14 is not None else pd.NA
    df["atr_20"] = atr20 if atr20 is not None else pd.NA

    # ── RSI(14) ───────────────────────────────────────────────────────
    rsi = _safe(ta.rsi, df["close"], length=14)
    df["rsi"] = rsi if rsi is not None else pd.NA

    # ── MACD (fast 12 / slow 26 / signal 9) ───────────────────────────
    macd = _safe(ta.macd, df["close"])
    if macd is not None:
        df["macd"]  = macd.iloc[:, 0]
        df["macds"] = macd.iloc[:, 1]
    else:
        df["macd"]  = pd.NA
        df["macds"] = pd.NA

    # ── OBV ───────────────────────────────────────────────────────────
    obv = _safe(ta.obv, df["close"], df["volume"])
    df["obv"] = obv if obv is not None else pd.NA

    # ── Realised volatility 20 ────────────────────────────────────────
    df["rv20"] = (
        df["close"]
        .pct_change()
        .rolling(20)
        .std()
        .fillna(pd.NA)
    )

    return df
