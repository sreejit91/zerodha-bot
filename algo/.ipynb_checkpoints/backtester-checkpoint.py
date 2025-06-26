"""Walk-forward back-tester that sizes trades by capital per entry."""
from __future__ import annotations
import pandas as pd
from .simple_model import load_or_train, predict_last
from .model        import LOOKBACK
from .features     import add_indicators


def backtest(
    df_raw: pd.DataFrame,
    model=None,
    capital: float = None,       # ₹ capital to deploy per trade
    contract_size: int = 1,      # units per lot/share
    sl_pct: float = 0.0015,      # 0.15% stop-loss
    tgt_pct: float = 0.0030,     # 0.30% target
    upper: float = 0.70,         # BUY if P(up) > upper
    lower: float = 0.30,         # SELL if P(up) < lower
):
    # ── prepare features & model ──────────────────────────────
    df = add_indicators(df_raw.copy())
    if model is None:
        model = load_or_train(df)    # trains or loads cached logistic model

    # initialise signal & PnL columns
    df["signal"] = None
    df["pnl"]    = None
    # optional: track dynamic qty
    df["qty"]    = None

    pos, entry = 0, 0.0             # +1 long, –1 short
    for idx in range(LOOKBACK, len(df)):
        lbl   = df.index[idx]
        price = df.at[lbl, "close"]

        # ── ML probability & trend filter ─────────────────────
        prob = predict_last(df.iloc[: idx + 1], model)
        ema_fast  = df["ema_8"].iloc[idx]
        ema_slow  = df["ema_21"].iloc[idx]
        bull_trend = ema_fast > ema_slow

        # decide BUY / SELL / None
        if   prob > upper and bull_trend:
            sig = "BUY"
        elif prob < lower and not bull_trend:
            sig = "SELL"
        else:
            sig = None

        # ── entry ────────────────────────────────────────────
        if pos == 0 and sig in ("BUY", "SELL"):
            pos   = 1 if sig == "BUY" else -1
            entry = price

            # compute quantity from capital
            if capital is not None:
                qty = int(capital // (entry * contract_size))
                qty = max(qty, 1)
            else:
                qty = 1

            df.at[lbl, "signal"] = sig
            df.at[lbl, "qty"]    = qty
            continue

        # ── exit ─────────────────────────────────────────────
        if pos != 0:
            tgt = entry * (1 + tgt_pct if pos > 0 else 1 - tgt_pct)
            sl  = entry * (1 - sl_pct  if pos > 0 else 1 + sl_pct)
            hit = (
                (pos > 0 and (price >= tgt or price <= sl)) or
                (pos < 0 and (price <= tgt or price >= sl))
            )
            if hit:
                df.at[lbl, "signal"] = "EXIT"
                df.at[lbl, "qty"]    = qty
                # pnl in ₹: price move × pos × qty × contract_size
                df.at[lbl, "pnl"]    = (price - entry) * pos * qty * contract_size
                pos = 0

    # ── compute metrics ───────────────────────────────────────
    closed = df.dropna(subset=["pnl"])
    metrics = {
        "round_trips": len(closed),
        "total_pnl"  : float(closed.pnl.sum()),
        "win_rate"   : float((closed.pnl > 0).mean() * 100) if not closed.empty else 0.0,
    }

    return df, metrics