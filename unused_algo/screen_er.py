"""Very‑lightweight screener that picks the most liquid trending symbol from NIFTY‑50."""
import pandas as pd
from algo.features import add_indicators

WATCH = {
    "RELIANCE": 738561,
    "TCS": 2953217,
    "INFY": 408065,
    # …
}


def pick(broker, days: int = 5):
    best, score = None, -999
    for sym, token in WATCH.items():
        df = add_indicators(broker.history(days=days, interval="15minute", instrument_token=token))
        if len(df) < 50:
            continue
        rvol = df.volume.tail(32).mean() / df.volume.tail(100).mean()
        trend = (df.ema_8.iloc[-1] - df.ema_21.iloc[-1]) / df.close.iloc[-1]
        _score = 0.7 * rvol + 0.3 * trend
        if _score > score:
            best, score = (sym, token), _score
    return best  # (symbol, token)