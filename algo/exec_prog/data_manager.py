#!/usr/bin/env python3
"""
exec_prog/data_manager.py

Interactively fetch & store historical OHLCV for given symbols, days & intervals,
appending to per-symbol CSVs in a `data/` directory.
"""

# ─── Now imports will work ────────────────────────────────────────────
import os
import pandas as pd
from algo.broker import KiteWrapper
from algo.config import load_config

def fetch_and_store(
    symbols: list[str],
    days: int,
    intervals: list[str],
    data_dir: str = "data"
) -> None:
    """
    For each symbol and interval:
      - fetch `days` of data via KiteWrapper.history()
      - append to data/<symbol>_<interval>.csv (dropping duplicates)
    """
    os.makedirs(data_dir, exist_ok=True)
    cfg  = load_config()
    wrap = KiteWrapper(cfg)

    for symbol in symbols:
        for interval in intervals:
            print(f"\n📥 Fetching {symbol} – last {days} days @ {interval} …")
            df_new = wrap.history(days=days, interval=interval, tradingsymbol=symbol)

            fname = f"{symbol}_{interval}.csv"
            path  = os.path.join(data_dir, fname)

            if os.path.exists(path):
                df_old = pd.read_csv(path, index_col=0, parse_dates=True)
                df     = pd.concat([df_old, df_new])
                df     = df[~df.index.duplicated(keep="first")]
                df.sort_index(inplace=True)
            else:
                df = df_new

            df.to_csv(path)
            print(f"✅ Saved {len(df)} rows → {path}")

def main():
    # 1️⃣ Get user inputs
    syms_input = input("Enter symbols (comma-separated, e.g. RELIANCE,TCS): ")
    symbols    = [s.strip().upper() for s in syms_input.split(",") if s.strip()]

    days_input = input("Enter number of past days to fetch (e.g. 500): ")
    days       = int(days_input.strip())

    intv_input = input("Enter intervals (comma-separated, e.g. 1minute,5minute): ")
    intervals  = [s.strip() for s in intv_input.split(",") if s.strip()]

    data_dir = input("Enter data directory (default 'data'): ").strip() or "data"

    # 2️⃣ Run fetch & store
    fetch_and_store(symbols, days, intervals, data_dir)

if __name__ == "__main__":
    main()
