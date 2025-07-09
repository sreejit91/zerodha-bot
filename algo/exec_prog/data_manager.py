#!/usr/bin/env python3
# algo/exec_prog/data_manager.py

import os
import sys
import datetime as dt
from typing import List, Optional

from algo.broker import KiteWrapper

def parse_symbols(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]

def parse_intervals(raw: str) -> List[str]:
    out = []
    for part in raw.split(","):
        ival = part.strip().lower()
        if not ival:
            continue
        # normalize "1minute" -> "minute"
        if ival == "1minute":
            ival = "minute"
        # normalize "1hour" -> "hour"
        elif ival == "1hour":
            ival = "hour"
        out.append(ival)
    return out

def prompt_date_range() -> Optional[tuple[dt.datetime, dt.datetime]]:
    rng = input("Enter date range (YYYY-MM-DD to YYYY-MM-DD), or blank to use ‘days’: ").strip()
    if not rng:
        return None
    try:
        start_str, end_str = [p.strip() for p in rng.split("to", 1)]
        start = dt.datetime.fromisoformat(start_str)
        # include the full end day until 23:59:59
        end   = dt.datetime.fromisoformat(end_str) + dt.timedelta(hours=23, minutes=59, seconds=59)
        return start, end
    except Exception:
        print("❌  Invalid format. Use e.g. 2025-07-01 to 2025-07-09")
        sys.exit(1)

def fetch_and_store(
    wrap: KiteWrapper,
    symbol: str,
    interval: str,
    data_dir: str,
    days: Optional[int] = None,
    from_to: Optional[tuple[dt.datetime, dt.datetime]] = None,
):
    if from_to:
        frm, to = from_to
        df = wrap.history(
            from_date=frm,
            to_date=to,
            interval=interval,
            tradingsymbol=symbol
        )
    else:
        df = wrap.history(
            days=days,
            interval=interval,
            tradingsymbol=symbol
        )

    os.makedirs(data_dir, exist_ok=True)
    fname = f"{symbol}_{interval}.csv"
    path = os.path.join(data_dir, fname)
    df.to_csv(path)
    print(f"✅ Saved {len(df)} rows → {path}")

def main():
    syms = parse_symbols(input("Enter symbols (comma-separated, e.g. RELIANCE,TCS): "))
    date_range = prompt_date_range()

    days: Optional[int]
    if date_range is None:
        days_str = input("Enter number of past days to fetch (e.g. 500): ").strip()
        if not days_str.isdigit() or int(days_str) <= 0:
            print("❌  Please enter a positive integer for days.")
            sys.exit(1)
        days = int(days_str)
    else:
        days = None

    raw_intervals = input("Enter intervals (comma-separated, e.g. 1minute,5minute,1hour): ")
    intervals = parse_intervals(raw_intervals)
    if not intervals:
        print("❌  No valid intervals provided.")
        sys.exit(1)

    data_dir = input("Enter data directory (default 'data'): ").strip() or "data"
    wrap = KiteWrapper()

    for symbol in syms:
        for interval in intervals:
            desc = f"{days} days" if days is not None else f"{date_range[0].date()}→{date_range[1].date()}"
            print(f"\n📥 Fetching {symbol} – {desc} @ {interval} …")
            fetch_and_store(wrap, symbol, interval, data_dir, days=days, from_to=date_range)

if __name__ == "__main__":
    main()
