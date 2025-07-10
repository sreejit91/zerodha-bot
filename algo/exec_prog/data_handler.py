#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
from algo.features import add_indicators

def resolve_paths():
    # __file__ → .../zerodha-bot/algo/exec_prog/data_handler.py
    # parent → .../zerodha-bot/algo/exec_prog
    # parent.parent → .../zerodha-bot/algo
    base = Path(__file__).resolve().parent.parent

    raw_dir  = base / "data"
    proc_dir = base / "data_processed"

    if not raw_dir.is_dir() or not proc_dir.is_dir():
        print(f"❌ Could not find data folders under {base!s}")
        sys.exit(1)

    return raw_dir, proc_dir

def main():
    raw_dir, proc_dir = resolve_paths()

    sym      = input("Enter symbol (e.g. RELIANCE): ").strip().upper()
    interval = input("Enter interval (e.g. 3minute, 5minute): ").strip()

    raw_fn  = raw_dir  / f"{sym}_{interval}.csv"
    proc_fn = proc_dir / f"{sym}_{interval}_feat.csv"

    if not raw_fn.exists():
        print(f"❌ Raw file not found: {raw_fn!s}")
        sys.exit(1)

    print(f"🔍 Loading raw bars from {raw_fn.name} …")
    df = pd.read_csv(raw_fn, index_col=0, parse_dates=True).sort_index()

    print(f"⚙️   Computing indicators …")
    df_feat = add_indicators(df).ffill()

    print(f"💾 Saving to {proc_fn.name} …")
    df_feat.to_csv(proc_fn)

    print(f"✅ Done — features written to {proc_fn!s}")

if __name__ == "__main__":
    main()
