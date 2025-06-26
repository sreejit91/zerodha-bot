# download_hist.py
import json
import pathlib
from datetime import datetime, timedelta
import pandas as pd
from kiteconnect import KiteConnect

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"

# How far back?
DAILY_YEARS   = 3
INTRADAY_DAYS = 180

# Intervals to download
INTERVALS = {
    "day"      : DAILY_YEARS   * 365,  # ~3 years of daily bars
    "5minute"  : INTRADAY_DAYS  # ~6 months of 5m bars
}

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def load_config():
    cfg = json.loads(CONFIG_PATH.read_text())
    for key in ("api_key","api_secret","access_token","instrument_token"):
        if key not in cfg:
            raise KeyError(f"Missing `{key}` in {CONFIG_PATH}")
    return cfg

def chunks(start: datetime, end: datetime, delta_days: int):
    """Yield (from_dt, to_dt) slices of up to delta_days each."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=delta_days), end)
        yield cur, nxt
        cur = nxt + timedelta(seconds=1)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # 1) Load creds & instrument
    cfg = load_config()
    kite = KiteConnect(api_key=cfg["api_key"])
    kite.set_access_token(cfg["access_token"])
    token = int(cfg["instrument_token"])
    
    now = datetime.now()
    
    for interval, days_back in INTERVALS.items():
        print(f"\nDownloading `{interval}` bars for last {days_back} days…")
        start_dt = now - timedelta(days=days_back)
        
        all_dfs = []
        # Kite limits ~1000 bars/request, so slice into 30-day chunks:
        slice_days = 30 if interval != "day" else 365
        for frm, to in chunks(start_dt, now, slice_days):
            data = kite.historical_data(
                instrument_token=token,
                from_date=frm,
                to_date=to,
                interval=interval,
                continuous=False
            )
            df = pd.DataFrame(data)
            all_dfs.append(df)
            print(f"  • fetched {len(df)} bars {frm.date()}→{to.date()}")
        
        # concat, dedupe, sort
        full = pd.concat(all_dfs, ignore_index=True)
        full.drop_duplicates(subset="date", keep="first", inplace=True)
        full.sort_values("date", inplace=True)
        
        # save
        fname = BASE_DIR / f"hist_{token}_{interval}.csv"
        full.to_csv(fname, index=False)
        print(f"✅ Saved {len(full)} total bars → {fname}")
    
    print("\nAll done — you can now load these CSVs into your back-test notebook.")

if __name__ == "__main__":
    main()