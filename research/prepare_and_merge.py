# research/prepare_and_merge.py

import pandas as pd
from pathlib import Path

from algo import KiteWrapper, _resolve_token
from research.config   import load_config
from research.features import add_indicators


def main():
    # ── 1) Locate today's raw tick file ────────────────────────────────
    files = sorted(Path(".").glob("imb_tape_*.csv"))
    if not files:
        print("❌ No 'imb_tape_YYYY-MM-DD.csv' file found in this folder.")
        print("   Run your imbalance_collector first and let it gather ticks.")
        return
    raw_path = files[-1]   # most recent by name
    print("✔️  Found tick tape:", raw_path.name)

    # ── 2) Load & resample to 3-min bars ───────────────────────────────
    df_raw = pd.read_csv(
        raw_path,
        parse_dates=["ts_utc"],
        index_col="ts_utc"
    )
    imb3 = (
        df_raw["imb"]
        .resample("3T")                  # 3-minute bins
        .agg(["mean","std"])
        .rename(columns={"mean":"imb_mean",
                         "std":"imb_std"})
    )
    print(f"✔️  Resampled → {len(imb3)} bars (should be ≤125)")

    # ── 3) Save the resampled file ────────────────────────────────────
    date_part = raw_path.stem.split("_")[-1]
    out_3m = Path(f"imb_3m_{date_part}.csv")
    imb3.to_csv(out_3m, index_label="ts_utc")
    print("✔️  3-min imbalance saved:", out_3m.name)

    # ── 4) Convert imbalance index UTC → naive IST ───────────────────
    imb3.index = (
        imb3.index
             .tz_localize("UTC")
             .tz_convert("Asia/Kolkata")
             .tz_localize(None)
    )
    print("✔️  Index converted to IST; first few times:\n", imb3.head().index)

    # ── 5) Fetch price history ────────────────────────────────────────
    cfg    = load_config()
    broker = KiteWrapper(cfg)
    hist   = broker.history(
        days=180,
        interval="3minute",
        tradingsymbol=cfg.tradingsymbol,
    )
    print(f"✔️  Loaded price history: {len(hist)} bars")

    # ── 6) Merge & build features ─────────────────────────────────────
    df = (
        hist
        .join(imb3, how="left")            # left-join keeps all price bars
        .pipe(add_indicators, debug=True)  # add your FEATURES
        .ffill()                           # fill early NaNs
    )
    print("✔️  Merged price + imbalance; sample columns:\n", df.columns[:10])

    # ── 7) Save merged DataFrame ──────────────────────────────────────
    merged_path = Path(f"merged_{cfg.tradingsymbol}_imb_3m_{date_part}.csv")
    df.to_csv(merged_path, index_label="timestamp")
    print("✔️  Saved merged data →", merged_path.name)

if __name__ == "__main__":
    main()
