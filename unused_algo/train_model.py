"""
train_model.py
────────────────────────────────────────────────────────────────────────
Retrains the intraday scalper model:

• Pulls the last 180 days of 3-minute candles for the symbol you set.
• Feature-engineers with algo.features.
• Fits / overwrites algo/model.pkl (Gradient-Boosting + scaler).
• Prints basic diagnostics so you know it worked.

Run this once after market close (or schedule it nightly).
"""

from __future__ import annotations
import datetime as dt

from algo import KiteWrapper
from algo import load_config
from algo import add_indicators
from algo import load_or_train, predict_last

# ── parameters you can tweak ────────────────────────────────────────
SYMBOL      = "HDFCBANK"     # NSE symbol you trade
INTERVAL    = "3minute"      # candle size
TRAIN_DAYS  = 180            # rolling window length
RETRAIN     = True           # force re-fit even if pickle exists
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("🚀  Training started —", dt.datetime.now().strftime("%d %b %Y %H:%M:%S"))

    # 1️⃣  Connect & download
    cfg   = load_config()
    kite  = KiteWrapper(cfg)
    hist  = kite.history(days=TRAIN_DAYS,
                         interval=INTERVAL,
                         tradingsymbol=SYMBOL)
    print(f"Fetched {len(hist):,} bars  "
          f"[{hist.index.min()} → {hist.index.max()}]")

    # 2️⃣  Feature-engineering
    df_feat = add_indicators(hist)
    print("Feature matrix shape:", df_feat.shape)

    # 3️⃣  Train (or overwrite) the model pickle
    model = load_or_train(df_feat, retrain=RETRAIN)

    # 4️⃣  Quick sanity check
    p_last = predict_last(df_feat, model)
    print(f"Sample P(up) on last bar: {p_last:.3f}")

    print("✅  Model saved ➜", (load_or_train.__module__ + ".pkl"))


if __name__ == "__main__":
    main()
