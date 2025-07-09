# backtest_simple.py

from algo import KiteWrapper
from algo import load_config
from algo import add_indicators
from algo import load_or_train_simple, predict_last_simple
from algo import backtest

# 1️⃣ Parameters
SYMBOL     = "HDFCBANK"
INTERVAL   = "3minute"
TRAIN_DAYS = 180
TEST_DAYS  = 20

CAPITAL    = 100_000
SL_PCT     = 0.0010
TP_PCT     = 0.0025
TRAIL_PCT  = 0.0020
HOLD_MAX   = 7

UPPER_PROB = 0.6    # your chosen thresholds
LOWER_PROB = 0.4

# 2️⃣ Fetch & feature-engineer train (though simple_model needs none)
cfg        = load_config()
broker     = KiteWrapper(cfg)

hist_train = broker.history(
    days=TRAIN_DAYS, interval=INTERVAL, tradingsymbol=SYMBOL
)
df_train = add_indicators(hist_train).ffill().bfill()

# 3️⃣ Instantiate the simple “model”
model = load_or_train_simple(df_train)

# 4️⃣ Fetch & feature-engineer test
hist_test = broker.history(
    days=TEST_DAYS, interval=INTERVAL, tradingsymbol=SYMBOL
)
df_test = add_indicators(hist_test).ffill().bfill()

# 5️⃣ Run backtest using the simple rule-based predictor
trades, metrics = backtest(
    df_test,
    model         = model,
    capital       = CAPITAL,
    contract_size = 1,
    sl_pct        = SL_PCT,
    tp_pct        = TP_PCT,
    trail_pct     = TRAIL_PCT,
    hold_max      = HOLD_MAX,
    upper         = UPPER_PROB,
    lower         = LOWER_PROB,
    predict_fn    = predict_last_simple,
)

# 6️⃣ Inspect results
print("Back-test metrics:", metrics)
print("\nFirst few trades:\n", trades.head())
