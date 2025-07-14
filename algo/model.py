# algo/model.py  ◆ LightGBM + early-stopping ◆
from __future__ import annotations
import pathlib, time, joblib
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb                              # pip install lightgbm

from algo.features import FEATURES, add_indicators  # your indicators list

LOOKBACK   = 30
HORIZON = 5
_MODELPATH = pathlib.Path(__file__).with_suffix(".pkl")

# --------------------------- helpers ----------------------------------------
def _make_sliding_X(df: pd.DataFrame) -> np.ndarray:
    win = len(FEATURES)
    arr = df[FEATURES].to_numpy("float64")
    view = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, win))
    return view.reshape(view.shape[0], -1)



def _prepare_xy(df: pd.DataFrame, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
    df = df.copy()

    # Step 1: Forward return and volatility
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["volatility"] = df["close"].pct_change().rolling(20).std()

    # Step 2: Added feature: Price vs VWAP
    df["price_vs_vwap"] = df["close"] - df.get("vwap", df["close"])

    # ✅ NEW FEATURES
    df["trend_strength"] = (df["macd"] - df["macd_signal"]).abs()
    df["vwap_gap"] = df["close"] - df.get("vwap", df["close"])
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-6)

    # Step 3: Label thresholds
    return_threshold = 0.0007 * horizon
    vol_threshold = 0.0003

    long_mask = (df["future_return"] > return_threshold) & (df["volatility"] > vol_threshold)
    short_mask = (df["future_return"] < -return_threshold) & (df["volatility"] > vol_threshold)

    df = df[long_mask | short_mask].copy()
    df["label"] = 0
    df.loc[long_mask, "label"] = 1

    # Step 4: Ensure required features are in list
    extra_feats = ["price_vs_vwap", "trend_strength", "vwap_gap", "bb_position"]
    for feat in extra_feats:
        if feat not in FEATURES:
            FEATURES.append(feat)

    # Step 5: Drop bad rows and create X, y
    df.dropna(subset=FEATURES + ["label"], inplace=True)
    X_all = _make_sliding_X(df)[:-horizon]
    y_all = df["label"].iloc[LOOKBACK - 1 : -horizon].to_numpy()

    # Step 6: Final filter
    mask = np.isfinite(X_all).all(axis=1)
    print(f"Prepared {len(y_all[mask])} samples | Class balance (mean): {np.mean(y_all[mask]):.3f}")
    return X_all[mask], y_all[mask]




def _build_pipe() -> Pipeline:
    lgb_params = dict(
    n_estimators= 1050,
    learning_rate= 0.017537607442521964,
    num_leaves= 76,
    max_depth= 5,
    min_child_samples= 96,
    subsample= 0.9284965223502559,
    colsample_bytree= 0.7023760690478573,
    reg_alpha= 1.6898034700530933,
    reg_lambda= 1.9674841696814904,
    random_state= 0,
    n_jobs= -1,
    metric= 'auc',
    verbose= -1
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("lgb",     lgb.LGBMClassifier(**lgb_params)),
    ])

# ------------------------ train / load --------------------------------------
def load_or_train(df: pd.DataFrame, retrain: bool = False, horizon =5) -> Pipeline:
    if _MODELPATH.exists() and not retrain:
        return joblib.load(_MODELPATH)

    X, y           = _prepare_xy(df, horizon=horizon)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, shuffle=False)

    pipe = _build_pipe()
    fit_params = dict(
        lgb__eval_set  = [(Xval, yval)],
        lgb__callbacks = [
            lgb.callback.early_stopping(50, verbose=False)
        ]
    )

    print("🔧  Training started …")
    t0 = time.time()
    pipe.fit(Xtr, ytr, **fit_params)          # <-- NO verbose kw-arg here
    print(f"✅  Finished in {time.time()-t0:.1f}s   "
          f"(best_iter = {pipe.named_steps['lgb'].best_iteration_}, "
          f"best_AUC = {pipe.named_steps['lgb'].best_score_['valid_0']['auc']:.4f})")

    # quick reference accuracy (optional)
    acc = accuracy_score(yval, pipe.predict(Xval))
    print(f"Hold-out accuracy: {acc:.3f}")

    joblib.dump(pipe, _MODELPATH)
    return pipe

# -------------------------- inference ---------------------------------------
def predict_last(df_window: pd.DataFrame, model: Pipeline) -> float:
    from algo.features import FEATURES

    LOOKBACK = model.n_features_in_ // len(FEATURES)

    if len(df_window) < LOOKBACK:
        return np.nan  # too short

    df_window = df_window.copy()
    df_window[FEATURES] = df_window[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = df_window[FEATURES].iloc[-LOOKBACK:]

    if X.isnull().values.any():
        return np.nan

    arr = X.to_numpy("float64").flatten().reshape(1, -1)
    return float(model.predict_proba(arr)[0, 1])




