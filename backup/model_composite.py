# -------------------------------------------------------------------------
# PART 1 ▸ MODEL (XGBoost)
# -------------------------------------------------------------------------
from __future__ import annotations
import pathlib, time, joblib
from typing import Tuple, List
import numpy as np
import pandas as pd
from xgboost import XGBClassifier                    # pip install xgboost>=2.0
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from algo.features import FEATURES, add_indicators   # keep using your feature set

LOOKBACK   = 60   # match your backtester default
HORIZON    = 5    # forward bars to evaluate outcome
_MODELPATH = pathlib.Path(__file__).with_suffix(".pkl")

# --------------------------- helpers --------------------------------------

def _make_sliding_X(df: pd.DataFrame) -> np.ndarray:
    win = len(FEATURES)
    arr = df[FEATURES].astype("float64").to_numpy()
    view = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, win))
    return view.reshape(view.shape[0], -1)


def _prepare_xy(df: pd.DataFrame, horizon: int = HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    """Binary label: +1 if next‑horizon return > ATR × 0.25  else 0 (short)."""
    df = df.copy()

    # Basic ATR for target sizing (n = 14)
    high, low, close = df["high"], df["low"], df["close"]
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    df["atr"] = tr.rolling(14).mean()

    # Forward return
    df["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1

    # Threshold = 25 % of current ATR
    thresh = (df["atr"] / df["close"] * 0.25).fillna(0)

    long_mask  = df["future_ret"] >  thresh
    short_mask = df["future_ret"] < -thresh

    df = df[long_mask | short_mask].copy()
    df["label"] = np.where(long_mask.loc[df.index], 1, 0)

    # Cleanup
    df.dropna(subset=FEATURES + ["label"], inplace=True)

    X_all = _make_sliding_X(df)[:-horizon]
    y_all = df["label"].iloc[LOOKBACK - 1 : -horizon].to_numpy()

    mask = np.isfinite(X_all).all(axis=1)
    print(f"Prepared {mask.sum()} samples | Class balance (mean): {y_all[mask].mean():.3f}")
    return X_all[mask], y_all[mask]


# ------------------------ pipeline ----------------------------------------

def _build_pipe() -> Pipeline:
    xgb_params = dict(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=0,
    )
    return Pipeline([
        ("imp",   SimpleImputer(strategy="mean")),
        ("xgb",   XGBClassifier(**xgb_params)),
    ])


# ------------------------ train / load ------------------------------------

def load_or_train_xgb(df: pd.DataFrame, retrain: bool = False, horizon: int = HORIZON):
    if _MODELPATH.exists() and not retrain:
        return joblib.load(_MODELPATH)

    X, y = _prepare_xy(df, horizon)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, shuffle=False)

    pipe = _build_pipe()
    fit_params = dict(xgb__eval_set=[(Xval, yval)], xgb__verbose=False, xgb__early_stopping_rounds=50)

    print("🔧  Training started …")
    t0 = time.time()
    pipe.fit(Xtr, ytr, **fit_params)
    elapsed = time.time() - t0
    auc  = roc_auc_score(yval, pipe.predict_proba(Xval)[:, 1])
    acc  = accuracy_score(yval, pipe.predict(Xval))
    print(f"✅  Finished in {elapsed:.1f}s | AUC={auc:.4f}  Acc={acc:.3f}")

    joblib.dump(pipe, _MODELPATH)
    return pipe


# -------------------------- inference -------------------------------------

def predict_last_xgb(df_window: pd.DataFrame, model: Pipeline):
    n_feat = model.n_features_in_ // len(FEATURES)
    if len(df_window) < n_feat:
        return np.nan
    X = df_window[FEATURES].iloc[-n_feat:].to_numpy("float64").flatten().reshape(1, -1)
    if np.isnan(X).any():
        return np.nan
    return float(model.predict_proba(X)[0, 1])