from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from algo.features import FEATURES, add_indicators

LOOKBACK  = 40
MODELPATH = pathlib.Path(__file__).with_name("hist_model.pkl")


def _make_sliding_X(df: pd.DataFrame) -> np.ndarray:
    """Return sliding-window feature matrix (n_samples × LOOKBACK·n_features)."""
    n_feat = len(FEATURES)
    arr = df[FEATURES].to_numpy(dtype="float64")
    sw = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, n_feat))
    return sw.reshape(sw.shape[0], -1)


def load_or_train_hist(df_raw: pd.DataFrame, retrain: bool = False) -> Pipeline:
    """
    Train or load a HistGradientBoosting model.
    """
    if MODELPATH.exists() and not retrain:
        return joblib.load(MODELPATH)

    df = add_indicators(df_raw.copy())
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    X_all = _make_sliding_X(df)
    y_all = (df['ret1'].shift(-1).iloc[LOOKBACK - 1:] > 0).astype(int).values

    # filter out NaN windows
    mask = ~np.isnan(X_all).any(axis=1)
    X, y = X_all[mask], y_all[mask]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('hgb', HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=3,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )),
    ])
    pipe.fit(X_tr, y_tr)

    acc = accuracy_score(y_val, pipe.predict(X_val))
    print(f"[hist_model] Validation accuracy: {acc:.3f} (n={len(y_val)})")

    joblib.dump(pipe, MODELPATH)
    return pipe


def predict_last_hist(df_raw: pd.DataFrame, model: Pipeline) -> float:
    """
    Return P(up) for the most recent bar using the hist_model.
    """
    df = add_indicators(df_raw.copy())
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    arr = df[FEATURES].tail(LOOKBACK).to_numpy(dtype="float64")
    X_last = arr.flatten().reshape(1, -1)
    return float(model.predict_proba(X_last)[0, 1])
