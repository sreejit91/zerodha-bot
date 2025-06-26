"""
algo/model.py
Unified ML pipeline for BOTH back-test and live trading.
"""

from __future__ import annotations
import pathlib, joblib
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .features import FEATURES

# ─────────────── constants ───────────────────────────────────────────
LOOKBACK   = 60                         # bars per sample (60 × 3 min ≈ 3 h)
_MODELPATH = pathlib.Path(__file__).with_suffix(".pkl")


# ─────────────── helpers ─────────────────────────────────────────────
def _ensure_ret1(df: pd.DataFrame) -> pd.DataFrame:
    if "ret1" not in df:
        df = df.copy()
        df["ret1"] = df["close"].pct_change()
    return df


def _make_sliding_X(df: pd.DataFrame) -> np.ndarray:
    """Return 2-D array of flattened look-back windows."""
    arr = df[FEATURES].to_numpy()
    n, m = arr.shape
    if n < LOOKBACK:
        return np.empty((0, LOOKBACK * m))
    out = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, m))[:, 0, :]
    return out.reshape(out.shape[0], -1)


def _prepare_xy(df_feat: pd.DataFrame):
    """
    Build (X, y) for classification.  Coerces FEATURE cols to float so that
    `np.isnan` works even if any string / pd.NA slips in.
    """
    df_feat = _ensure_ret1(df_feat)
    df_feat[FEATURES] = df_feat[FEATURES].apply(pd.to_numeric, errors="coerce")

    X = _make_sliding_X(df_feat).astype("float64", copy=False)

    y = (
        df_feat["ret1"]
        .shift(-1)
        .iloc[LOOKBACK - 1 :]
        .gt(0)
        .astype("int8")
        .to_numpy()
    )

    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def _build_pipe() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(n_estimators=300, max_depth=3)),
        ]
    )


# ─────────────── public API ──────────────────────────────────────────
def load_or_train(df_feat: pd.DataFrame, *, retrain: bool = False) -> Pipeline:
    """
    Load cached model or (re)train on *df_feat*.
    """
    if _MODELPATH.exists() and not retrain:
        return joblib.load(_MODELPATH)

    X, y   = _prepare_xy(df_feat)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    pipe = _build_pipe().fit(X_tr, y_tr)

    acc = accuracy_score(y_val, pipe.predict(X_val))
    print(f"Validation accuracy: {acc:.3f}  (n={len(y_val)})")

    joblib.dump(pipe, _MODELPATH)
    return pipe


def predict_last(df_feat: pd.DataFrame, model: Pipeline) -> float:
    """
    Return P(up) for the **latest bar**.
    """
    df_feat = _ensure_ret1(df_feat)
    X_last  = df_feat[FEATURES].tail(LOOKBACK).to_numpy().reshape(1, -1)
    return float(model.predict_proba(X_last)[0, 1])
