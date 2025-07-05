from __future__ import annotations
import pathlib
import joblib
import numpy as np
import pandas as pd
import time

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .features import FEATURES, add_indicators

LOOKBACK    = 60
_MODELPATH  = pathlib.Path(__file__).with_suffix(".pkl")


def _make_sliding_X(df: pd.DataFrame) -> np.ndarray:
    """n_windows × (LOOKBACK * n_features)"""
    win = len(FEATURES)
    arr = df[FEATURES].to_numpy(dtype="float64")
    vw = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, win))
    return vw.reshape(vw.shape[0], -1)


def _prepare_xy(df_feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = add_indicators(df_feat.copy())
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df.loc[:, "ret1"] = df["close"].pct_change()
    X_all = _make_sliding_X(df)
    if X_all.shape[0] < 2:
        return np.empty((0, LOOKBACK * len(FEATURES))), np.empty((0,), dtype=int)
    X = X_all[:-1]
    yser = df["ret1"].shift(-1)
    y = (yser.iloc[LOOKBACK - 1 : -1] > 0).astype(int).to_numpy()
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def _build_pipe() -> Pipeline:
    """
    Builds a scikit-learn pipeline with verbosity on the GB classifier.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=300,
                max_depth=3,
                verbose=1,        # show training progress
                random_state=0
            )
        ),
    ])


def load_or_train(df: pd.DataFrame, retrain: bool = False) -> Pipeline:
    """
    Load existing model or train a new one.
    Wraps fit() with timing prints so you can see training start/finish.

    In a Jupyter notebook, you can also use:
        %time model = load_or_train(df, retrain=True)
    """
    if _MODELPATH.exists() and not retrain:
        return joblib.load(_MODELPATH)

    X, y = _prepare_xy(df)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("🔧  Training started at", time.strftime("%H:%M:%S"))
    start = time.time()
    pipe = _build_pipe().fit(Xtr, ytr)
    end = time.time()
    print(f"✅  Training finished at {time.strftime('%H:%M:%S')}  (took {end-start:.1f}s)")

    acc = accuracy_score(yv, pipe.predict(Xv))
    print(f"Validation accuracy: {acc:.3f}")
    joblib.dump(pipe, _MODELPATH)
    return pipe


def predict_last(df_feat: pd.DataFrame, model: Pipeline) -> float:
    """Return P(up) on the **latest** bar."""
    df = add_indicators(df_feat.copy())
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df.loc[:, "ret1"] = df["close"].pct_change()
    arr = df[FEATURES].iloc[-LOOKBACK:].to_numpy(dtype="float64")
    X_last = arr.flatten().reshape(1, -1)
    return float(model.predict_proba(X_last)[0, 1])
