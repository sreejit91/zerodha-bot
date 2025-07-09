# algo/reg_model.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from algo.features import FEATURES

# Number of bars needed to form one training sample
LOOKBACK = 40

# Where to save/load the trained model
MODEL_PATH = Path(__file__).parent / "reg_model.pkl"


def _make_sliding_X(df_feat: pd.DataFrame) -> np.ndarray:
    arr = df_feat[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    n_rows, n_feats = arr.shape
    if n_rows == 0:
        return np.empty((0, 0))
    window = min(LOOKBACK, n_rows)
    if n_rows < LOOKBACK:
        print(f"[reg_model] Warning: only {n_rows} rows; reducing lookback to {window}")
    sw = np.lib.stride_tricks.sliding_window_view(arr, (window, n_feats))
    return sw.reshape(sw.shape[0], -1)


def load_or_train_reg(df_raw: pd.DataFrame, retrain: bool = False) -> LogisticRegression:
    """
    Train or load a logistic regression model using df_raw.
    Attaches training metrics as `model.training_metrics` dict.
    Returns the trained `LogisticRegression` model.
    """
    df = df_raw.copy().ffill()
    df['ret1'] = df['close'].pct_change().fillna(0)

    X_all = _make_sliding_X(df)
    if X_all.size == 0:
        raise ValueError(f"No feature windows generated (shape {X_all.shape}). Increase data or reduce LOOKBACK.")

    y_series = (df['ret1'].shift(-1) > 0).astype(int)
    y_all = y_series.iloc[-len(X_all):].values

    mask = ~np.isnan(X_all).any(axis=1)
    X_all, y_all = X_all[mask], y_all[mask]
    if X_all.shape[0] == 0:
        raise ValueError("All feature windows contain NaN. Check indicator setup or data quality.")

    n = X_all.shape[0]
    split = max(1, int(n * 0.8)) if n > 1 else 1
    X_train, y_train = X_all[:split], y_all[:split]

    # Load existing model if not retraining
    if not retrain:
        try:
            model = joblib.load(MODEL_PATH)
            model.training_metrics = {}
            print(f"[reg_model] Loaded model from {MODEL_PATH.resolve()}")
            return model
        except FileNotFoundError:
            print("[reg_model] No existing model found; training a new one.")

    # Ensure directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"[reg_model] Trained new model and saved to {MODEL_PATH.resolve()}")

    # Compute training metrics
    y_pred = model.predict(X_train)
    metrics = {
        'accuracy': accuracy_score(y_train, y_pred),
        'precision': precision_score(y_train, y_pred, zero_division=0),
        'recall': recall_score(y_train, y_pred, zero_division=0),
        'f1': f1_score(y_train, y_pred, zero_division=0),
    }
    print(f"[reg_model] Training metrics: {metrics}")
    model.training_metrics = metrics
    return model


def predict_last(df_window: pd.DataFrame, model: LogisticRegression) -> float:
    X = df_window[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64").reshape(1, -1)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=1)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    return model.predict_proba(X)[0, 1]
