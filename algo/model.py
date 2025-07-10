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

LOOKBACK   = 60
_MODELPATH = pathlib.Path(__file__).with_suffix(".pkl")

# --------------------------- helpers ----------------------------------------
def _make_sliding_X(df: pd.DataFrame) -> np.ndarray:
    win  = len(FEATURES)
    arr  = df[FEATURES].to_numpy("float64")
    view = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, win))
    return view.reshape(view.shape[0], -1)

def _prepare_xy(df_feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = add_indicators(df_feat.copy())
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df["ret1"]   = df["close"].pct_change()

    X_all = _make_sliding_X(df)
    if X_all.shape[0] < 2:
        return np.empty((0, LOOKBACK * len(FEATURES))), np.empty((0,), dtype=int)

    X     = X_all[:-1]
    y_raw = df["ret1"].shift(-1)
    y     = (y_raw.iloc[LOOKBACK - 1 : -1] > 0).astype(int).to_numpy()

    # Use np.isfinite instead of just ~np.isnan for NaN and Inf
    mask  = np.isfinite(X).all(axis=1)
    return X[mask], y[mask]


def _build_pipe() -> Pipeline:
    lgb_params = dict(
        n_estimators     = 600,
        learning_rate    = 0.05,
        num_leaves       = 31,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 0,
        n_jobs           = -1,
        metric           = "auc",
        verbose          = -1,        # OK *here* (constructor)
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("lgb",     lgb.LGBMClassifier(**lgb_params)),
    ])

# ------------------------ train / load --------------------------------------
def load_or_train(df: pd.DataFrame, retrain: bool = False) -> Pipeline:
    if _MODELPATH.exists() and not retrain:
        return joblib.load(_MODELPATH)

    X, y           = _prepare_xy(df)
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
def predict_last(df_feat: pd.DataFrame, model: Pipeline) -> float:
    df = add_indicators(df_feat.copy())
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    arr = df[FEATURES].iloc[-LOOKBACK:].to_numpy("float64")
    X_last = arr.flatten().reshape(1, -1)
    return float(model.predict_proba(X_last)[0, 1])
