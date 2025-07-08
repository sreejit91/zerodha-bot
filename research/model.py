from __future__ import annotations
import pathlib, time, joblib, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── NEW: use LightGBM instead of GradientBoosting ────────────────────
import lightgbm as lgb                         # pip install lightgbm

from .features import FEATURES, add_indicators

LOOKBACK   = 60      # bars in each feature window
N_AHEAD    = 10      # ── predict CLOSE↑ over next 10 bars (≈30 min)
_MODELPATH = pathlib.Path(__file__).with_suffix(".pkl")

# ─────────────────────────────────────────────────────────────────────
def _make_sliding_X(df: pd.DataFrame) -> np.ndarray:
    win = len(FEATURES)
    arr = df[FEATURES].to_numpy(dtype="float64")
    vw  = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, win))
    return vw.reshape(vw.shape[0], -1)         # n_windows × (LOOKBACK·n_feat)


def _prepare_xy(df_feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df  = add_indicators(df_feat.copy())
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    X_all = _make_sliding_X(df)
    if X_all.shape[0] < 2:
        return np.empty((0, LOOKBACK * len(FEATURES))), np.empty((0,), dtype=int)

    # target = 1 if close will be higher in N_AHEAD bars
    yser = (df["close"].shift(-N_AHEAD) > df["close"]).astype(int)
    y    = yser.iloc[LOOKBACK - 1 : -N_AHEAD].to_numpy()

    X = X_all[:-N_AHEAD]                       # align X and y
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def _build_pipe() -> Pipeline:
    """LightGBM pipeline (much stronger on noisy intraday data)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler(with_mean=False)),
        ("gbm", lgb.LGBMClassifier(
            n_estimators=600,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.8,
            random_state=0,
            verbose=-1,
        )),
    ])


def load_or_train(df: pd.DataFrame, retrain: bool = False) -> Pipeline:
    """
    Load existing model (if compatible) or train anew.
    Set retrain=True whenever FEATURES / LOOKBACK / N_AHEAD change.
    """
    if _MODELPATH.exists() and not retrain:
        try:
            return joblib.load(_MODELPATH)
        except Exception:
            print("⚠️  Old model incompatible; retraining…")

    X, y = _prepare_xy(df)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("🔧  Training started at", time.strftime("%H:%M:%S"))
    start = time.time()
    pipe  = _build_pipe().fit(Xtr, ytr)
    print(f"✅  Training finished in {time.time()-start:.1f}s")

    acc = accuracy_score(yv, pipe.predict(Xv))
    print(f"Validation accuracy: {acc:.3f}")
    joblib.dump(pipe, _MODELPATH)
    return pipe


def predict_last(df_feat: pd.DataFrame, model: Pipeline) -> float:
    """Return P(up) N_AHEAD bars ahead on the latest bar."""
    df = add_indicators(df_feat.copy())
    arr = df[FEATURES].iloc[-LOOKBACK:].to_numpy(dtype="float64")
    X_last = arr.flatten().reshape(1, -1)
    return float(model.predict_proba(X_last)[0, 1])
