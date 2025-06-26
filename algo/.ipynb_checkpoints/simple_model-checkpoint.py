"""Lightweight logistic-regression model (<2 s fit)."""
from __future__ import annotations
import joblib, logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import add_indicators, FEATURES
from .model    import LOOKBACK, _prepare_xy
FEATURES = [
    "close","ema_8","ema_21","atr","ret1",
    "rsi14","macd","macds","obv","rv20",
]
MODEL_PATH = Path("models/logreg.joblib")

def load_or_train(df, retrain: bool = False):
    if MODEL_PATH.exists() and not retrain:
        logging.info("Loading cached logistic model …")
        return joblib.load(MODEL_PATH)

    X, y = _prepare_xy(add_indicators(df))
    X = X.reshape(len(X), -1)        # 30×10 → 300-D

    clf = make_pipeline(
        StandardScaler(),                 # scales each of the 300 numeric inputs
        LogisticRegression(
            max_iter=5000,                # gives lbfgs room to converge
            solver="lbfgs",
            class_weight="balanced",      # helps if y is slightly imbalanced
        )
    )
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    logging.info("Saved logistic model → %s", MODEL_PATH)
    return clf

def predict_last(df_feat, model) -> float:
    X_last = (
        df_feat[FEATURES]
        .tail(LOOKBACK)
        .values
        .reshape(1, -1)
    )
    return float(model.predict_proba(X_last)[0, 1])   # P(up)