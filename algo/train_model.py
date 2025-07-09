#!/usr/bin/env python3
"""
train_model.py

Central training pipeline:
 - loads feature-augmented CSV(s) from data_processed/
 - prepares sliding-window X, y
 - trains a GradientBoostingClassifier
 - reports validation AUC and feature importances
 - saves the final model to disk (train_model.pkl)
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from algo.exec_prog.data_handler import DataHandler
from features import ENABLED

LOOKBACK   = 60
MODELPATH  = Path(__file__).with_suffix(".pkl")

def make_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a DataFrame with raw OHLCV + your ENABLED indicators (already ffill'd),
    build X (n_samples x (LOOKBACK * n_feats)) and y (binary label: next-bar up/down).
    """
    # target: did close go up next bar?
    yser = df["close"].pct_change().shift(-1) > 0
    y_all = yser.astype(int)

    # features to include in window: 'close' + each enabled indicator
    feats = ["close"] + ENABLED
    arr   = df[feats].to_numpy(dtype="float64")

    # sliding window: shape = (n_windows, LOOKBACK, n_feats)
    windows = np.lib.stride_tricks.sliding_window_view(arr, (LOOKBACK, len(feats)))
    X_all   = windows.reshape(windows.shape[0], -1)           # flatten last two dims

    # we lose 1 label at the end because of shift(-1)
    X = X_all[:-1]
    y = y_all.iloc[LOOKBACK-1:-1].to_numpy()

    # drop any rows with NaNs in X
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("gb",      GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            verbose=1,
            random_state=0
        )),
    ])

def main(
    csvs: list[Path],
    test_size: float,
    retrain: bool
):
    # 1) Load + concat all processed CSVs via DataHandler
    dh   = DataHandler(raw_dir="data", proc_dir="data_processed")
    dfs  = []
    for p in csvs:
        # ensure the processed file exists (will build if needed)
        df = dh.load_features(symbol=p.stem.split("_")[0],
                              interval=p.stem.split("_")[1] + p.suffix.replace(".csv",""))
        dfs.append(df)
    data = pd.concat(dfs).sort_index()

    # 2) Load existing model or train new
    if MODELPATH.exists() and not retrain:
        model = joblib.load(MODELPATH)
        print(f"Loaded existing model from {MODELPATH}")
    else:
        X, y = make_Xy(data)
        Xtr, Xv, ytr, yv = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        print("🔧 Training model …")
        model = build_pipeline().fit(Xtr, ytr)

        probas = model.predict_proba(Xv)[:,1]
        auc    = roc_auc_score(yv, probas)
        print(f"✅ Validation AUC: {auc:.4f}")

        joblib.dump(model, MODELPATH)
        print(f"Model saved to {MODELPATH}")

    # 3) Feature importances
    gb = model.named_steps["gb"]
    fi = gb.feature_importances_
    n_feats = len(ENABLED) + 1  # +1 for 'close'
    mat     = fi.reshape(LOOKBACK, n_feats)
    impt    = mat.sum(axis=0)
    names   = ["close"] + ENABLED

    print("\nFeature importances (summed over lookback):")
    for name, score in zip(names, impt):
        print(f"  {name:<12} {score:.6f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train or load your intraday model"
    )
    p.add_argument(
        "--csvs", "-c", required=True,
        help="Comma-separated processed CSV paths, e.g. data_processed/RELIANCE_5minute_feat.csv"
    )
    p.add_argument(
        "--test-size", "-t", type=float, default=0.2,
        help="Fraction for validation split (default 0.2)"
    )
    p.add_argument(
        "--retrain", "-r", action="store_true",
        help="Force retraining even if model file exists"
    )
    args = p.parse_args()

    paths = [Path(x) for x in args.csvs.split(",")]
    main(paths, test_size=args.test_size, retrain=args.retrain)
