# ============================================================================
#  algo/model.py
# ============================================================================
"""LSTM-based directional model – training, persistence, inference.
Includes metrics compile on load to silence absl warning, and suppresses absl logs."""

# Suppress TensorFlow absl warning about compiled metrics
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import os, logging
from typing import Tuple
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

MODEL_PATH = Path("models/lstm_model.h5")
LOOKBACK = 15

def _prepare_xy(df: pd.DataFrame):
    from .features import FEATURES
    X, y = [], []
    for i in range(LOOKBACK, len(df) - 1):      # predict next bar
        cols = ["close","ema_8","ema_21","atr","ret1","rsi14","macd","macds","obv","rv20",]
        X.append(df[cols].iloc[i - LOOKBACK : i].values)
        y.append(int(df["close"].iloc[i+1] > df["close"].iloc[i]))
    return np.asarray(X), np.asarray(y)

def _build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(df: pd.DataFrame, epochs: int = 20):
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    X, y = _prepare_xy(df)
    model = _build_model((LOOKBACK, X.shape[2]))
    es = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=64, validation_split=0.2, callbacks=[es], verbose=1)
    model.save(MODEL_PATH)
    logging.info("Model saved to %s", MODEL_PATH)
    return model


def load_or_train(df: pd.DataFrame, retrain: bool = False, epochs: int = 20):
    if not retrain and MODEL_PATH.exists():
        logging.info("Loading cached model …")
        model = load_model(MODEL_PATH, compile=False)
        # Build metrics to avoid absl warning
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    return train(df, epochs=epochs)
def predict_last(model, df: pd.DataFrame) -> float:
    """
    Given a trained model and a DataFrame with indicators,
    prepare the last LOOKBACK window and return P(up) for that bar.
    """
    # reuse your _prepare_xy to build X,y; we only need X
    X, _ = _prepare_xy(df)
    # Grab the most recent sample
    last_X = X[-1].reshape(1, LOOKBACK, X.shape[2])
    # model.predict returns array [[prob]]
    prob = model.predict(last_X, verbose=0)[0, 0]
    return float(prob)