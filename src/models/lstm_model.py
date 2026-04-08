# src/models/lstm_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src import config


@dataclass
class LSTMArtifacts:
    """
    Container for fitted LSTM artifacts.
    """
    model: keras.Model
    history: dict
    sequence_length: int
    n_features: int
    target_name: str


def build_lstm_regressor(
    sequence_length: int,
    n_features: int,
) -> keras.Model:
    """
    Build a simple LSTM regressor for next-month return prediction.
    """
    lstm_units = getattr(config, "LSTM_UNITS", 32)
    dense_units = getattr(config, "LSTM_DENSE_UNITS", 16)
    dropout = getattr(config, "LSTM_DROPOUT", 0.20)
    recurrent_dropout = getattr(config, "LSTM_RECURRENT_DROPOUT", 0.0)
    l2_value = getattr(config, "LSTM_L2", 1e-4)
    learning_rate = getattr(config, "LSTM_LEARNING_RATE", 1e-3)

    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),

        layers.LSTM(
            units=lstm_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_value),
            recurrent_regularizer=regularizers.l2(l2_value),
        ),

        layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_value),
        ),
        layers.Dropout(dropout),

        layers.Dense(1, activation="linear"),
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )
    return model


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a time-aware train/validation split using the final portion
    of the training data as validation.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    n_rows = len(X)
    split_idx = int(n_rows * (1 - validation_fraction))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Validation split produced an empty train or validation set.")

    return X_train, y_train, X_val, y_val


def fit_lstm(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    target_name: str,
) -> LSTMArtifacts:
    """
    Fit the LSTM model using a time-aware validation split.
    """
    if X_train_full.ndim != 3:
        raise ValueError("X_train_full must be 3D: (samples, sequence_length, n_features).")

    sequence_length = X_train_full.shape[1]
    n_features = X_train_full.shape[2]

    validation_fraction = getattr(config, "LSTM_VALIDATION_FRACTION", 0.15)

    X_train, y_train, X_val, y_val = split_train_validation(
        X=X_train_full,
        y=y_train_full,
        validation_fraction=validation_fraction,
    )

    model = build_lstm_regressor(
        sequence_length=sequence_length,
        n_features=n_features,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=getattr(config, "LSTM_EARLY_STOPPING_PATIENCE", 10),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=getattr(config, "LSTM_LR_PATIENCE", 5),
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=getattr(config, "LSTM_EPOCHS", 50),
        batch_size=getattr(config, "LSTM_BATCH_SIZE", 64),
        verbose=0,
        callbacks=callbacks,
    )

    return LSTMArtifacts(
        model=model,
        history=history.history,
        sequence_length=sequence_length,
        n_features=n_features,
        target_name=target_name,
    )


def predict_lstm(
    artifacts: LSTMArtifacts,
    X: np.ndarray,
) -> np.ndarray:
    """
    Predict next-month returns from LSTM sequence inputs.
    """
    if X.ndim != 3:
        raise ValueError("X must be 3D: (samples, sequence_length, n_features).")

    preds = artifacts.model.predict(X, verbose=0).reshape(-1)
    return preds.astype(float)