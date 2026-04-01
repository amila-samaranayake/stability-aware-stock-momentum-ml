# src/models/nn_mlp.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src import config


@dataclass
class MLPArtifacts:
    """
    Container for fitted MLP artifacts.
    """
    scaler: object
    model: keras.Model
    feature_cols: list[str]
    target_col: str
    history: dict


def prepare_xy(
    df_long: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a long ML dataframe into feature matrix X and target vector y.
    """
    X = df_long[feature_cols].to_numpy(dtype=float)
    y = df_long[target_col].to_numpy(dtype=float)
    return X, y


def build_mlp_regressor(input_dim: int) -> keras.Model:
    """
    Build the configured MLP regression model.
    """
    l2_value = getattr(config, "NN_L2", 1e-4)
    dropout_1 = getattr(config, "NN_DROPOUT_1", 0.20)
    dropout_2 = getattr(config, "NN_DROPOUT_2", 0.15)
    dropout_3 = getattr(config, "NN_DROPOUT_3", 0.10)

    hidden_1 = getattr(config, "NN_HIDDEN_1", 64)
    hidden_2 = getattr(config, "NN_HIDDEN_2", 32)
    hidden_3 = getattr(config, "NN_HIDDEN_3", 16)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(
            hidden_1,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_value),
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_1),

        layers.Dense(
            hidden_2,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_value),
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_2),

        layers.Dense(
            hidden_3,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_value),
        ),
        layers.Dropout(dropout_3),

        layers.Dense(1, activation="linear"),
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=getattr(config, "NN_LEARNING_RATE", 1e-3)
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_mlp_with_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> MLPArtifacts:
    """
    Fit the scaler and MLP model using a time-aware validation split.
    """
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)

    scaler_type = getattr(config, "SCALER_TYPE", "robust").lower()
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    val_fraction = getattr(config, "NN_VALIDATION_FRACTION", 0.15)
    n_rows = len(X_train_scaled)
    split_idx = int(n_rows * (1 - val_fraction))

    X_tr = X_train_scaled[:split_idx]
    y_tr = y_train[:split_idx]
    X_val = X_train_scaled[split_idx:]
    y_val = y_train[split_idx:]

    model = build_mlp_regressor(input_dim=X_train_scaled.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=getattr(config, "NN_EARLY_STOPPING_PATIENCE", 10),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=getattr(config, "NN_LR_PATIENCE", 5),
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=getattr(config, "NN_EPOCHS", 100),
        batch_size=getattr(config, "NN_BATCH_SIZE", 128),
        verbose=0,
        callbacks=callbacks,
    )

    return MLPArtifacts(
        scaler=scaler,
        model=model,
        feature_cols=feature_cols,
        target_col=target_col,
        history=history.history,
    )


def predict_returns(
    artifacts: MLPArtifacts,
    df_long: pd.DataFrame,
    pred_col: str = "pred_return",
) -> pd.DataFrame:
    """
    Predict returns for each row of a long ML dataframe.
    """
    X, _ = prepare_xy(df_long, artifacts.feature_cols, artifacts.target_col)
    X_scaled = artifacts.scaler.transform(X)
    preds = artifacts.model.predict(X_scaled, verbose=0).reshape(-1)

    out = df_long.copy()
    out[pred_col] = preds
    return out