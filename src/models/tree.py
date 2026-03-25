# src/models/tree.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src import config


@dataclass
class RandomForestArtifacts:
    scaler: object
    model: RandomForestRegressor
    feature_cols: list[str]
    target_col: str


def prepare_xy(
    df_long: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X = df_long[feature_cols].to_numpy(dtype=float)
    y = df_long[target_col].to_numpy(dtype=float)
    return X, y


def fit_random_forest_with_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> RandomForestArtifacts:
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)

    scaler_type = getattr(config, "SCALER_TYPE", "robust").lower()
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=getattr(config, "RF_N_ESTIMATORS", 300),
        max_depth=getattr(config, "RF_MAX_DEPTH", 6),
        min_samples_leaf=getattr(config, "RF_MIN_SAMPLES_LEAF", 20),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    return RandomForestArtifacts(
        scaler=scaler,
        model=model,
        feature_cols=feature_cols,
        target_col=target_col,
    )


def predict_returns(
    artifacts: RandomForestArtifacts,
    df_long: pd.DataFrame,
    pred_col: str = "pred_return",
) -> pd.DataFrame:
    X, _ = prepare_xy(df_long, artifacts.feature_cols, artifacts.target_col)
    X_scaled = artifacts.scaler.transform(X)
    preds = artifacts.model.predict(X_scaled)

    out = df_long.copy()
    out[pred_col] = preds
    return out