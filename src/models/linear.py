# src/models/linear.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler

from src import config


@dataclass
class LinearModelArtifacts:
    """
    Container for fitted Ridge model artifacts.
    """
    scaler: object
    model: Ridge
    feature_cols: list[str]
    target_col: str


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


def fit_ridge_with_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float = 1.0,
) -> LinearModelArtifacts:
    """
    Fit a scaler on training features only, then fit Ridge regression.
    """
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)

    scaler_type = getattr(config, "SCALER_TYPE", "robust").lower()
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)

    return LinearModelArtifacts(
        scaler=scaler,
        model=model,
        feature_cols=feature_cols,
        target_col=target_col,
    )


def predict_returns(
    artifacts: LinearModelArtifacts,
    df_long: pd.DataFrame,
    pred_col: str = "pred_return",
) -> pd.DataFrame:
    """
    Predict returns for each row of a long ML dataframe.
    """
    X, _ = prepare_xy(df_long, artifacts.feature_cols, artifacts.target_col)
    X_scaled = artifacts.scaler.transform(X)
    preds = artifacts.model.predict(X_scaled)

    out = df_long.copy()
    out[pred_col] = preds
    return out