# src/models/linear.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge


@dataclass
class LinearModelArtifacts:
    scaler: RobustScaler
    model: Ridge
    feature_cols: list[str]
    target_col: str


def prepare_xy(
    df_long: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert long ML dataset (indexed by date,ticker) to X,y arrays.
    """
    X = df_long[feature_cols].to_numpy(dtype=float)
    y = df_long[target_col].to_numpy(dtype=float)
    return X, y


def fit_ridge_with_robust_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float = 1.0,
) -> LinearModelArtifacts:
    """
    Fit RobustScaler on train features ONLY, then fit Ridge regression.
    """
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)

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
    Predict next-month returns for each (date,ticker) row.
    Returns a copy of df_long with an added prediction column.
    """
    X, _ = prepare_xy(df_long, artifacts.feature_cols, artifacts.target_col)
    X_scaled = artifacts.scaler.transform(X)
    preds = artifacts.model.predict(X_scaled)

    out = df_long.copy()
    out[pred_col] = preds
    return out