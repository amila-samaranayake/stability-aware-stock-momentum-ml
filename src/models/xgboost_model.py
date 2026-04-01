# src/models/xgboost_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from src import config


@dataclass
class XGBoostArtifacts:
    """
    Container for fitted XGBoost artifacts.
    """
    model: XGBRegressor
    feature_cols: list[str]
    target_col: str
    feature_importances_: pd.Series


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


def fit_xgboost(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> XGBoostArtifacts:
    """
    Fit an XGBoost regressor using the configured hyperparameters.
    """
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)

    model = XGBRegressor(
        n_estimators=getattr(config, "XGB_N_ESTIMATORS", 300),
        max_depth=getattr(config, "XGB_MAX_DEPTH", 4),
        learning_rate=getattr(config, "XGB_LEARNING_RATE", 0.05),
        subsample=getattr(config, "XGB_SUBSAMPLE", 0.8),
        colsample_bytree=getattr(config, "XGB_COLSAMPLE_BYTREE", 0.8),
        reg_alpha=getattr(config, "XGB_REG_ALPHA", 0.0),
        reg_lambda=getattr(config, "XGB_REG_LAMBDA", 1.0),
        min_child_weight=getattr(config, "XGB_MIN_CHILD_WEIGHT", 5),
        gamma=getattr(config, "XGB_GAMMA", 0.0),
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    feature_importances = pd.Series(
        model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    return XGBoostArtifacts(
        model=model,
        feature_cols=feature_cols,
        target_col=target_col,
        feature_importances_=feature_importances,
    )


def predict_returns(
    artifacts: XGBoostArtifacts,
    df_long: pd.DataFrame,
    pred_col: str = "pred_return",
) -> pd.DataFrame:
    """
    Predict returns for each row of a long ML dataframe.
    """
    X, _ = prepare_xy(df_long, artifacts.feature_cols, artifacts.target_col)
    preds = artifacts.model.predict(X)

    out = df_long.copy()
    out[pred_col] = preds
    return out