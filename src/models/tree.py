# src/models/tree.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src import config


@dataclass
class RandomForestArtifacts:
    """
    Container for fitted Random Forest artifacts.
    """
    model: RandomForestRegressor
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


def fit_random_forest(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> RandomForestArtifacts:
    """
    Fit a Random Forest regressor using the configured hyperparameters.
    """
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)

    model = RandomForestRegressor(
        n_estimators=getattr(config, "RF_N_ESTIMATORS", 300),
        max_depth=getattr(config, "RF_MAX_DEPTH", 6),
        min_samples_leaf=getattr(config, "RF_MIN_SAMPLES_LEAF", 20),
        min_samples_split=getattr(config, "RF_MIN_SAMPLES_SPLIT", 40),
        max_features=getattr(config, "RF_MAX_FEATURES", "sqrt"),
        bootstrap=getattr(config, "RF_BOOTSTRAP", True),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    feature_importances = pd.Series(
        model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    return RandomForestArtifacts(
        model=model,
        feature_cols=feature_cols,
        target_col=target_col,
        feature_importances_=feature_importances,
    )


def predict_returns(
    artifacts: RandomForestArtifacts,
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