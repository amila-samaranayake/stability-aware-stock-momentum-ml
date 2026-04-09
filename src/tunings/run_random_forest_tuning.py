from __future__ import annotations

import json
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src import config
from src.models.tree import fit_random_forest, predict_returns
from src.utils.paths import get_feature_dataset_paths


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute directional accuracy.
    """
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def ranking_metrics_by_month(
    df_pred: pd.DataFrame,
    target_col: str,
    pred_col: str = "pred_return",
    top_pct: float = 0.20,
) -> dict:
    """
    Compute average monthly ranking metrics.
    """
    if not isinstance(df_pred.index, pd.MultiIndex):
        raise ValueError("df_pred must be indexed by (date, ticker).")

    spearman_list = []
    hitrate_list = []

    for _, group in df_pred.groupby(level="date"):
        if group.shape[0] < 10:
            continue

        spearman = group[[pred_col, target_col]].corr(method="spearman").iloc[0, 1]
        if not np.isnan(spearman):
            spearman_list.append(float(spearman))

        k = max(1, int(np.ceil(group.shape[0] * top_pct)))
        pred_top = set(group.nlargest(k, pred_col).index.get_level_values("ticker"))
        true_top = set(group.nlargest(k, target_col).index.get_level_values("ticker"))
        hitrate_list.append(float(len(pred_top.intersection(true_top)) / k))

    return {
        "SpearmanRankCorr_mean": float(np.mean(spearman_list)) if spearman_list else float("nan"),
        "TopKHitRate_mean": float(np.mean(hitrate_list)) if hitrate_list else float("nan"),
        "Months_evaluated": int(len(hitrate_list)),
    }


def build_time_folds_from_dates(
    unique_dates: pd.Index,
    n_splits: int = 5,
) -> list[tuple[pd.Index, pd.Index]]:
    """
    Build expanding-window time-based folds.
    """
    unique_dates = pd.to_datetime(unique_dates).sort_values()
    n_dates = len(unique_dates)

    fold_size = n_dates // (n_splits + 1)
    if fold_size < 6:
        raise ValueError("Not enough dates to create stable time-based folds.")

    folds: list[tuple[pd.Index, pd.Index]] = []

    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        val_end = min((i + 1) * fold_size, n_dates)

        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]

        if len(val_dates) == 0:
            continue

        folds.append((train_dates, val_dates))

    return folds


def combined_score(
    rmse: float,
    directional_accuracy_value: float,
    spearman_value: float,
    topk_value: float,
) -> float:
    """
    Combined tuning score.

    Higher is better.
    RMSE enters with a negative sign because lower RMSE is better.
    """
    rmse_term = -rmse
    return (
        0.20 * rmse_term
        + 0.20 * directional_accuracy_value
        + 0.25 * spearman_value
        + 0.35 * topk_value
    )


def select_best_row(summary_df: pd.DataFrame, selection_metric: str) -> pd.Series:
    """
    Select best parameter row based on requested metric.
    """
    metric = selection_metric.lower()

    if metric == "rmse":
        return summary_df.sort_values("RMSE_mean", ascending=True).iloc[0]
    if metric == "directional_accuracy":
        return summary_df.sort_values("DirectionalAccuracy_mean", ascending=False).iloc[0]
    if metric == "spearman":
        return summary_df.sort_values("SpearmanRankCorr_mean", ascending=False).iloc[0]
    if metric == "topk_hit_rate":
        return summary_df.sort_values("TopKHitRate_mean", ascending=False).iloc[0]

    return summary_df.sort_values("CombinedScore_mean", ascending=False).iloc[0]


def main() -> None:
    """
    Tune Random Forest with time-aware expanding folds.
    """
    feature_paths = get_feature_dataset_paths(config.FEATURE_SOURCE)
    ml_train_path = feature_paths["train"]

    results_dir = os.path.join(
        "experiments",
        "results",
        f"exp04_random_forest_tuning_{config.FEATURE_SOURCE}",
    )
    os.makedirs(results_dir, exist_ok=True)

    fold_results_path = os.path.join(results_dir, "rf_tuning_fold_results.csv")
    summary_results_path = os.path.join(results_dir, "rf_tuning_summary.csv")
    best_params_path = os.path.join(results_dir, "best_rf_params.json")

    ml_train = pd.read_parquet(ml_train_path)

    target_cols = [c for c in ml_train.columns if c.startswith("y_next")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly 1 target column, found: {target_cols}")
    target_col = target_cols[0]

    feature_cols = [c for c in ml_train.columns if c != target_col]

    top_pct = getattr(config, "TOP_PERCENTAGE", 0.20)
    n_splits = getattr(config, "RF_TUNING_SPLITS", 5)
    selection_metric = getattr(config, "RF_SELECTION_METRIC", "combined_score")

    unique_dates = ml_train.index.get_level_values("date").unique().sort_values()
    folds = build_time_folds_from_dates(unique_dates, n_splits=n_splits)

    n_estimators_grid = getattr(config, "RF_N_ESTIMATORS_GRID", [300, 500])
    max_depth_grid = getattr(config, "RF_MAX_DEPTH_GRID", [4, 6, 8])
    min_samples_leaf_grid = getattr(config, "RF_MIN_SAMPLES_LEAF_GRID", [10, 20, 30])
    min_samples_split_grid = getattr(config, "RF_MIN_SAMPLES_SPLIT_GRID", [20, 40])
    max_features_grid = getattr(config, "RF_MAX_FEATURES_GRID", ["sqrt", 0.5])

    all_results = []

    param_grid = list(
        product(
            n_estimators_grid,
            max_depth_grid,
            min_samples_leaf_grid,
            min_samples_split_grid,
            max_features_grid,
        )
    )

    for (
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        max_features,
    ) in param_grid:
        for fold_id, (train_dates, val_dates) in enumerate(folds, start=1):
            train_mask = ml_train.index.get_level_values("date").isin(train_dates)
            val_mask = ml_train.index.get_level_values("date").isin(val_dates)

            fold_train = ml_train.loc[train_mask].copy()
            fold_val = ml_train.loc[val_mask].copy()

            old_n_estimators = getattr(config, "RF_N_ESTIMATORS", None)
            old_max_depth = getattr(config, "RF_MAX_DEPTH", None)
            old_min_samples_leaf = getattr(config, "RF_MIN_SAMPLES_LEAF", None)
            old_min_samples_split = getattr(config, "RF_MIN_SAMPLES_SPLIT", None)
            old_max_features = getattr(config, "RF_MAX_FEATURES", None)

            config.RF_N_ESTIMATORS = n_estimators
            config.RF_MAX_DEPTH = max_depth
            config.RF_MIN_SAMPLES_LEAF = min_samples_leaf
            config.RF_MIN_SAMPLES_SPLIT = min_samples_split
            config.RF_MAX_FEATURES = max_features

            artifacts = fit_random_forest(
                train_df=fold_train,
                feature_cols=feature_cols,
                target_col=target_col,
            )

            pred_val = predict_returns(artifacts, fold_val, pred_col="pred_return")

            y_true = pred_val[target_col].to_numpy(dtype=float)
            y_pred = pred_val["pred_return"].to_numpy(dtype=float)

            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            dir_acc = directional_accuracy(y_true, y_pred)

            rank_metrics = ranking_metrics_by_month(
                pred_val,
                target_col=target_col,
                pred_col="pred_return",
                top_pct=top_pct,
            )

            score = combined_score(
                rmse=rmse,
                directional_accuracy_value=dir_acc,
                spearman_value=rank_metrics["SpearmanRankCorr_mean"],
                topk_value=rank_metrics["TopKHitRate_mean"],
            )

            all_results.append(
                {
                    "n_estimators": int(n_estimators),
                    "max_depth": max_depth,
                    "min_samples_leaf": int(min_samples_leaf),
                    "min_samples_split": int(min_samples_split),
                    "max_features": str(max_features),
                    "fold": fold_id,
                    "train_start": str(train_dates.min().date()),
                    "train_end": str(train_dates.max().date()),
                    "val_start": str(val_dates.min().date()),
                    "val_end": str(val_dates.max().date()),
                    "RMSE": rmse,
                    "DirectionalAccuracy": dir_acc,
                    "SpearmanRankCorr_mean": rank_metrics["SpearmanRankCorr_mean"],
                    "TopKHitRate_mean": rank_metrics["TopKHitRate_mean"],
                    "Months_evaluated": rank_metrics["Months_evaluated"],
                    "CombinedScore": score,
                }
            )

            if old_n_estimators is not None:
                config.RF_N_ESTIMATORS = old_n_estimators
            if old_max_depth is not None:
                config.RF_MAX_DEPTH = old_max_depth
            if old_min_samples_leaf is not None:
                config.RF_MIN_SAMPLES_LEAF = old_min_samples_leaf
            if old_min_samples_split is not None:
                config.RF_MIN_SAMPLES_SPLIT = old_min_samples_split
            if old_max_features is not None:
                config.RF_MAX_FEATURES = old_max_features

    fold_df = pd.DataFrame(all_results)
    fold_df.to_csv(fold_results_path, index=False)

    summary_df = (
        fold_df.groupby(
            [
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "min_samples_split",
                "max_features",
            ],
            as_index=False,
        )
        .agg(
            {
                "RMSE": ["mean", "std"],
                "DirectionalAccuracy": ["mean", "std"],
                "SpearmanRankCorr_mean": ["mean", "std"],
                "TopKHitRate_mean": ["mean", "std"],
                "CombinedScore": ["mean", "std"],
                "Months_evaluated": "mean",
            }
        )
    )

    summary_df.columns = [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "RMSE_mean",
        "RMSE_std",
        "DirectionalAccuracy_mean",
        "DirectionalAccuracy_std",
        "SpearmanRankCorr_mean",
        "SpearmanRankCorr_std",
        "TopKHitRate_mean",
        "TopKHitRate_std",
        "CombinedScore_mean",
        "CombinedScore_std",
        "Months_evaluated_mean",
    ]

    summary_df = summary_df.sort_values("CombinedScore_mean", ascending=False).reset_index(drop=True)
    summary_df.to_csv(summary_results_path, index=False)

    best_row = select_best_row(summary_df, selection_metric=selection_metric)

    best_params = {
        "selection_metric": selection_metric,
        "feature_source": config.FEATURE_SOURCE,
        "n_splits": int(n_splits),
        "best_params": {
            "RF_N_ESTIMATORS": int(best_row["n_estimators"]),
            "RF_MAX_DEPTH": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
            "RF_MIN_SAMPLES_LEAF": int(best_row["min_samples_leaf"]),
            "RF_MIN_SAMPLES_SPLIT": int(best_row["min_samples_split"]),
            "RF_MAX_FEATURES": best_row["max_features"],
        },
        "best_user_attrs": {
            "RMSE_mean": float(best_row["RMSE_mean"]),
            "DirectionalAccuracy_mean": float(best_row["DirectionalAccuracy_mean"]),
            "SpearmanRankCorr_mean": float(best_row["SpearmanRankCorr_mean"]),
            "TopKHitRate_mean": float(best_row["TopKHitRate_mean"]),
            "CombinedScore_mean": float(best_row["CombinedScore_mean"]),
        },
    }

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print("=== RF tuning complete ===")
    print("Fold results saved to:", fold_results_path)
    print("Summary saved to:", summary_results_path)
    print("Best params saved to:", best_params_path)

    print("\n=== BEST PARAMS ===")
    print(json.dumps(best_params, indent=4))

    print("\n=== TOP 10 SETTINGS BY COMBINED SCORE ===")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()