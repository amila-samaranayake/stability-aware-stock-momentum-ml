# src/run_linear_tuning.py

import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from src import config
from src.models.linear import fit_ridge_with_scaler, predict_returns
from src.utils.paths import get_feature_dataset_paths, get_experiment_dir


FEATURE_DATASET_PATHS = get_feature_dataset_paths(config.FEATURE_SOURCE)

ML_TRAIN_PATH = FEATURE_DATASET_PATHS["train"]

RESULTS_DIR = get_experiment_dir("exp02_linear_ridge_tuning", config.FEATURE_SOURCE)
FOLD_RESULTS_PATH = os.path.join(RESULTS_DIR, "ridge_tuning_fold_results.csv")
SUMMARY_RESULTS_PATH = os.path.join(RESULTS_DIR, "ridge_tuning_summary.csv")
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, "best_ridge_params.json")


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute sign-based directional accuracy.
    """
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def ranking_metrics_by_month(
    df_pred: pd.DataFrame,
    target_col: str,
    pred_col: str = "pred_return",
    top_pct: float = 0.20,
) -> dict:
    """
    Compute ranking metrics month by month.
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
    Build expanding-window time-based folds from sorted unique dates.
    """
    unique_dates = pd.to_datetime(unique_dates).sort_values()
    n_dates = len(unique_dates)

    fold_size = n_dates // (n_splits + 1)
    if fold_size < 6:
        raise ValueError("Not enough dates to build stable time-based folds.")

    folds = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        val_end = min((i + 1) * fold_size, n_dates)

        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]

        if len(val_dates) == 0:
            continue

        folds.append((train_dates, val_dates))

    return folds


def select_best_alpha(summary_df: pd.DataFrame, selection_metric: str) -> float:
    """
    Select the best Ridge alpha according to the configured selection metric.
    """
    metric = selection_metric.lower()

    if metric == "rmse":
        row = summary_df.sort_values("RMSE_mean", ascending=True).iloc[0]
    elif metric == "directional_accuracy":
        row = summary_df.sort_values("DirectionalAccuracy_mean", ascending=False).iloc[0]
    elif metric == "spearman":
        row = summary_df.sort_values("SpearmanRankCorr_mean", ascending=False).iloc[0]
    else:
        row = summary_df.sort_values("TopKHitRate_mean", ascending=False).iloc[0]

    return float(row["alpha"])


def main() -> None:
    """
    Run time-aware Ridge hyperparameter tuning on the selected feature dataset.
    """
    print(f"Using feature source: {config.FEATURE_SOURCE}")
    print(f"ML train path: {ML_TRAIN_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ml_train = pd.read_parquet(ML_TRAIN_PATH)

    target_cols = [c for c in ml_train.columns if c.startswith("y_next")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly 1 target column, found: {target_cols}")
    target_col = target_cols[0]

    feature_cols = [c for c in ml_train.columns if c != target_col]

    alpha_grid = getattr(config, "RIDGE_ALPHA_GRID", [0.01, 0.1, 1.0, 10.0, 100.0])
    n_splits = getattr(config, "RIDGE_TUNING_SPLITS", 5)
    selection_metric = getattr(config, "RIDGE_SELECTION_METRIC", "topk_hit_rate")
    top_pct = getattr(config, "TOP_PERCENTAGE", 0.20)

    unique_dates = ml_train.index.get_level_values("date").unique().sort_values()
    folds = build_time_folds_from_dates(unique_dates, n_splits=n_splits)

    all_results = []

    for alpha in alpha_grid:
        for fold_id, (train_dates, val_dates) in enumerate(folds, start=1):
            train_mask = ml_train.index.get_level_values("date").isin(train_dates)
            val_mask = ml_train.index.get_level_values("date").isin(val_dates)

            fold_train = ml_train.loc[train_mask].copy()
            fold_val = ml_train.loc[val_mask].copy()

            artifacts = fit_ridge_with_scaler(
                train_df=fold_train,
                feature_cols=feature_cols,
                target_col=target_col,
                alpha=alpha,
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

            all_results.append({
                "alpha": float(alpha),
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
            })

    fold_df = pd.DataFrame(all_results)
    fold_df.to_csv(FOLD_RESULTS_PATH, index=False)

    summary_df = (
        fold_df
        .groupby("alpha", as_index=False)
        .agg({
            "RMSE": ["mean", "std"],
            "DirectionalAccuracy": ["mean", "std"],
            "SpearmanRankCorr_mean": ["mean", "std"],
            "TopKHitRate_mean": ["mean", "std"],
            "Months_evaluated": "mean",
        })
    )

    summary_df.columns = [
        "alpha",
        "RMSE_mean", "RMSE_std",
        "DirectionalAccuracy_mean", "DirectionalAccuracy_std",
        "SpearmanRankCorr_mean", "SpearmanRankCorr_std",
        "TopKHitRate_mean", "TopKHitRate_std",
        "Months_evaluated_mean",
    ]

    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

    best_alpha = select_best_alpha(summary_df, selection_metric=selection_metric)

    best_params = {
        "best_alpha": best_alpha,
        "selection_metric": selection_metric,
        "alpha_grid": list(alpha_grid),
        "n_splits": int(n_splits),
        "feature_source": config.FEATURE_SOURCE,
    }

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)

    print("== Ridge tuning complete ==")
    print("Fold results saved to:", FOLD_RESULTS_PATH)
    print("Summary saved to:", SUMMARY_RESULTS_PATH)
    print("Best params saved to:", BEST_PARAMS_PATH)

    print("\n=== TUNING SUMMARY ===")
    print(summary_df.to_string(index=False))

    print(f"\nBest alpha by '{selection_metric}': {best_alpha}")


if __name__ == "__main__":
    main()