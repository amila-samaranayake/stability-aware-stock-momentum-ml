# src/tunings/run_xgboost_tuning.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

from src import config
from src.models.xgboost_model import fit_xgboost, predict_returns
from src.utils.paths import get_feature_dataset_paths, get_experiment_dir


FEATURE_DATASET_PATHS = get_feature_dataset_paths(config.FEATURE_SOURCE)

ML_TRAIN_PATH = FEATURE_DATASET_PATHS["train"]

RESULTS_DIR = Path(get_experiment_dir("exp03_xgboost_tuning", config.FEATURE_SOURCE))
FOLD_RESULTS_PATH = RESULTS_DIR / "xgboost_tuning_fold_results.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "xgboost_tuning_summary.csv"
BEST_PARAMS_PATH = RESULTS_DIR / "best_xgboost_params.json"
STUDY_CSV_PATH = RESULTS_DIR / "optuna_trials.csv"


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
) -> dict[str, float]:
    """
    Compute ranking metrics month by month.
    """
    if not isinstance(df_pred.index, pd.MultiIndex):
        raise ValueError("df_pred must be indexed by (date, ticker).")

    spearman_list: list[float] = []
    hitrate_list: list[float] = []

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
    Build expanding-window folds using ordered unique monthly dates.
    """
    unique_dates = pd.to_datetime(unique_dates).sort_values()
    n_dates = len(unique_dates)

    fold_size = n_dates // (n_splits + 1)
    if fold_size < 6:
        raise ValueError("Not enough dates to build stable time-based folds.")

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


def get_target_and_features(ml_train: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Resolve target and feature columns from training dataset.
    """
    target_cols = [c for c in ml_train.columns if c.startswith("y_next")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly 1 target column, found: {target_cols}")

    target_col = target_cols[0]
    feature_cols = [c for c in ml_train.columns if c != target_col]
    return target_col, feature_cols


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Suggest a narrower, more practical XGBoost search space.
    """
    return {
        "XGB_N_ESTIMATORS": trial.suggest_int("XGB_N_ESTIMATORS", 300, 900, step=100),
        "XGB_MAX_DEPTH": trial.suggest_int("XGB_MAX_DEPTH", 2, 5),
        "XGB_LEARNING_RATE": trial.suggest_float("XGB_LEARNING_RATE", 0.005, 0.03, log=True),
        "XGB_SUBSAMPLE": trial.suggest_float("XGB_SUBSAMPLE", 0.60, 0.90),
        "XGB_COLSAMPLE_BYTREE": trial.suggest_float("XGB_COLSAMPLE_BYTREE", 0.60, 1.00),
        "XGB_REG_ALPHA": trial.suggest_float("XGB_REG_ALPHA", 0.0, 2.0),
        "XGB_REG_LAMBDA": trial.suggest_float("XGB_REG_LAMBDA", 1.0, 5.0),
        "XGB_MIN_CHILD_WEIGHT": trial.suggest_int("XGB_MIN_CHILD_WEIGHT", 5, 20),
        "XGB_GAMMA": trial.suggest_float("XGB_GAMMA", 0.0, 1.0),
    }


def apply_temp_xgb_config(params: dict[str, Any]) -> dict[str, Any]:
    """
    Temporarily override config values for one trial.
    """
    old_values: dict[str, Any] = {}
    for key, value in params.items():
        old_values[key] = getattr(config, key)
        setattr(config, key, value)
    return old_values


def restore_temp_xgb_config(old_values: dict[str, Any]) -> None:
    """
    Restore original config values after a trial.
    """
    for key, value in old_values.items():
        setattr(config, key, value)


def make_combined_score(
    rmse_mean: float,
    directional_accuracy_mean: float,
    spearman_mean: float,
    topk_mean: float,
) -> float:
    """
    Combined tuning score.

    Higher is better.

    We reward:
    - Top-k hit rate most
    - directional accuracy second
    - Spearman third

    We lightly penalize RMSE by converting it into a bounded inverse form.
    """
    rmse_component = 1.0 / (1.0 + rmse_mean)

    score = (
        0.40 * topk_mean
        + 0.30 * directional_accuracy_mean
        + 0.20 * spearman_mean
        + 0.10 * rmse_component
    )
    return float(score)


def make_objective(
    ml_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    folds: list[tuple[pd.Index, pd.Index]],
    top_pct: float,
    fold_records: list[dict[str, Any]],
):
    """
    Create Optuna objective function using a combined validation score.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        old_values = apply_temp_xgb_config(params)

        rmse_values: list[float] = []
        dir_values: list[float] = []
        spearman_values: list[float] = []
        topk_values: list[float] = []

        try:
            for fold_id, (train_dates, val_dates) in enumerate(folds, start=1):
                train_mask = ml_train.index.get_level_values("date").isin(train_dates)
                val_mask = ml_train.index.get_level_values("date").isin(val_dates)

                fold_train = ml_train.loc[train_mask].copy()
                fold_val = ml_train.loc[val_mask].copy()

                artifacts = fit_xgboost(
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

                rmse_values.append(rmse)
                dir_values.append(dir_acc)
                spearman_values.append(rank_metrics["SpearmanRankCorr_mean"])
                topk_values.append(rank_metrics["TopKHitRate_mean"])

                fold_records.append({
                    "trial_number": trial.number,
                    **params,
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

            rmse_mean = float(np.mean(rmse_values))
            dir_mean = float(np.mean(dir_values))
            spearman_mean = float(np.mean(spearman_values))
            topk_mean = float(np.mean(topk_values))

            combined_score = make_combined_score(
                rmse_mean=rmse_mean,
                directional_accuracy_mean=dir_mean,
                spearman_mean=spearman_mean,
                topk_mean=topk_mean,
            )

            trial.set_user_attr("RMSE_mean", rmse_mean)
            trial.set_user_attr("DirectionalAccuracy_mean", dir_mean)
            trial.set_user_attr("SpearmanRankCorr_mean", spearman_mean)
            trial.set_user_attr("TopKHitRate_mean", topk_mean)
            trial.set_user_attr("CombinedScore_mean", combined_score)

            return combined_score

        finally:
            restore_temp_xgb_config(old_values)

    return objective


def main() -> None:
    """
    Tune XGBoost hyperparameters with Optuna using time-aware folds
    and a combined validation objective.
    """
    print(f"Using feature source: {config.FEATURE_SOURCE}")
    print(f"ML train path: {ML_TRAIN_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ml_train = pd.read_parquet(ML_TRAIN_PATH)
    target_col, feature_cols = get_target_and_features(ml_train)

    n_trials = getattr(config, "XGB_TUNING_TRIALS", 40)
    n_splits = getattr(config, "XGB_TUNING_SPLITS", 5)
    top_pct = getattr(config, "TOP_PERCENTAGE", 0.20)

    unique_dates = ml_train.index.get_level_values("date").unique().sort_values()
    folds = build_time_folds_from_dates(unique_dates, n_splits=n_splits)

    fold_records: list[dict[str, Any]] = []

    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgboost_tuning_{config.FEATURE_SOURCE}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = make_objective(
        ml_train=ml_train,
        feature_cols=feature_cols,
        target_col=target_col,
        folds=folds,
        top_pct=top_pct,
        fold_records=fold_records,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(FOLD_RESULTS_PATH, index=False)

    summary_df = (
        fold_df
        .groupby(
            [
                "trial_number",
                "XGB_N_ESTIMATORS",
                "XGB_MAX_DEPTH",
                "XGB_LEARNING_RATE",
                "XGB_SUBSAMPLE",
                "XGB_COLSAMPLE_BYTREE",
                "XGB_REG_ALPHA",
                "XGB_REG_LAMBDA",
                "XGB_MIN_CHILD_WEIGHT",
                "XGB_GAMMA",
            ],
            as_index=False,
        )
        .agg({
            "RMSE": ["mean", "std"],
            "DirectionalAccuracy": ["mean", "std"],
            "SpearmanRankCorr_mean": ["mean", "std"],
            "TopKHitRate_mean": ["mean", "std"],
            "Months_evaluated": "mean",
        })
    )

    summary_df.columns = [
        "trial_number",
        "XGB_N_ESTIMATORS",
        "XGB_MAX_DEPTH",
        "XGB_LEARNING_RATE",
        "XGB_SUBSAMPLE",
        "XGB_COLSAMPLE_BYTREE",
        "XGB_REG_ALPHA",
        "XGB_REG_LAMBDA",
        "XGB_MIN_CHILD_WEIGHT",
        "XGB_GAMMA",
        "RMSE_mean", "RMSE_std",
        "DirectionalAccuracy_mean", "DirectionalAccuracy_std",
        "SpearmanRankCorr_mean", "SpearmanRankCorr_std",
        "TopKHitRate_mean", "TopKHitRate_std",
        "Months_evaluated_mean",
    ]

    summary_df["CombinedScore_mean"] = summary_df.apply(
        lambda row: make_combined_score(
            rmse_mean=float(row["RMSE_mean"]),
            directional_accuracy_mean=float(row["DirectionalAccuracy_mean"]),
            spearman_mean=float(row["SpearmanRankCorr_mean"]),
            topk_mean=float(row["TopKHitRate_mean"]),
        ),
        axis=1,
    )

    summary_df = summary_df.sort_values(
        ["CombinedScore_mean", "TopKHitRate_mean", "DirectionalAccuracy_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(STUDY_CSV_PATH, index=False)

    best_params = study.best_params
    best_summary = {
        "best_trial_number": study.best_trial.number,
        "best_objective_value": float(study.best_value),
        "selection_metric": "combined_score",
        "feature_source": config.FEATURE_SOURCE,
        "n_trials": int(n_trials),
        "n_splits": int(n_splits),
        "best_params": best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_summary, f, indent=4)

    print("\n== XGBoost tuning complete ==")
    print("Fold results saved to:", FOLD_RESULTS_PATH)
    print("Summary saved to:", SUMMARY_RESULTS_PATH)
    print("Best params saved to:", BEST_PARAMS_PATH)
    print("Optuna trials saved to:", STUDY_CSV_PATH)

    print("\n=== TOP 5 TRIALS BY COMBINED SCORE ===")
    print(
        summary_df[
            [
                "trial_number",
                "CombinedScore_mean",
                "TopKHitRate_mean",
                "DirectionalAccuracy_mean",
                "SpearmanRankCorr_mean",
                "RMSE_mean",
                "XGB_N_ESTIMATORS",
                "XGB_MAX_DEPTH",
                "XGB_LEARNING_RATE",
                "XGB_SUBSAMPLE",
                "XGB_COLSAMPLE_BYTREE",
                "XGB_REG_ALPHA",
                "XGB_REG_LAMBDA",
                "XGB_MIN_CHILD_WEIGHT",
                "XGB_GAMMA",
            ]
        ].head(5).to_string(index=False)
    )

    print("\n=== BEST TRIAL ===")
    print(json.dumps(best_summary, indent=4))


if __name__ == "__main__":
    main()