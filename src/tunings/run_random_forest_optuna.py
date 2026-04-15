# src/tunings/run_random_forest_optuna.py

from __future__ import annotations

import json
import os

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

from src import config
from src.models.tree import fit_random_forest, predict_returns
from src.utils.paths import get_feature_dataset_paths


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def ranking_metrics_by_month(
    df_pred: pd.DataFrame,
    target_col: str,
    pred_col: str = "pred_return",
    top_pct: float = 0.20,
) -> dict:
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
    unique_dates = pd.to_datetime(unique_dates).sort_values()
    n_dates = len(unique_dates)

    fold_size = n_dates // (n_splits + 1)
    if fold_size < 6:
        raise ValueError("Not enough dates to create stable time-based folds.")

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


def combined_score(
    rmse: float,
    directional_accuracy_value: float,
    spearman_value: float,
    topk_value: float,
) -> float:
    rmse_term = -rmse
    return (
        0.20 * rmse_term
        + 0.20 * directional_accuracy_value
        + 0.25 * spearman_value
        + 0.35 * topk_value
    )


def main() -> None:
    feature_paths = get_feature_dataset_paths(config.FEATURE_SOURCE)
    ml_train_path = feature_paths["train"]

    results_dir = os.path.join(
        "experiments",
        "results",
        f"exp04_random_forest_optuna_{config.FEATURE_SOURCE}",
    )
    os.makedirs(results_dir, exist_ok=True)

    trials_path = os.path.join(results_dir, "rf_optuna_trials.csv")
    best_params_path = os.path.join(results_dir, "best_rf_optuna_params.json")

    ml_train = pd.read_parquet(ml_train_path)

    target_cols = [c for c in ml_train.columns if c.startswith("y_next")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly 1 target column, found: {target_cols}")
    target_col = target_cols[0]

    feature_cols = [c for c in ml_train.columns if c != target_col]

    top_pct = getattr(config, "TOP_PERCENTAGE", 0.20)
    n_splits = getattr(config, "RF_TUNING_SPLITS", 5)
    n_trials = getattr(config, "RF_OPTUNA_N_TRIALS", 40)

    unique_dates = ml_train.index.get_level_values("date").unique().sort_values()
    folds = build_time_folds_from_dates(unique_dates, n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("RF_N_ESTIMATORS", 200, 700, step=100)
        max_depth = trial.suggest_int("RF_MAX_DEPTH", 3, 8)
        min_samples_leaf = trial.suggest_int("RF_MIN_SAMPLES_LEAF", 5, 40, step=5)
        min_samples_split = trial.suggest_int("RF_MIN_SAMPLES_SPLIT", 10, 80, step=10)
        max_features = trial.suggest_categorical("RF_MAX_FEATURES", ["sqrt", "log2", 0.4, 0.5, 0.7])

        rmse_values = []
        diracc_values = []
        spearman_values = []
        topk_values = []

        old_n_estimators = getattr(config, "RF_N_ESTIMATORS", None)
        old_max_depth = getattr(config, "RF_MAX_DEPTH", None)
        old_min_samples_leaf = getattr(config, "RF_MIN_SAMPLES_LEAF", None)
        old_min_samples_split = getattr(config, "RF_MIN_SAMPLES_SPLIT", None)
        old_max_features = getattr(config, "RF_MAX_FEATURES", None)

        try:
            for train_dates, val_dates in folds:
                train_mask = ml_train.index.get_level_values("date").isin(train_dates)
                val_mask = ml_train.index.get_level_values("date").isin(val_dates)

                fold_train = ml_train.loc[train_mask].copy()
                fold_val = ml_train.loc[val_mask].copy()

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

                rmse_values.append(rmse)
                diracc_values.append(dir_acc)
                spearman_values.append(rank_metrics["SpearmanRankCorr_mean"])
                topk_values.append(rank_metrics["TopKHitRate_mean"])

            rmse_mean = float(np.mean(rmse_values))
            diracc_mean = float(np.mean(diracc_values))
            spearman_mean = float(np.mean(spearman_values))
            topk_mean = float(np.mean(topk_values))

            score = combined_score(
                rmse=rmse_mean,
                directional_accuracy_value=diracc_mean,
                spearman_value=spearman_mean,
                topk_value=topk_mean,
            )

            trial.set_user_attr("RMSE_mean", rmse_mean)
            trial.set_user_attr("DirectionalAccuracy_mean", diracc_mean)
            trial.set_user_attr("SpearmanRankCorr_mean", spearman_mean)
            trial.set_user_attr("TopKHitRate_mean", topk_mean)
            trial.set_user_attr("CombinedScore_mean", score)

            return score

        finally:
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

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    trials_rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "objective_value": trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        trials_rows.append(row)

    trials_df = pd.DataFrame(trials_rows).sort_values("objective_value", ascending=False)
    trials_df.to_csv(trials_path, index=False)

    best_trial = study.best_trial
    best_params = {
        "best_trial_number": best_trial.number,
        "best_objective_value": best_trial.value,
        "selection_metric": "combined_score",
        "feature_source": config.FEATURE_SOURCE,
        "n_trials": int(n_trials),
        "n_splits": int(n_splits),
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
    }

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print("=== RF Optuna tuning complete ===")
    print("Trials saved to:", trials_path)
    print("Best params saved to:", best_params_path)

    print("\n=== BEST TRIAL ===")
    print(json.dumps(best_params, indent=4))

    print("\n=== TOP 10 TRIALS BY COMBINED SCORE ===")
    print(trials_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()