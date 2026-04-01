# src/run_xgboost.py

import os
import json
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.models.xgboost_model import fit_xgboost, predict_returns
from src.strategies.momentum import select_top_assets, build_equal_weight_weights
from src.evaluation.backtest import (
    compute_portfolio_returns,
    compute_equity_curve,
    apply_transaction_costs,
)
from src.evaluation.metrics import summarize_metrics, turnover
from src.utils.paths import (
    get_feature_dataset_paths,
    get_processed_returns_paths,
    get_experiment_dir,
)


FEATURE_DATASET_PATHS = get_feature_dataset_paths(config.FEATURE_SOURCE)
RETURNS_PATHS = get_processed_returns_paths()

ML_TRAIN_PATH = FEATURE_DATASET_PATHS["train"]
ML_TEST_PATH = FEATURE_DATASET_PATHS["test"]

RET_TRAIN_PATH = RETURNS_PATHS["train_monthly"]
RET_TEST_PATH = RETURNS_PATHS["test_monthly"]

RESULTS_DIR = get_experiment_dir("exp03_xgboost", config.FEATURE_SOURCE)

METRICS_TRAIN_PATH = os.path.join(RESULTS_DIR, "metrics_train.json")
METRICS_TEST_PATH = os.path.join(RESULTS_DIR, "metrics_test_2025.json")
METRICS_TRAIN_COSTS_PATH = os.path.join(RESULTS_DIR, "metrics_train_with_costs.json")
METRICS_TEST_COSTS_PATH = os.path.join(RESULTS_DIR, "metrics_test_2025_with_costs.json")

EQUITY_TRAIN_PATH = os.path.join(RESULTS_DIR, "equity_train.csv")
EQUITY_TEST_PATH = os.path.join(RESULTS_DIR, "equity_test_2025.csv")
PRED_METRICS_PATH = os.path.join(RESULTS_DIR, "prediction_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")


def predictions_to_weights(pred_long: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """
    Convert long predictions into equal-weight portfolio weights.
    """
    pred_wide = pred_long["pred_return"].unstack("ticker").sort_index()
    selected = select_top_assets(signal=pred_wide, top_pct=top_pct)
    return build_equal_weight_weights(selected)


def regression_prediction_metrics(
    df_pred: pd.DataFrame,
    target_col: str,
    pred_col: str = "pred_return",
) -> dict:
    """
    Compute regression-style prediction metrics.
    """
    y_true = df_pred[target_col].to_numpy(dtype=float)
    y_pred = df_pred[pred_col].to_numpy(dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Directional_Accuracy": dir_acc,
    }


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


def compute_cost_adjusted_results(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
):
    """
    Compute net metrics for each configured transaction cost rate.
    """
    turnover_series = turnover(weights)
    cost_results = {}

    for cost_rate in config.TRANSACTION_COST_RATES:
        net_returns = apply_transaction_costs(
            portfolio_simple_returns=gross_returns,
            turnover_series=turnover_series,
            cost_rate=cost_rate,
        )
        net_equity = compute_equity_curve(net_returns)
        net_metrics = summarize_metrics(net_returns, net_equity, weights)

        key = f"cost_{int(cost_rate * 10000)}bps"
        cost_results[key] = net_metrics

    return turnover_series, cost_results


def main() -> None:
    """
    Run XGBoost on the selected feature dataset, evaluate predictions,
    backtest the resulting monthly portfolio, and save gross and net outputs.
    """
    print(f"Using feature source: {config.FEATURE_SOURCE}")
    print(f"ML train path: {ML_TRAIN_PATH}")
    print(f"ML test path: {ML_TEST_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ml_train = pd.read_parquet(ML_TRAIN_PATH)
    ml_test = pd.read_parquet(ML_TEST_PATH)

    ret_train = pd.read_parquet(RET_TRAIN_PATH)
    ret_test = pd.read_parquet(RET_TEST_PATH)

    target_cols = [c for c in ml_train.columns if c.startswith("y_next")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly 1 target column, found: {target_cols}")
    target_col = target_cols[0]

    feature_cols = [c for c in ml_train.columns if c != target_col]
    top_pct = getattr(config, "TOP_PERCENTAGE", 0.20)

    artifacts = fit_xgboost(
        train_df=ml_train,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    artifacts.feature_importances_.to_csv(FEATURE_IMPORTANCE_PATH, header=True)

    pred_train = predict_returns(artifacts, ml_train, pred_col="pred_return")
    pred_test = predict_returns(artifacts, ml_test, pred_col="pred_return")

    acc_train = regression_prediction_metrics(
        pred_train,
        target_col=target_col,
        pred_col="pred_return",
    )
    acc_test = regression_prediction_metrics(
        pred_test,
        target_col=target_col,
        pred_col="pred_return",
    )

    rank_train = ranking_metrics_by_month(
        pred_train,
        target_col=target_col,
        pred_col="pred_return",
        top_pct=top_pct,
    )
    rank_test = ranking_metrics_by_month(
        pred_test,
        target_col=target_col,
        pred_col="pred_return",
        top_pct=top_pct,
    )

    pred_metrics = {
        "train": {"regression": acc_train, "ranking": rank_train},
        "test_2025": {"regression": acc_test, "ranking": rank_test},
    }
    with open(PRED_METRICS_PATH, "w") as f:
        json.dump(pred_metrics, f, indent=4)

    print("\n=== PREDICTION METRICS (TRAIN 2015–2024) ===")
    for k, v in acc_train.items():
        print(f"{k}: {v:.6f}")
    for k, v in rank_train.items():
        print(f"{k}: {v}")

    print("\n=== PREDICTION METRICS (TEST 2025) ===")
    for k, v in acc_test.items():
        print(f"{k}: {v:.6f}")
    for k, v in rank_test.items():
        print(f"{k}: {v}")

    w_train = predictions_to_weights(pred_train, top_pct=top_pct)
    w_test = predictions_to_weights(pred_test, top_pct=top_pct)

    w_train = w_train[ret_train.columns.intersection(w_train.columns)]
    w_test = w_test[ret_test.columns.intersection(w_test.columns)]

    port_ret_train = compute_portfolio_returns(
        w_train,
        ret_train,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    equity_train = compute_equity_curve(port_ret_train)

    port_ret_test = compute_portfolio_returns(
        w_test,
        ret_test,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    equity_test = compute_equity_curve(port_ret_test)

    metrics_train = summarize_metrics(port_ret_train, equity_train, w_train)
    metrics_test = summarize_metrics(port_ret_test, equity_test, w_test)

    turnover_train, cost_results_train = compute_cost_adjusted_results(
        gross_returns=port_ret_train,
        weights=w_train,
    )
    turnover_test, cost_results_test = compute_cost_adjusted_results(
        gross_returns=port_ret_test,
        weights=w_test,
    )

    equity_train.to_csv(EQUITY_TRAIN_PATH)
    equity_test.to_csv(EQUITY_TEST_PATH)

    with open(METRICS_TRAIN_PATH, "w") as f:
        json.dump(metrics_train, f, indent=4)
    with open(METRICS_TEST_PATH, "w") as f:
        json.dump(metrics_test, f, indent=4)

    with open(METRICS_TRAIN_COSTS_PATH, "w") as f:
        json.dump(cost_results_train, f, indent=4)
    with open(METRICS_TEST_COSTS_PATH, "w") as f:
        json.dump(cost_results_test, f, indent=4)

    print("\n=== XGBoost experiment saved to:", RESULTS_DIR)

    print("\n=== TRAIN STRATEGY METRICS (2015–2024) ===")
    for k, v in metrics_train.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TRAIN WITH COSTS ===")
    for k, v in cost_results_train.items():
        print(
            k,
            "-> cumulative_return:", round(v["cumulative_return"], 4),
            "sharpe:", round(v["sharpe_ratio"], 4),
        )

    print("\n=== TEST STRATEGY METRICS (2025) ===")
    for k, v in metrics_test.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TEST WITH COSTS ===")
    for k, v in cost_results_test.items():
        print(
            k,
            "-> cumulative_return:", round(v["cumulative_return"], 4),
            "sharpe:", round(v["sharpe_ratio"], 4),
        )


if __name__ == "__main__":
    main()