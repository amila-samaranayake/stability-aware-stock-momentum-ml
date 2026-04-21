# src/run_lstm.py

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.evaluation.backtest import (
    apply_transaction_costs,
    compute_equity_curve,
    compute_portfolio_returns,
)
from src.evaluation.metrics import summarize_metrics, turnover
from src.features_lstm import load_lstm_sample_set, lstm_sample_set_to_long_dataframe
from src.models.lstm_model import fit_lstm, predict_lstm
from src.strategies.momentum import build_equal_weight_weights, select_top_assets
from src.utils.paths import get_experiment_dir, get_processed_returns_paths


LSTM_FEATURE_DIR = Path("data/processed/features_lstm")

TRAIN_NPZ_PATH = LSTM_FEATURE_DIR / "lstm_train_daily_2015_2024.npz"
TEST_NPZ_PATH = LSTM_FEATURE_DIR / "lstm_test_daily_2025.npz"

RETURNS_PATHS = get_processed_returns_paths()
RET_TRAIN_PATH = RETURNS_PATHS["train_monthly"]
RET_TEST_PATH = RETURNS_PATHS["test_monthly"]

RESULTS_DIR = Path(get_experiment_dir("exp06_lstm", "daily"))

METRICS_TRAIN_PATH = RESULTS_DIR / "metrics_train.json"
METRICS_TEST_PATH = RESULTS_DIR / "metrics_test_2025.json"
METRICS_TRAIN_COSTS_PATH = RESULTS_DIR / "metrics_train_with_costs.json"
METRICS_TEST_COSTS_PATH = RESULTS_DIR / "metrics_test_2025_with_costs.json"

EQUITY_TRAIN_PATH = RESULTS_DIR / "equity_train.csv"
EQUITY_TEST_PATH = RESULTS_DIR / "equity_test_2025.csv"
PRED_METRICS_PATH = RESULTS_DIR / "prediction_metrics.json"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"

TRAIN_PREDICTIONS_PATH = RESULTS_DIR / "train_predictions.csv"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"

LOSS_CURVE_PATH = RESULTS_DIR / "loss_curve.png"
PRED_SCATTER_TRAIN_PATH = RESULTS_DIR / "pred_vs_actual_train.png"
PRED_SCATTER_TEST_PATH = RESULTS_DIR / "pred_vs_actual_test_2025.png"


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def build_prediction_dataframe(
    meta_df: pd.DataFrame,
    preds: np.ndarray,
    pred_col: str = "pred_return",
) -> pd.DataFrame:
    """
    Attach predictions to LSTM metadata dataframe.
    """
    out = meta_df.copy()
    out[pred_col] = preds.astype(float)
    return out.sort_index()


def save_prediction_table(
    df_pred: pd.DataFrame,
    target_col: str,
    out_path: Path,
    pred_col: str = "pred_return",
) -> None:
    """
    Save standardized prediction table for downstream diagnostics.

    Output columns:
    - date
    - ticker
    - pred_return
    - target_col
    """
    if not isinstance(df_pred.index, pd.MultiIndex):
        raise ValueError("df_pred must be indexed by (date, ticker).")

    out = df_pred[[target_col, pred_col]].copy().reset_index()

    expected_cols = {"date", "ticker", target_col, pred_col}
    missing = expected_cols - set(out.columns)
    if missing:
        raise ValueError(
            f"Prediction table is missing required columns after reset_index: {sorted(missing)}"
        )

    out.to_csv(out_path, index=False)


def plot_loss_curve(history: dict, save_path: Path) -> None:
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_prediction_scatter(
    df_pred: pd.DataFrame,
    target_col: str,
    pred_col: str,
    save_path: Path,
    title: str,
) -> None:
    """
    Plot actual vs predicted returns.
    """
    y_true = df_pred[target_col].to_numpy(dtype=float)
    y_pred = df_pred[pred_col].to_numpy(dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, s=12)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    """
    Train and evaluate the LSTM experiment using the separate sequence dataset.
    """
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading LSTM datasets...")
    train_set = load_lstm_sample_set(str(TRAIN_NPZ_PATH))
    test_set = load_lstm_sample_set(str(TEST_NPZ_PATH))

    train_meta = lstm_sample_set_to_long_dataframe(train_set)
    test_meta = lstm_sample_set_to_long_dataframe(test_set)

    ret_train = pd.read_parquet(RET_TRAIN_PATH)
    ret_test = pd.read_parquet(RET_TEST_PATH)

    target_col = train_set.target_name
    top_pct = getattr(config, "TOP_PERCENTAGE", 0.20)

    print("Train X shape:", train_set.X.shape)
    print("Test X shape:", test_set.X.shape)
    print("Target column:", target_col)

    artifacts = fit_lstm(
        X_train_full=train_set.X,
        y_train_full=train_set.y,
        target_name=target_col,
    )

    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(artifacts.history, f, indent=4)

    plot_loss_curve(artifacts.history, LOSS_CURVE_PATH)

    pred_train_values = predict_lstm(artifacts, train_set.X)
    pred_test_values = predict_lstm(artifacts, test_set.X)

    pred_train = build_prediction_dataframe(train_meta, pred_train_values, pred_col="pred_return")
    pred_test = build_prediction_dataframe(test_meta, pred_test_values, pred_col="pred_return")

    save_prediction_table(
        df_pred=pred_train,
        target_col=target_col,
        out_path=TRAIN_PREDICTIONS_PATH,
        pred_col="pred_return",
    )
    save_prediction_table(
        df_pred=pred_test,
        target_col=target_col,
        out_path=TEST_PREDICTIONS_PATH,
        pred_col="pred_return",
    )

    plot_prediction_scatter(
        pred_train,
        target_col=target_col,
        pred_col="pred_return",
        save_path=PRED_SCATTER_TRAIN_PATH,
        title="LSTM Predicted vs Actual (Train)",
    )
    plot_prediction_scatter(
        pred_test,
        target_col=target_col,
        pred_col="pred_return",
        save_path=PRED_SCATTER_TEST_PATH,
        title="LSTM Predicted vs Actual (Test 2025)",
    )

    acc_train = regression_prediction_metrics(pred_train, target_col=target_col, pred_col="pred_return")
    acc_test = regression_prediction_metrics(pred_test, target_col=target_col, pred_col="pred_return")

    rank_train = ranking_metrics_by_month(pred_train, target_col=target_col, pred_col="pred_return", top_pct=top_pct)
    rank_test = ranking_metrics_by_month(pred_test, target_col=target_col, pred_col="pred_return", top_pct=top_pct)

    pred_metrics = {
        "train": {"regression": acc_train, "ranking": rank_train},
        "test_2025": {"regression": acc_test, "ranking": rank_test},
    }
    with open(PRED_METRICS_PATH, "w") as f:
        json.dump(pred_metrics, f, indent=4)

    print("\n=== PREDICTION METRICS (TRAIN 2015–2024) ===")
    for key, value in acc_train.items():
        print(f"{key}: {value:.6f}")
    for key, value in rank_train.items():
        print(f"{key}: {value}")

    print("\n=== PREDICTION METRICS (TEST 2025) ===")
    for key, value in acc_test.items():
        print(f"{key}: {value:.6f}")
    for key, value in rank_test.items():
        print(f"{key}: {value}")

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

    print("\n=== LSTM experiment saved to:", RESULTS_DIR)
    print(f"Saved train predictions -> {TRAIN_PREDICTIONS_PATH}")
    print(f"Saved test predictions  -> {TEST_PREDICTIONS_PATH}")

    print("\nSaved plots:")
    print("Loss curve ->", LOSS_CURVE_PATH)
    print("Train scatter ->", PRED_SCATTER_TRAIN_PATH)
    print("Test scatter ->", PRED_SCATTER_TEST_PATH)

    print("\n=== TRAIN STRATEGY METRICS (2015–2024) ===")
    for key, value in metrics_train.items():
        print(f"{key}: {value:.4f}")

    print("\n=== TRAIN WITH COSTS ===")
    for key, value in cost_results_train.items():
        print(
            key,
            "-> cumulative_return:", round(value["cumulative_return"], 4),
            "sharpe:", round(value["sharpe_ratio"], 4),
        )

    print("\n=== TEST STRATEGY METRICS (2025) ===")
    for key, value in metrics_test.items():
        print(f"{key}: {value:.4f}")

    print("\n=== TEST WITH COSTS ===")
    for key, value in cost_results_test.items():
        print(
            key,
            "-> cumulative_return:", round(value["cumulative_return"], 4),
            "sharpe:", round(value["sharpe_ratio"], 4),
        )


if __name__ == "__main__":
    main()