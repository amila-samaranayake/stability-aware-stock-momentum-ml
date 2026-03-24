# src/run_xgboost.py

import os
import json
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.models.xgboost_model import fit_xgboost_with_scaler, predict_returns
from src.strategies.momentum import select_top_assets, build_equal_weight_weights
from src.evaluation.backtest import compute_portfolio_returns, compute_equity_curve
from src.evaluation.metrics import summarize_metrics

ML_TRAIN_PATH = "data/processed/ml_train_2015_2024.parquet"
ML_TEST_PATH = "data/processed/ml_test_2025.parquet"

RET_TRAIN_PATH = "data/processed/train_monthly_2015_2024.parquet"
RET_TEST_PATH = "data/processed/test_monthly_2025.parquet"

RESULTS_DIR = "experiments/results/exp03_xgboost"
METRICS_TRAIN_PATH = os.path.join(RESULTS_DIR, "metrics_train.json")
METRICS_TEST_PATH = os.path.join(RESULTS_DIR, "metrics_test_2025.json")
EQUITY_TRAIN_PATH = os.path.join(RESULTS_DIR, "equity_train.csv")
EQUITY_TEST_PATH = os.path.join(RESULTS_DIR, "equity_test_2025.csv")
PRED_METRICS_PATH = os.path.join(RESULTS_DIR, "prediction_metrics.json")


def predictions_to_weights(pred_long: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    pred_wide = pred_long["pred_return"].unstack("ticker").sort_index()
    selected = select_top_assets(signal=pred_wide, top_pct=top_pct)
    weights = build_equal_weight_weights(selected)
    return weights


def regression_prediction_metrics(
    df_pred: pd.DataFrame,
    target_col: str,
    pred_col: str = "pred_return",
) -> dict:
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
    if not isinstance(df_pred.index, pd.MultiIndex):
        raise ValueError("df_pred must be indexed by (date, ticker).")

    spearman_list = []
    hitrate_list = []

    for _, g in df_pred.groupby(level="date"):
        if g.shape[0] < 10:
            continue

        s = g[[pred_col, target_col]].corr(method="spearman").iloc[0, 1]
        if not np.isnan(s):
            spearman_list.append(float(s))

        k = max(1, int(np.ceil(g.shape[0] * top_pct)))
        pred_top = set(g.nlargest(k, pred_col).index.get_level_values("ticker"))
        true_top = set(g.nlargest(k, target_col).index.get_level_values("ticker"))

        hitrate_list.append(float(len(pred_top.intersection(true_top)) / k))

    return {
        "SpearmanRankCorr_mean": float(np.mean(spearman_list)) if spearman_list else float("nan"),
        "TopKHitRate_mean": float(np.mean(hitrate_list)) if hitrate_list else float("nan"),
        "Months_evaluated": int(len(hitrate_list)),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ml_train = pd.read_parquet(ML_TRAIN_PATH)
    ml_test = pd.read_parquet(ML_TEST_PATH)

    ret_train = pd.read_parquet(RET_TRAIN_PATH)
    ret_test = pd.read_parquet(RET_TEST_PATH)

    feature_cols = [c for c in ml_train.columns if c.startswith(("ret_", "vol_", "rsi_"))]
    target_cols = [c for c in ml_train.columns if c.startswith("y_next")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly 1 target column, found: {target_cols}")
    target_col = target_cols[0]

    top_pct = getattr(config, "TOP_PERCENTAGE", 0.20)

    artifacts = fit_xgboost_with_scaler(
        train_df=ml_train,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    pred_train = predict_returns(artifacts, ml_train, pred_col="pred_return")
    pred_test = predict_returns(artifacts, ml_test, pred_col="pred_return")

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
        w_train, ret_train, use_log_returns=config.USE_LOG_RETURNS
    )
    equity_train = compute_equity_curve(port_ret_train)

    port_ret_test = compute_portfolio_returns(
        w_test, ret_test, use_log_returns=config.USE_LOG_RETURNS
    )
    equity_test = compute_equity_curve(port_ret_test)

    metrics_train = summarize_metrics(port_ret_train, equity_train, w_train)
    metrics_test = summarize_metrics(port_ret_test, equity_test, w_test)

    equity_train.to_csv(EQUITY_TRAIN_PATH)
    equity_test.to_csv(EQUITY_TEST_PATH)

    with open(METRICS_TRAIN_PATH, "w") as f:
        json.dump(metrics_train, f, indent=4)
    with open(METRICS_TEST_PATH, "w") as f:
        json.dump(metrics_test, f, indent=4)

    print("\n== XGBoost experiment saved to:", RESULTS_DIR)

    print("\n=== TRAIN STRATEGY METRICS (2015–2024) ===")
    for k, v in metrics_train.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TEST STRATEGY METRICS (2025) ===")
    for k, v in metrics_test.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()