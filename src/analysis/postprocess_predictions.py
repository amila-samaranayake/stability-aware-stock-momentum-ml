from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.evaluation.backtest import (
    apply_transaction_costs,
    compute_equity_curve,
    compute_portfolio_returns,
)
from src.evaluation.metrics import summarize_metrics, turnover
from src.strategies.momentum import build_equal_weight_weights, select_top_assets
from src.utils.paths import get_processed_returns_paths


def load_prediction_file(path: str) -> pd.DataFrame:
    """
    Load saved prediction file.

    Expected columns:
    - date
    - ticker
    - pred_return
    - y_next_1m
    """
    file_path = Path(path)

    if file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    if {"date", "ticker"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "ticker"]).sort_index()

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Prediction file must have (date, ticker) index or date/ticker columns.")

    required_cols = {"pred_return", "y_next_1m"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.copy()


def monthly_winsorize_predictions(
    df: pd.DataFrame,
    pred_col: str = "pred_return",
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> pd.DataFrame:
    """
    Winsorize predictions cross-sectionally within each month.
    """
    out = df.copy()

    def _clip_group(group: pd.DataFrame) -> pd.DataFrame:
        lower = group[pred_col].quantile(lower_q)
        upper = group[pred_col].quantile(upper_q)
        group[pred_col] = group[pred_col].clip(lower=lower, upper=upper)
        return group

    out = out.groupby(level="date", group_keys=False).apply(_clip_group)
    return out


def monthly_shrink_predictions(
    df: pd.DataFrame,
    pred_col: str = "pred_return",
    shrinkage: float = 0.20,
) -> pd.DataFrame:
    """
    Shrink predictions toward the monthly cross-sectional mean.

    new_pred = (1 - shrinkage) * pred + shrinkage * monthly_mean
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1.")

    out = df.copy()

    def _shrink_group(group: pd.DataFrame) -> pd.DataFrame:
        group_mean = group[pred_col].mean()
        group[pred_col] = (1.0 - shrinkage) * group[pred_col] + shrinkage * group_mean
        return group

    out = out.groupby(level="date", group_keys=False).apply(_shrink_group)
    return out


def estimate_stock_bias(
    train_df: pd.DataFrame,
    pred_col: str = "pred_return",
    target_col: str = "y_next_1m",
    min_obs: int = 6,
) -> pd.Series:
    """
    Estimate average stock-level prediction bias from train data.

    bias = mean(pred - actual) for each ticker
    """
    tmp = train_df.copy()
    tmp["error"] = tmp[pred_col] - tmp[target_col]

    grouped = tmp.groupby(level="ticker")["error"].agg(["mean", "count"])
    grouped.loc[grouped["count"] < min_obs, "mean"] = 0.0

    return grouped["mean"]


def apply_stock_bias_correction(
    df: pd.DataFrame,
    stock_bias: pd.Series,
    pred_col: str = "pred_return",
) -> pd.DataFrame:
    """
    Subtract train-estimated stock bias from predictions.
    """
    out = df.copy()

    tickers = out.index.get_level_values("ticker")
    bias_values = pd.Series(tickers, index=out.index).map(stock_bias).fillna(0.0)
    out[pred_col] = out[pred_col] - bias_values.to_numpy(dtype=float)

    return out


def regression_prediction_metrics(
    df_pred: pd.DataFrame,
    target_col: str = "y_next_1m",
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
    target_col: str = "y_next_1m",
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


def predictions_to_weights(
    pred_long: pd.DataFrame,
    top_pct: float,
    pred_col: str = "pred_return",
) -> pd.DataFrame:
    pred_wide = pred_long[pred_col].unstack("ticker").sort_index()
    selected = select_top_assets(signal=pred_wide, top_pct=top_pct)
    return build_equal_weight_weights(selected)


def compute_cost_adjusted_results(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
) -> tuple[pd.Series, dict]:
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
    parser = argparse.ArgumentParser(description="Post-process saved stock return predictions.")
    parser.add_argument("--train-predictions", type=str, required=True, help="Train predictions file.")
    parser.add_argument("--test-predictions", type=str, required=True, help="Test predictions file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--winsor-lower", type=float, default=None, help="Lower monthly winsor quantile, e.g. 0.05")
    parser.add_argument("--winsor-upper", type=float, default=None, help="Upper monthly winsor quantile, e.g. 0.95")
    parser.add_argument("--shrinkage", type=float, default=0.0, help="Monthly shrinkage toward cross-sectional mean.")
    parser.add_argument("--stock-bias-correction", action="store_true", help="Apply stock-level bias correction from train predictions.")
    parser.add_argument("--top-pct", type=float, default=0.20, help="Top fraction for portfolio selection.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_df = load_prediction_file(args.train_predictions)
    test_df = load_prediction_file(args.test_predictions)

    result_config = {
        "winsor_lower": args.winsor_lower,
        "winsor_upper": args.winsor_upper,
        "shrinkage": args.shrinkage,
        "stock_bias_correction": args.stock_bias_correction,
        "top_pct": args.top_pct,
    }

    # Apply same transformation to train and test predictions
    if args.winsor_lower is not None and args.winsor_upper is not None:
        train_df = monthly_winsorize_predictions(
            train_df,
            pred_col="pred_return",
            lower_q=args.winsor_lower,
            upper_q=args.winsor_upper,
        )
        test_df = monthly_winsorize_predictions(
            test_df,
            pred_col="pred_return",
            lower_q=args.winsor_lower,
            upper_q=args.winsor_upper,
        )

    if args.shrinkage > 0.0:
        train_df = monthly_shrink_predictions(train_df, pred_col="pred_return", shrinkage=args.shrinkage)
        test_df = monthly_shrink_predictions(test_df, pred_col="pred_return", shrinkage=args.shrinkage)

    if args.stock_bias_correction:
        stock_bias = estimate_stock_bias(train_df, pred_col="pred_return", target_col="y_next_1m")
        train_df = apply_stock_bias_correction(train_df, stock_bias, pred_col="pred_return")
        test_df = apply_stock_bias_correction(test_df, stock_bias, pred_col="pred_return")
        stock_bias.to_csv(output_dir / "estimated_stock_bias.csv", header=True)

    # Prediction metrics
    pred_metrics = {
        "train": {
            "regression": regression_prediction_metrics(train_df),
            "ranking": ranking_metrics_by_month(train_df, top_pct=args.top_pct),
        },
        "test_2025": {
            "regression": regression_prediction_metrics(test_df),
            "ranking": ranking_metrics_by_month(test_df, top_pct=args.top_pct),
        },
    }

    with open(output_dir / "prediction_metrics_postprocessed.json", "w") as f:
        json.dump(pred_metrics, f, indent=4)

    # Portfolio backtest on test
    returns_paths = get_processed_returns_paths()
    ret_test = pd.read_parquet(returns_paths["test_monthly"])

    weights_test = predictions_to_weights(test_df, top_pct=args.top_pct, pred_col="pred_return")
    weights_test = weights_test[ret_test.columns.intersection(weights_test.columns)]

    port_ret_test = compute_portfolio_returns(
        weights_test,
        ret_test,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    equity_test = compute_equity_curve(port_ret_test)
    strategy_metrics_test = summarize_metrics(port_ret_test, equity_test, weights_test)
    turnover_test, cost_results_test = compute_cost_adjusted_results(port_ret_test, weights_test)

    with open(output_dir / "strategy_metrics_test_postprocessed.json", "w") as f:
        json.dump(strategy_metrics_test, f, indent=4)

    with open(output_dir / "strategy_metrics_test_postprocessed_with_costs.json", "w") as f:
        json.dump(cost_results_test, f, indent=4)

    with open(output_dir / "postprocess_config.json", "w") as f:
        json.dump(result_config, f, indent=4)

    test_df.reset_index().to_csv(output_dir / "test_predictions_postprocessed.csv", index=False)
    weights_test.to_csv(output_dir / "weights_test_postprocessed.csv")
    equity_test.to_csv(output_dir / "equity_test_postprocessed.csv")

    print("Saved post-processed outputs to:", output_dir)

    print("\n=== TEST PREDICTION METRICS ===")
    for k, v in pred_metrics["test_2025"]["regression"].items():
        print(f"{k}: {v:.6f}")
    for k, v in pred_metrics["test_2025"]["ranking"].items():
        print(f"{k}: {v}")

    print("\n=== TEST STRATEGY METRICS ===")
    for k, v in strategy_metrics_test.items():
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