# src/run_baseline.py

import os
import json
import pandas as pd

from src.config import LOOKBACK_MONTHS, TOP_PERCENTAGE
from src.strategies.momentum import build_momentum_portfolio
from src.evaluation.backtest import compute_portfolio_returns, compute_equity_curve
from src.evaluation.metrics import summarize_metrics

TRAIN_PATH = "data/processed/train_monthly_2015_2024.parquet"
TEST_PATH = "data/processed/test_monthly_2025.parquet"

RESULTS_DIR = "experiments/results/exp01_baseline"
METRICS_TRAIN_PATH = os.path.join(RESULTS_DIR, "metrics_train.json")
METRICS_TEST_PATH = os.path.join(RESULTS_DIR, "metrics_test_2025.json")
EQUITY_TRAIN_PATH = os.path.join(RESULTS_DIR, "equity_train.csv")
EQUITY_TEST_PATH = os.path.join(RESULTS_DIR, "equity_test_2025.csv")


def run_momentum(returns_df: pd.DataFrame):
    out = build_momentum_portfolio(
        returns_monthly=returns_df,
        lookback_months=LOOKBACK_MONTHS,
        top_pct=TOP_PERCENTAGE,
    )
    weights = out["weights"]
    port_ret = compute_portfolio_returns(weights, returns_df)
    equity = compute_equity_curve(port_ret)
    metrics = summarize_metrics(port_ret, equity, weights)
    return metrics, equity


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Train ---
    returns_train = pd.read_parquet(TRAIN_PATH)
    metrics_train, equity_train = run_momentum(returns_train)

    equity_train.to_csv(EQUITY_TRAIN_PATH)
    with open(METRICS_TRAIN_PATH, "w") as f:
        json.dump(metrics_train, f, indent=4)

    # --- Test (2025) ---
    returns_test = pd.read_parquet(TEST_PATH)

    # IMPORTANT: momentum needs lookback history.
    # For a clean 2025 test, we should build signals using history that includes 2024.
    # Easiest approach: concatenate train + test, then evaluate only 2025.
    returns_all = pd.concat([returns_train, returns_test]).sort_index()

    out_all = build_momentum_portfolio(
        returns_monthly=returns_all,
        lookback_months=LOOKBACK_MONTHS,
        top_pct=TOP_PERCENTAGE,
    )
    weights_all = out_all["weights"]

    # Slice weights and returns to 2025 for test evaluation
    w_2025 = weights_all.loc[returns_test.index]
    port_ret_2025 = compute_portfolio_returns(w_2025, returns_test)
    equity_2025 = compute_equity_curve(port_ret_2025)

    metrics_2025 = summarize_metrics(port_ret_2025, equity_2025, w_2025)

    equity_2025.to_csv(EQUITY_TEST_PATH)
    with open(METRICS_TEST_PATH, "w") as f:
        json.dump(metrics_2025, f, indent=4)

    # --- Print ---
    print("Baseline results saved to:", RESULTS_DIR)

    print("\n=== TRAIN STRATEGY METRICS (2015–2024) ===")
    for k, v in metrics_train.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TEST STRATEGY METRICS (2025) ===")
    for k, v in metrics_2025.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()