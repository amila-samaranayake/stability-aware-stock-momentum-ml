# src/run_baseline.py

import os
import json
import pandas as pd

from src import config
from src.config import LOOKBACK_MONTHS, TOP_PERCENTAGE
from src.strategies.momentum import build_momentum_portfolio
from src.evaluation.backtest import compute_portfolio_returns, compute_equity_curve
from src.evaluation.metrics import summarize_metrics, turnover
from src.plotting import plot_equity_curve, plot_drawdown, plot_turnover

TRAIN_PATH = "data/processed/train_monthly_2015_2024.parquet"
TEST_PATH = "data/processed/test_monthly_2025.parquet"

RESULTS_DIR = "experiments/results/exp01_baseline"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

METRICS_TRAIN_PATH = os.path.join(RESULTS_DIR, "metrics_train.json")
METRICS_TEST_PATH = os.path.join(RESULTS_DIR, "metrics_test_2025.json")
EQUITY_TRAIN_PATH = os.path.join(RESULTS_DIR, "equity_train.csv")
EQUITY_TEST_PATH = os.path.join(RESULTS_DIR, "equity_test_2025.csv")
WEIGHTS_TRAIN_PATH = os.path.join(RESULTS_DIR, "weights_train.csv")
WEIGHTS_TEST_PATH = os.path.join(RESULTS_DIR, "weights_test_2025.csv")


def run_momentum(returns_df: pd.DataFrame):
    """
    Build momentum portfolio, run backtest, and compute metrics for a given returns DataFrame.
    """
    out = build_momentum_portfolio(
        returns_monthly=returns_df,
        lookback_months=LOOKBACK_MONTHS,
        top_pct=TOP_PERCENTAGE,
        use_log_returns=config.USE_LOG_RETURNS,
    )

    weights = out["weights"]

    port_ret = compute_portfolio_returns(
        weights=weights,
        returns_monthly=returns_df,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    equity = compute_equity_curve(port_ret)
    metrics = summarize_metrics(port_ret, equity, weights)

    return metrics, equity, weights


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =========================
    # TRAIN (2015–2024)
    # =========================
    returns_train = pd.read_parquet(TRAIN_PATH)

    metrics_train, equity_train, weights_train = run_momentum(returns_train)

    equity_train.to_csv(EQUITY_TRAIN_PATH)
    weights_train.to_csv(WEIGHTS_TRAIN_PATH)

    with open(METRICS_TRAIN_PATH, "w") as f:
        json.dump(metrics_train, f, indent=4)

    turnover_train = turnover(weights_train)

    # Plots: train
    plot_equity_curve(
        equity_train,
        title="Baseline Momentum Equity Curve (Train 2015–2024)",
        save_path=os.path.join(FIGURES_DIR, "equity_train.png"),
    )
    plot_drawdown(
        equity_train,
        title="Baseline Momentum Drawdown (Train 2015–2024)",
        save_path=os.path.join(FIGURES_DIR, "drawdown_train.png"),
    )
    plot_turnover(
        turnover_train,
        title="Baseline Momentum Turnover (Train 2015–2024)",
        save_path=os.path.join(FIGURES_DIR, "turnover_train.png"),
    )

    # =========================
    # TEST (2025)
    # =========================
    returns_test = pd.read_parquet(TEST_PATH)

    # Need prior history for 12-month momentum signal
    returns_all = pd.concat([returns_train, returns_test]).sort_index()

    out_all = build_momentum_portfolio(
        returns_monthly=returns_all,
        lookback_months=LOOKBACK_MONTHS,
        top_pct=TOP_PERCENTAGE,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    weights_all = out_all["weights"]

    # Slice 2025 only
    weights_test = weights_all.loc[returns_test.index]

    port_ret_test = compute_portfolio_returns(
        weights=weights_test,
        returns_monthly=returns_test,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    equity_test = compute_equity_curve(port_ret_test)
    metrics_test = summarize_metrics(port_ret_test, equity_test, weights_test)

    equity_test.to_csv(EQUITY_TEST_PATH)
    weights_test.to_csv(WEIGHTS_TEST_PATH)

    with open(METRICS_TEST_PATH, "w") as f:
        json.dump(metrics_test, f, indent=4)

    turnover_test = turnover(weights_test)

    # Plots: test
    plot_equity_curve(
        equity_test,
        title="Baseline Momentum Equity Curve (Test 2025)",
        save_path=os.path.join(FIGURES_DIR, "equity_test_2025.png"),
    )
    plot_drawdown(
        equity_test,
        title="Baseline Momentum Drawdown (Test 2025)",
        save_path=os.path.join(FIGURES_DIR, "drawdown_test_2025.png"),
    )
    plot_turnover(
        turnover_test,
        title="Baseline Momentum Turnover (Test 2025)",
        save_path=os.path.join(FIGURES_DIR, "turnover_test_2025.png"),
    )

    # =========================
    # PRINT
    # =========================
    print("Baseline results saved to:", RESULTS_DIR)

    print("\n=== TRAIN STRATEGY METRICS (2015–2024) ===")
    for k, v in metrics_train.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TEST STRATEGY METRICS (2025) ===")
    for k, v in metrics_test.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()