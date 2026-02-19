# src/run_baseline.py

import os
import json
import pandas as pd

from src.config import LOOKBACK_MONTHS, TOP_PERCENTAGE
from src.strategies.momentum import build_momentum_portfolio
from src.evaluation.backtest import compute_portfolio_returns, compute_equity_curve
from src.evaluation.metrics import summarize_metrics
from src.plotting import plot_equity_curve, plot_drawdown
from src.evaluation.metrics import turnover
from src.plotting import plot_turnover

# Paths
TRAIN_PATH = "data/processed/train_monthly_2015_2024.parquet"
RESULTS_DIR = "experiments/results/exp01_baseline"
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
EQUITY_PATH = os.path.join(RESULTS_DIR, "equity_curve.csv")


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Load monthly training returns
    returns_train = pd.read_parquet(TRAIN_PATH)

    # 2) Build momentum portfolio
    momentum_output = build_momentum_portfolio(
        returns_monthly=returns_train,
        lookback_months=LOOKBACK_MONTHS,
        top_pct=TOP_PERCENTAGE,
    )

    weights = momentum_output["weights"]

    # 3) Backtest
    portfolio_returns = compute_portfolio_returns(weights, returns_train)
    equity_curve = compute_equity_curve(portfolio_returns)

    # 4) Metrics
    metrics = summarize_metrics(
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        weights=weights,
    )

    # 5) Save outputs
    equity_curve.to_csv(EQUITY_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Baseline results saved to:", RESULTS_DIR)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Plot results
    plot_equity_curve(equity_curve, title="Momentum Strategy Equity Curve (Train 2015–2024)")
    plot_drawdown(equity_curve, title="Momentum Strategy Drawdown (Train 2015–2024)")

    # Optional: plot turnover
    t_series = turnover(weights)
    plot_turnover(t_series, title="Momentum Strategy Turnover")



if __name__ == "__main__":
    main()
