# src/run_baseline.py

import os
import json
import pandas as pd

from src import config
from src.strategies.momentum import build_momentum_portfolio
from src.evaluation.backtest import (
    compute_portfolio_returns,
    compute_equity_curve,
    apply_transaction_costs,
)
from src.evaluation.metrics import summarize_metrics, turnover
from src.plotting import plot_equity_curve, plot_drawdown, plot_turnover
from src.utils.paths import get_processed_returns_paths, get_experiment_dir


RETURNS_PATHS = get_processed_returns_paths()

TRAIN_PATH = RETURNS_PATHS["train_monthly"]
TEST_PATH = RETURNS_PATHS["test_monthly"]

RESULTS_DIR = get_experiment_dir("exp01_baseline", "monthly")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

METRICS_TRAIN_PATH = os.path.join(RESULTS_DIR, "metrics_train.json")
METRICS_TEST_PATH = os.path.join(RESULTS_DIR, "metrics_test_2025.json")
METRICS_TRAIN_COSTS_PATH = os.path.join(RESULTS_DIR, "metrics_train_with_costs.json")
METRICS_TEST_COSTS_PATH = os.path.join(RESULTS_DIR, "metrics_test_2025_with_costs.json")

EQUITY_TRAIN_PATH = os.path.join(RESULTS_DIR, "equity_train.csv")
EQUITY_TEST_PATH = os.path.join(RESULTS_DIR, "equity_test_2025.csv")
WEIGHTS_TRAIN_PATH = os.path.join(RESULTS_DIR, "weights_train.csv")
WEIGHTS_TEST_PATH = os.path.join(RESULTS_DIR, "weights_test_2025.csv")


def run_momentum(returns_df: pd.DataFrame):
    """
    Build the baseline momentum portfolio, run the gross backtest,
    and return metrics, equity curve, weights, and gross returns.
    """
    out = build_momentum_portfolio(
        returns_monthly=returns_df,
        lookback_months=config.LOOKBACK_MONTHS,
        top_pct=config.TOP_PERCENTAGE,
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

    return metrics, equity, weights, port_ret


def compute_cost_adjusted_results(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
):
    """
    Compute net performance metrics for each configured transaction cost rate.
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
    Run the baseline monthly momentum strategy on train and test periods,
    save gross and net metrics, and export figures and portfolio outputs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    returns_train = pd.read_parquet(TRAIN_PATH)
    metrics_train, equity_train, weights_train, port_ret_train = run_momentum(returns_train)

    equity_train.to_csv(EQUITY_TRAIN_PATH)
    weights_train.to_csv(WEIGHTS_TRAIN_PATH)

    with open(METRICS_TRAIN_PATH, "w") as f:
        json.dump(metrics_train, f, indent=4)

    turnover_train, cost_results_train = compute_cost_adjusted_results(
        gross_returns=port_ret_train,
        weights=weights_train,
    )

    with open(METRICS_TRAIN_COSTS_PATH, "w") as f:
        json.dump(cost_results_train, f, indent=4)

    plot_equity_curve(
        equity_train,
        title="Baseline Momentum Equity Curve (Train 2015–2024)",
        save_path=os.path.join(FIGURES_DIR, "equity_train.png"),
        label="Baseline Momentum",
    )
    plot_drawdown(
        equity_train,
        title="Baseline Momentum Drawdown (Train 2015–2024)",
        save_path=os.path.join(FIGURES_DIR, "drawdown_train.png"),
        label="Baseline Momentum",
    )
    plot_turnover(
        turnover_train,
        title="Baseline Momentum Turnover (Train 2015–2024)",
        save_path=os.path.join(FIGURES_DIR, "turnover_train.png"),
        label="Baseline Momentum",
    )

    for cost_rate in config.TRANSACTION_COST_RATES:
        net_returns = apply_transaction_costs(port_ret_train, turnover_train, cost_rate)
        net_equity = compute_equity_curve(net_returns)

        plot_equity_curve(
            net_equity,
            title=f"Baseline Net Equity Curve (Train 2015–2024, {int(cost_rate * 10000)} bps)",
            save_path=os.path.join(FIGURES_DIR, f"equity_train_{int(cost_rate * 10000)}bps.png"),
            label=f"Baseline Net ({int(cost_rate * 10000)} bps)",
        )

    returns_test = pd.read_parquet(TEST_PATH)

    returns_all = pd.concat([returns_train, returns_test]).sort_index()
    out_all = build_momentum_portfolio(
        returns_monthly=returns_all,
        lookback_months=config.LOOKBACK_MONTHS,
        top_pct=config.TOP_PERCENTAGE,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    weights_all = out_all["weights"]

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

    turnover_test, cost_results_test = compute_cost_adjusted_results(
        gross_returns=port_ret_test,
        weights=weights_test,
    )

    with open(METRICS_TEST_COSTS_PATH, "w") as f:
        json.dump(cost_results_test, f, indent=4)

    plot_equity_curve(
        equity_test,
        title="Baseline Momentum Equity Curve (Test 2025)",
        save_path=os.path.join(FIGURES_DIR, "equity_test_2025.png"),
        label="Baseline Momentum",
    )
    plot_drawdown(
        equity_test,
        title="Baseline Momentum Drawdown (Test 2025)",
        save_path=os.path.join(FIGURES_DIR, "drawdown_test_2025.png"),
        label="Baseline Momentum",
    )
    plot_turnover(
        turnover_test,
        title="Baseline Momentum Turnover (Test 2025)",
        save_path=os.path.join(FIGURES_DIR, "turnover_test_2025.png"),
        label="Baseline Momentum",
    )

    for cost_rate in config.TRANSACTION_COST_RATES:
        net_returns = apply_transaction_costs(port_ret_test, turnover_test, cost_rate)
        net_equity = compute_equity_curve(net_returns)

        plot_equity_curve(
            net_equity,
            title=f"Baseline Net Equity Curve (Test 2025, {int(cost_rate * 10000)} bps)",
            save_path=os.path.join(FIGURES_DIR, f"equity_test_2025_{int(cost_rate * 10000)}bps.png"),
            label=f"Baseline Net ({int(cost_rate * 10000)} bps)",
        )

    print("Baseline results saved to:", RESULTS_DIR)

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