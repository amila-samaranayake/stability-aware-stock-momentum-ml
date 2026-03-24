# src/evaluation/backtest.py

from __future__ import annotations
import pandas as pd


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns_monthly: pd.DataFrame
) -> pd.Series:
    """
    Compute portfolio returns from weights and asset returns.

    IMPORTANT:
    - weights at time t are applied to returns at time t+1
      (so decisions are made using information up to t).

    weights: DataFrame indexed by month-end dates, columns = tickers.
    returns_monthly: DataFrame indexed by month-end dates, columns = tickers.

    Returns:
    - Series of portfolio returns indexed by month-end dates (aligned to t+1).
    """
    # Align columns and index
    common_cols = weights.columns.intersection(returns_monthly.columns)
    w = weights[common_cols].copy()
    r = returns_monthly[common_cols].copy()

    # Shift weights forward: use weights decided at t for next month's returns
    w_next = w.shift(1)

    # Portfolio return at month t = sum_i w_next[t, i] * r[t, i]
    port_ret = (w_next * r).sum(axis=1)

    # Drop first row (will be 0/NaN due to shift)
    port_ret = port_ret.dropna()

    return port_ret


def compute_equity_curve(portfolio_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """
    Convert returns series to an equity curve (cumulative growth).
    """
    equity = (1.0 + portfolio_returns).cumprod() * start_value
    return equity
