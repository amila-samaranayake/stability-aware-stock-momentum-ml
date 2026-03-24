# src/evaluation/backtest.py

from __future__ import annotations
import numpy as np
import pandas as pd


def to_simple_returns(returns: pd.DataFrame | pd.Series, use_log_returns: bool) -> pd.DataFrame | pd.Series:
    """
    Convert returns to simple returns if needed.
    - If use_log_returns=True: simple = exp(log) - 1
    - Else: already simple
    """
    if not use_log_returns:
        return returns
    return np.exp(returns) - 1.0


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns_monthly: pd.DataFrame,
    use_log_returns: bool = False
) -> pd.Series:
    """
    Compute portfolio SIMPLE returns from weights and asset returns.

    Timing:
    - weights at time t are applied to returns at time t+1 (weights shifted by 1).

    If returns_monthly are log returns, they are converted to simple returns:
      r_simple = exp(r_log) - 1
    """
    # Align columns
    common_cols = weights.columns.intersection(returns_monthly.columns)
    w = weights[common_cols].copy()
    r = returns_monthly[common_cols].copy()

    # Apply timing: use weights decided at t for next month returns
    w_next = w.shift(1)

    # Convert asset returns to simple returns for portfolio math
    r_simple = to_simple_returns(r, use_log_returns=use_log_returns)

    # Portfolio simple return
    port_ret_simple = (w_next * r_simple).sum(axis=1)

    # Drop first row due to shift
    return port_ret_simple.dropna()


def compute_equity_curve(
    portfolio_simple_returns: pd.Series,
    start_value: float = 1.0
) -> pd.Series:
    """
    Equity curve from SIMPLE portfolio returns:
      Equity_t = Equity_{t-1} * (1 + r_t)
    """
    return (1.0 + portfolio_simple_returns).cumprod() * start_value

def apply_transaction_costs(
    portfolio_simple_returns: pd.Series,
    turnover_series: pd.Series,
    cost_rate: float
) -> pd.Series:
    """
    Apply turnover-based transaction costs to portfolio simple returns.

    Parameters
    ----------
    portfolio_simple_returns : pd.Series
        Gross portfolio returns (already in SIMPLE return form).
    turnover_series : pd.Series
        Turnover per period, typically in [0, 1].
    cost_rate : float
        Transaction cost rate per unit turnover.

        Examples:
        - 0.0005 = 5 bps
        - 0.0010 = 10 bps
        - 0.0020 = 20 bps

    Returns
    -------
    pd.Series
        Net portfolio returns after transaction costs.
    """
    aligned_turnover = turnover_series.reindex(portfolio_simple_returns.index).fillna(0.0)
    transaction_cost = aligned_turnover * cost_rate
    net_returns = portfolio_simple_returns - transaction_cost
    return net_returns