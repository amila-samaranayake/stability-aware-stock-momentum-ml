# src/evaluation/metrics.py

from __future__ import annotations
import numpy as np
import pandas as pd


def cumulative_return(portfolio_returns: pd.Series) -> float:
    """Total cumulative return over the period."""
    return float((1.0 + portfolio_returns).prod() - 1.0)


def annualized_return(portfolio_returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized return assuming monthly data by default."""
    n = portfolio_returns.shape[0]
    if n == 0:
        return np.nan
    growth = (1.0 + portfolio_returns).prod()
    return float(growth ** (periods_per_year / n) - 1.0)


def annualized_volatility(portfolio_returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized volatility for periodic returns."""
    if portfolio_returns.shape[0] < 2:
        return np.nan
    return float(portfolio_returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown from an equity curve.
    Returns a negative number (e.g., -0.35 means -35% max drawdown).
    """
    if equity_curve.empty:
        return np.nan
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    return float(drawdowns.min())


def sharpe_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> float:
    """
    Simple Sharpe ratio with optional risk-free rate (per period risk-free assumed 0 by default).
    """
    if portfolio_returns.shape[0] < 2:
        return np.nan
    excess = portfolio_returns - risk_free_rate
    vol = excess.std(ddof=1)
    if vol == 0:
        return np.nan
    return float((excess.mean() / vol) * np.sqrt(periods_per_year))


def turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute portfolio turnover over time for equal-weight long-only portfolios.

    Turnover definition (simple, selection-based):
    turnover_t = (# assets removed from portfolio at t) / (portfolio size at t-1)

    We compute using the change in holdings (non-zero weights).

    Returns:
    - Series indexed by time (same as weights index), with turnover in [0, 1].
    """
    # Holdings indicator: 1 if weight > 0 else 0
    hold = (weights > 0).astype(int)

    # Compare holdings month-to-month
    prev = hold.shift(1)

    # Assets removed: were held before (prev=1) and not held now (hold=0)
    removed = ((prev == 1) & (hold == 0)).sum(axis=1)

    # Portfolio size last period
    prev_size = prev.sum(axis=1).replace(0, np.nan)

    t = (removed / prev_size).fillna(0.0)

    # First period turnover is 0 by convention (no previous portfolio)
    t.iloc[0] = 0.0
    return t


def summarize_metrics(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    weights: pd.DataFrame
) -> dict:
    """
    Return a dictionary of key metrics for reporting.
    """
    t = turnover(weights)
    return {
        "cumulative_return": cumulative_return(portfolio_returns),
        "annualized_return": annualized_return(portfolio_returns),
        "annualized_volatility": annualized_volatility(portfolio_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "avg_turnover": float(t.mean()),
        "median_turnover": float(t.median()),
        "max_turnover": float(t.max()),
    }
