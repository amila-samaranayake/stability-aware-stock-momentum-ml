# src/strategies/momentum.py

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_signal(
    returns_monthly: pd.DataFrame,
    lookback_months: int = 12
) -> pd.DataFrame:
    """
    Compute momentum signal using past cumulative returns over `lookback_months`.

    Signal at month t is based ONLY on months (t-lookback_months ... t-1),
    so we shift by 1 to avoid look-ahead bias.

    returns_monthly: DataFrame indexed by month-end dates, columns = tickers.
    """
    # Use log-style compounding without logs: Π(1+r) - 1
    past_cum = (1.0 + returns_monthly).rolling(window=lookback_months).apply(
        lambda x: np.prod(x) - 1.0,
        raw=True
    )

    # Shift by 1 month so signal at t uses data up to t-1
    signal = past_cum.shift(1)
    return signal


def select_top_assets(
    signal: pd.DataFrame,
    top_pct: float = 0.20,
    min_assets: int = 1
) -> pd.DataFrame:
    """
    Select top assets each month based on the signal.

    Returns a boolean DataFrame (same shape as signal):
    True = selected in portfolio for that month.
    """
    if not (0 < top_pct <= 1):
        raise ValueError("top_pct must be in (0, 1].")

    n_assets = signal.shape[1]
    k = max(min_assets, int(np.ceil(n_assets * top_pct)))

    # For each row (month), mark top k tickers by signal
    selected = pd.DataFrame(False, index=signal.index, columns=signal.columns)

    for dt in signal.index:
        row = signal.loc[dt].dropna()
        if row.empty:
            continue
        top = row.nlargest(min(k, len(row))).index
        selected.loc[dt, top] = True

    return selected


def build_equal_weight_weights(
    selected: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert selection boolean DF into equal-weight portfolio weights.

    If no assets selected in a month, weights are all zeros for that month.
    """
    weights = selected.astype(float)
    row_sums = weights.sum(axis=1)

    # Avoid divide-by-zero: only normalize rows where sum > 0
    weights = weights.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return weights


def build_momentum_portfolio(
    returns_monthly: pd.DataFrame,
    lookback_months: int = 12,
    top_pct: float = 0.20
) -> dict:
    """
    Full momentum pipeline:
    - compute signal
    - select top assets
    - compute equal weights

    Returns a dict with signal, selected mask, and weights.
    """
    signal = compute_momentum_signal(returns_monthly, lookback_months=lookback_months)
    selected = select_top_assets(signal, top_pct=top_pct)
    weights = build_equal_weight_weights(selected)

    return {
        "signal": signal,
        "selected": selected,
        "weights": weights
    }
