# src/preprocessing.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessResult:
    returns_daily: pd.DataFrame
    returns_monthly: pd.DataFrame
    train_monthly: pd.DataFrame
    test_monthly: pd.DataFrame


def compute_returns(adj_close: pd.DataFrame, use_log_returns: bool = False) -> pd.DataFrame:
    """
    Compute daily returns from adjusted close prices.

    Parameters
    ----------
    adj_close : pd.DataFrame
        Daily adjusted close prices with DateTimeIndex and tickers as columns.
    use_log_returns : bool
        If True, compute log returns.
        If False, compute simple returns.

    Returns
    -------
    pd.DataFrame
        Daily return series.
    """
    adj_close = adj_close.sort_index()

    if use_log_returns:
        returns = np.log(adj_close).diff()
    else:
        returns = adj_close.pct_change(fill_method=None)

    return returns


def daily_to_monthly_compound(
    returns_daily: pd.DataFrame,
    use_log_returns: bool = False
) -> pd.DataFrame:
    """
    Convert daily returns to monthly returns.

    Parameters
    ----------
    returns_daily : pd.DataFrame
        Daily return series.
    use_log_returns : bool
        If True, sum log returns within month.
        If False, compound simple returns within month.

    Returns
    -------
    pd.DataFrame
        Monthly return series.
    """
    returns_daily = returns_daily.sort_index()

    if use_log_returns:
        monthly = returns_daily.resample("ME").sum()
    else:
        monthly = (1.0 + returns_daily).resample("ME").prod() - 1.0

    return monthly


def drop_tickers_with_missing(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.10
) -> pd.DataFrame:
    """
    Drop tickers that:
    1) are entirely missing
    2) exceed the allowed missing ratio
    """
    non_all_null = df.columns[df.notna().any(axis=0)]
    df = df[non_all_null]

    missing_ratio = df.isna().mean()
    keep = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    return df[keep]


def drop_tickers_by_monthly_coverage(
    monthly_returns: pd.DataFrame,
    train_end_date: str,
    min_coverage: float = 0.95
) -> pd.DataFrame:
    """
    Keep tickers with sufficient monthly coverage in the training period.
    """
    train = monthly_returns.loc[:train_end_date]
    coverage = train.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    return monthly_returns[keep]


def fill_small_gaps(
    df: pd.DataFrame,
    max_consecutive_nans: int = 1
) -> pd.DataFrame:
    """
    Forward-fill only small gaps to avoid inventing too much data.
    """
    return df.ffill(limit=max_consecutive_nans)


def split_train_test_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    test_start_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test by date.

    Supports:
    - DatetimeIndex
    - MultiIndex with first level named 'date'
    """
    if isinstance(df.index, pd.MultiIndex):
        dates = pd.to_datetime(df.index.get_level_values("date"))
        train = df[dates <= pd.Timestamp(train_end_date)].copy()
        test = df[dates >= pd.Timestamp(test_start_date)].copy()
    else:
        df = df.sort_index()
        train = df.loc[:train_end_date].copy()
        test = df.loc[test_start_date:].copy()

    return train, test


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Save dataframe to parquet or csv depending on file extension.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if filepath.endswith(".parquet"):
        df.to_parquet(filepath)
    elif filepath.endswith(".csv"):
        df.to_csv(filepath, index=True)
    else:
        raise ValueError("Unsupported format. Use .parquet or .csv")


def preprocess_prices_to_returns(
    adj_close: pd.DataFrame,
    train_end_date: str,
    test_start_date: str,
    max_missing_ratio: float = 0.10,
    fill_gap_limit: int = 1,
    use_log_returns: bool = False
) -> PreprocessResult:
    """
    Monthly benchmark preprocessing pipeline.

    Steps
    -----
    1) Compute daily returns
    2) Convert daily returns to monthly returns
    3) Drop tickers with excessive missingness
    4) Drop tickers with insufficient monthly history in train period
    5) Fill only very small gaps
    6) Split into train and test monthly data
    """
    returns_daily = compute_returns(adj_close, use_log_returns=use_log_returns)
    returns_monthly = daily_to_monthly_compound(
        returns_daily,
        use_log_returns=use_log_returns,
    )

    all_null = returns_monthly.columns[~returns_monthly.notna().any(axis=0)]
    print("All-null tickers (monthly):", list(all_null))

    print("Before missing filter:", returns_monthly.shape)
    returns_monthly = drop_tickers_with_missing(
        returns_monthly,
        max_missing_ratio=max_missing_ratio,
    )

    before_cov = returns_monthly.shape
    returns_monthly = drop_tickers_by_monthly_coverage(
        monthly_returns=returns_monthly,
        train_end_date=train_end_date,
        min_coverage=0.95,
    )
    print("After coverage filter:", before_cov, "->", returns_monthly.shape)

    returns_monthly = fill_small_gaps(
        returns_monthly,
        max_consecutive_nans=fill_gap_limit,
    )

    train_monthly, test_monthly = split_train_test_by_date(
        returns_monthly,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
    )

    return PreprocessResult(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        train_monthly=train_monthly,
        test_monthly=test_monthly,
    )


def basic_sanity_report(monthly_returns: pd.DataFrame) -> None:
    """
    Quick sanity checks for monthly return data.
    """
    print("Monthly returns shape:", monthly_returns.shape)
    print("Date range:", monthly_returns.index.min(), "->", monthly_returns.index.max())

    desc = monthly_returns.stack().describe(percentiles=[0.01, 0.05, 0.95, 0.99])
    print("\nMonthly returns distribution:")
    print(desc)

    missing = monthly_returns.isna().mean().sort_values(ascending=False)
    print("\nMissing ratio (top 10):")
    print(missing.head(10))