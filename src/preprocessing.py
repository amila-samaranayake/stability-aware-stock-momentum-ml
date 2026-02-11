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


def compute_simple_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns from Adjusted Close prices.
    Return_t = (P_t / P_{t-1}) - 1

    adj_close: DataFrame with DateTimeIndex and tickers as columns.
    """
    adj_close = adj_close.sort_index()
    returns = adj_close.pct_change()
    return returns


def daily_to_monthly_compound(returns_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily returns to monthly returns by compounding:
    monthly_return = Π(1 + r_daily) - 1 over each month.

    returns_daily: DataFrame of daily returns.
    """
    returns_daily = returns_daily.sort_index()

    # Compound within each calendar month
    monthly = (1.0 + returns_daily).resample("M").prod() - 1.0
    return monthly


def drop_tickers_with_missing(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.10
) -> pd.DataFrame:
    """
    Drop tickers (columns) that have too much missing data.
    """
    missing_ratio = df.isna().mean()
    keep = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    return df[keep]


def fill_small_gaps(df: pd.DataFrame, max_consecutive_nans: int = 1) -> pd.DataFrame:
    """
    Fill small gaps only (to avoid inventing data).
    - Forward-fill up to `max_consecutive_nans` consecutive NaNs.
    """
    # limit=1 means fill at most 1 NaN in a row
    return df.ffill(limit=max_consecutive_nans)


def split_train_test_by_date(
    monthly_returns: pd.DataFrame,
    train_end_date: str,
    test_start_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split monthly returns into train and test using date boundaries.
    """
    monthly_returns = monthly_returns.sort_index()
    train = monthly_returns.loc[:train_end_date].copy()
    test = monthly_returns.loc[test_start_date:].copy()
    return train, test


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Save dataframe to parquet or csv.
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
    fill_gap_limit: int = 1
) -> PreprocessResult:
    """
    Full preprocessing pipeline:
    1) Daily simple returns
    2) Monthly compounded returns
    3) Drop tickers with too many missing values
    4) Fill small gaps (optional, conservative)
    5) Split train/test by date
    """
    # 1) Daily returns
    returns_daily = compute_simple_returns(adj_close)

    # 2) Monthly returns (compounded)
    returns_monthly = daily_to_monthly_compound(returns_daily)

    # 3) Drop tickers with lots of missing data (on monthly)
    returns_monthly = drop_tickers_with_missing(returns_monthly, max_missing_ratio=max_missing_ratio)

    # 4) Fill small gaps only (conservative)
    returns_monthly = fill_small_gaps(returns_monthly, max_consecutive_nans=fill_gap_limit)

    # 5) Split
    train_monthly, test_monthly = split_train_test_by_date(
        returns_monthly, train_end_date=train_end_date, test_start_date=test_start_date
    )

    return PreprocessResult(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        train_monthly=train_monthly,
        test_monthly=test_monthly
    )


def basic_sanity_report(monthly_returns: pd.DataFrame) -> None:
    """
    Quick sanity checks to catch obvious issues early.
    """
    print("Monthly returns shape:", monthly_returns.shape)
    print("Date range:", monthly_returns.index.min(), "->", monthly_returns.index.max())

    # Check typical magnitude (returns should not be huge regularly)
    desc = monthly_returns.stack().describe(percentiles=[0.01, 0.05, 0.95, 0.99])
    print("\nMonthly returns distribution (stacked across tickers):")
    print(desc)

    # Missingness
    missing = monthly_returns.isna().mean().sort_values(ascending=False)
    print("\nMissing ratio (top 10):")
    print(missing.head(10))
