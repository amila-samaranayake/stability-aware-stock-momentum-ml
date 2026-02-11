# src/data_download.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import yfinance as yf


@dataclass
class DownloadResult:
    adj_close: pd.DataFrame
    missing_ratio: pd.Series


def download_adj_close(
    tickers: List[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = False,
) -> DownloadResult:
    """
    Download Adjusted Close prices for given tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        Yahoo Finance tickers (e.g., 'VOD.L', 'AZN.L').
    start_date : str
        Inclusive start date in YYYY-MM-DD.
    end_date : str
        Inclusive end date in YYYY-MM-DD.
    auto_adjust : bool
        If True, yfinance auto-adjusts OHLC. We keep False and use 'Adj Close'.

    Returns
    -------
    DownloadResult
        Contains Adj Close DataFrame (Date index) and missing ratios per ticker.
    """
    if not tickers:
        raise ValueError("Tickers list is empty.")

    # yfinance supports multiple tickers in one call
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by="column",
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    # When multiple tickers: columns are a MultiIndex (field, ticker)
    # We only want Adjusted Close.
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" not in df.columns.get_level_values(0)) and ("Close" in df.columns.get_level_values(0)):
            # Fallback if Adj Close is missing
            adj = df["Close"].copy()
        else:
            adj = df["Adj Close"].copy()
    else:
        # Single ticker case: columns are flat
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        adj = df[[col]].copy()
        # Standardize to have ticker as column name if possible
        if len(tickers) == 1:
            adj.columns = [tickers[0]]

    # Ensure DateTimeIndex
    adj.index = pd.to_datetime(adj.index)

    # Basic missingness report
    missing_ratio = adj.isna().mean().sort_values(ascending=False)

    return DownloadResult(adj_close=adj, missing_ratio=missing_ratio)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a dataframe to parquet (preferred) or csv based on file extension.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if filepath.endswith(".parquet"):
        df.to_parquet(filepath)
    elif filepath.endswith(".csv"):
        df.to_csv(filepath, index=True)
    else:
        raise ValueError("Unsupported file format. Use .parquet or .csv")


def print_missing_summary(missing_ratio: pd.Series, top_n: int = 10) -> None:
    """
    Print a quick missing data summary.
    """
    print("\nMissing ratio (top offenders):")
    print(missing_ratio.head(top_n))
    print(f"\nTickers with >10% missing: {(missing_ratio > 0.10).sum()}")
