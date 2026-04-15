from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yfinance as yf


@dataclass
class DownloadResult:
    adj_close: pd.DataFrame
    missing_ratio: pd.Series


@dataclass
class OHLCVDownloadResult:
    ohlcv: pd.DataFrame
    adj_close: pd.DataFrame
    missing_ratio_adj_close: pd.Series


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Save dataframe to parquet.
    """
    df.to_parquet(path)


def print_missing_summary(missing_ratio: pd.Series, top_n: int = 20) -> None:
    """
    Print missing-value ratios by ticker.
    """
    print("\n=== Missing Ratio Summary ===")
    print(missing_ratio.sort_values(ascending=False).head(top_n))


def download_adj_close(
    tickers: list[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = False,
) -> DownloadResult:
    """
    Download adjusted close prices only.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            adj_close = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            adj_close = data["Close"].copy()
        else:
            raise ValueError("Could not find 'Adj Close' or 'Close' in downloaded data.")
    else:
        raise ValueError("Expected MultiIndex columns from yfinance download.")

    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close = adj_close.sort_index()

    missing_ratio = adj_close.isna().mean()

    return DownloadResult(
        adj_close=adj_close,
        missing_ratio=missing_ratio,
    )


def download_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = False,
) -> OHLCVDownloadResult:
    """
    Download full OHLCV data and also extract adjusted close.

    Output format:
    - ohlcv: MultiIndex columns (field, ticker)
      fields expected: Open, High, Low, Close, Adj Close, Volume
    - adj_close: flat ticker columns
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from yfinance download.")

    available_fields = list(pd.Index(data.columns.get_level_values(0)).unique())

    required_base_fields = ["Open", "High", "Low", "Close", "Volume"]
    missing_base_fields = [f for f in required_base_fields if f not in available_fields]
    if missing_base_fields:
        raise ValueError(f"Missing required OHLCV fields: {missing_base_fields}")

    if "Adj Close" in available_fields:
        adj_close = data["Adj Close"].copy()
    else:
        adj_close = data["Close"].copy()

    desired_fields = [f for f in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if f in available_fields]
    ohlcv = data[desired_fields].copy()

    ohlcv.index = pd.to_datetime(ohlcv.index)
    ohlcv = ohlcv.sort_index()

    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close = adj_close.sort_index()

    missing_ratio_adj_close = adj_close.isna().mean()

    return OHLCVDownloadResult(
        ohlcv=ohlcv,
        adj_close=adj_close,
        missing_ratio_adj_close=missing_ratio_adj_close,
    )