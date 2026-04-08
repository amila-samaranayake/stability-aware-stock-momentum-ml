# src/features_lstm.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class LSTMSampleSet:
    X: np.ndarray
    y: np.ndarray
    dates: pd.Index
    tickers: pd.Index
    target_name: str
    feature_names: list[str]


def compute_monthly_prices_from_adj_close(
    adj_close_daily: pd.DataFrame,
    rule: str = "ME",
) -> pd.DataFrame:
    adj_close_daily = adj_close_daily.sort_index()
    return adj_close_daily.resample(rule).last()


def compute_simple_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    adj_close = adj_close.sort_index()
    return adj_close.pct_change(fill_method=None)


def daily_to_monthly_compound(
    daily_returns: pd.DataFrame,
    rule: str = "ME",
) -> pd.DataFrame:
    daily_returns = daily_returns.sort_index()
    return (1.0 + daily_returns).resample(rule).prod() - 1.0


def build_next_month_target(
    monthly_returns: pd.DataFrame,
    horizon_months: int = 1,
    target_name: str = "y_next_1m",
) -> pd.DataFrame:
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive.")

    target = monthly_returns.shift(-horizon_months)
    target.columns.name = "ticker"
    return target


def split_lstm_dataset_by_date(
    dataset: LSTMSampleSet,
    train_end_date: str,
    test_start_date: str,
) -> Tuple[LSTMSampleSet, LSTMSampleSet]:
    dates = pd.to_datetime(dataset.dates)

    train_mask = dates <= pd.to_datetime(train_end_date)
    test_mask = dates >= pd.to_datetime(test_start_date)

    train_set = LSTMSampleSet(
        X=dataset.X[train_mask],
        y=dataset.y[train_mask],
        dates=dataset.dates[train_mask],
        tickers=dataset.tickers[train_mask],
        target_name=dataset.target_name,
        feature_names=dataset.feature_names,
    )

    test_set = LSTMSampleSet(
        X=dataset.X[test_mask],
        y=dataset.y[test_mask],
        dates=dataset.dates[test_mask],
        tickers=dataset.tickers[test_mask],
        target_name=dataset.target_name,
        feature_names=dataset.feature_names,
    )

    return train_set, test_set


def lstm_sample_set_to_long_dataframe(dataset: LSTMSampleSet) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dataset.dates), dataset.tickers],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {dataset.target_name: dataset.y.astype(float)},
        index=index,
    ).sort_index()


def save_lstm_sample_set(dataset: LSTMSampleSet, filepath: str) -> None:
    np.savez_compressed(
        filepath,
        X=dataset.X,
        y=dataset.y,
        dates=np.array(dataset.dates.astype(str)),
        tickers=np.array(dataset.tickers.astype(str)),
        target_name=np.array([dataset.target_name]),
        feature_names=np.array(dataset.feature_names, dtype=str),
    )


def load_lstm_sample_set(filepath: str) -> LSTMSampleSet:
    data = np.load(filepath, allow_pickle=True)
    return LSTMSampleSet(
        X=data["X"],
        y=data["y"],
        dates=pd.Index(pd.to_datetime(data["dates"])),
        tickers=pd.Index(data["tickers"].astype(str)),
        target_name=str(data["target_name"][0]),
        feature_names=list(data["feature_names"].astype(str)),
    )


def rolling_volatility(
    daily_returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    return daily_returns.rolling(window=window, min_periods=window).std()


def moving_average_ratio(
    adj_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    ma = adj_close.rolling(window=window, min_periods=window).mean()
    return (adj_close / ma) - 1.0


def rolling_drawdown(
    adj_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    rolling_high = adj_close.rolling(window=window, min_periods=window).max()
    return (adj_close / rolling_high) - 1.0


def compute_rsi(
    adj_close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    delta = adj_close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def normalize_sequence_per_feature(
    seq: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (seq - mean) / std


def build_lstm_multifeature_sequence_dataset(
    adj_close: pd.DataFrame,
    market_ticker: str | None = None,
    sequence_length: int = 60,
    target_horizon_months: int = 1,
    target_name: str = "y_next_1m",
    normalize_per_sequence: bool = True,
) -> LSTMSampleSet:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than 1.")

    adj_close = adj_close.sort_index()

    daily_returns = compute_simple_returns(adj_close)
    monthly_prices = compute_monthly_prices_from_adj_close(adj_close)
    monthly_returns = daily_to_monthly_compound(daily_returns)
    target_wide = build_next_month_target(
        monthly_returns=monthly_returns,
        horizon_months=target_horizon_months,
        target_name=target_name,
    )

    vol_20d = rolling_volatility(daily_returns, window=20)
    vol_60d = rolling_volatility(daily_returns, window=60)
    ma_ratio_20 = moving_average_ratio(adj_close, window=20)
    ma_ratio_60 = moving_average_ratio(adj_close, window=60)
    drawdown_60d = rolling_drawdown(adj_close, window=60)
    rsi_14d = compute_rsi(adj_close, window=14)

    if market_ticker is not None and market_ticker in daily_returns.columns:
        market_returns = daily_returns[market_ticker].copy()
    else:
        market_returns = daily_returns.mean(axis=1)

    common_tickers = (
        adj_close.columns
        .intersection(daily_returns.columns)
        .intersection(monthly_prices.columns)
        .intersection(target_wide.columns)
        .intersection(vol_20d.columns)
        .intersection(vol_60d.columns)
        .intersection(ma_ratio_20.columns)
        .intersection(ma_ratio_60.columns)
        .intersection(drawdown_60d.columns)
        .intersection(rsi_14d.columns)
    )

    if market_ticker in common_tickers:
        common_tickers = common_tickers.drop(market_ticker)

    daily_returns = daily_returns[common_tickers]
    vol_20d = vol_20d[common_tickers]
    vol_60d = vol_60d[common_tickers]
    ma_ratio_20 = ma_ratio_20[common_tickers]
    ma_ratio_60 = ma_ratio_60[common_tickers]
    drawdown_60d = drawdown_60d[common_tickers]
    rsi_14d = rsi_14d[common_tickers]
    monthly_prices = monthly_prices[common_tickers]
    target_wide = target_wide[common_tickers]

    month_end_dates = monthly_prices.index
    daily_index = daily_returns.index
    market_returns = market_returns.reindex(daily_index)

    feature_names = [
        "daily_return",
        "excess_return",
        "market_return",
        "vol_20d",
        "vol_60d",
        "ma_ratio_20",
        "ma_ratio_60",
        "drawdown_60d",
        "rsi_14d",
    ]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    date_list: list[pd.Timestamp] = []
    ticker_list: list[str] = []

    for current_month_end in month_end_dates:
        if current_month_end not in target_wide.index:
            continue

        eligible_daily_dates = daily_index[daily_index <= current_month_end]
        if len(eligible_daily_dates) < sequence_length:
            continue

        seq_dates = eligible_daily_dates[-sequence_length:]
        target_row = target_wide.loc[current_month_end]

        ret_slice = daily_returns.loc[seq_dates]
        vol20_slice = vol_20d.loc[seq_dates]
        vol60_slice = vol_60d.loc[seq_dates]
        ma20_slice = ma_ratio_20.loc[seq_dates]
        ma60_slice = ma_ratio_60.loc[seq_dates]
        drawdown_slice = drawdown_60d.loc[seq_dates]
        rsi_slice = rsi_14d.loc[seq_dates]
        market_slice = market_returns.loc[seq_dates]

        for ticker in common_tickers:
            stock_ret = ret_slice[ticker].to_numpy(dtype=float)
            market_ret = market_slice.to_numpy(dtype=float)
            excess_ret = stock_ret - market_ret

            seq_matrix = np.column_stack([
                stock_ret,
                excess_ret,
                market_ret,
                vol20_slice[ticker].to_numpy(dtype=float),
                vol60_slice[ticker].to_numpy(dtype=float),
                ma20_slice[ticker].to_numpy(dtype=float),
                ma60_slice[ticker].to_numpy(dtype=float),
                drawdown_slice[ticker].to_numpy(dtype=float),
                rsi_slice[ticker].to_numpy(dtype=float),
            ])

            target_value = target_row[ticker]

            if np.isnan(seq_matrix).any():
                continue
            if pd.isna(target_value):
                continue

            if normalize_per_sequence:
                seq_matrix = normalize_sequence_per_feature(seq_matrix)

            X_list.append(seq_matrix.astype(np.float32))
            y_list.append(float(target_value))
            date_list.append(pd.Timestamp(current_month_end))
            ticker_list.append(str(ticker))

    if not X_list:
        raise ValueError("No valid LSTM samples were created. Check sequence length and data coverage.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    return LSTMSampleSet(
        X=X,
        y=y,
        dates=pd.Index(date_list),
        tickers=pd.Index(ticker_list),
        target_name=target_name,
        feature_names=feature_names,
    )