from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with a sorted DatetimeIndex.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _sample_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample a daily dataframe at month-end.
    """
    return df.resample("ME").last()


def _stack_wide_to_long(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert a wide date x ticker dataframe into a long MultiIndex dataframe.

    Output index levels are (date, ticker).
    """
    out = df_wide.stack().to_frame(name=value_name)
    out.index.names = ["date", "ticker"]
    return out


def build_return_feature(
    daily_returns: pd.DataFrame,
    window: int,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """
    Build rolling return features over a trading-day window.
    """
    daily_returns = _ensure_datetime_index(daily_returns)

    if use_log_returns:
        return daily_returns.rolling(window=window, min_periods=window).sum()

    return (1.0 + daily_returns).rolling(window=window, min_periods=window).apply(
        lambda x: np.prod(x) - 1.0,
        raw=True,
    )


def build_volatility_feature(
    daily_returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Build rolling standard deviation of daily returns.
    """
    daily_returns = _ensure_datetime_index(daily_returns)
    return daily_returns.rolling(window=window, min_periods=window).std()


def build_moving_average_ratio(
    adj_close: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """
    Build moving-average ratio feature: MA_short / MA_long - 1.
    """
    adj_close = _ensure_datetime_index(adj_close)

    ma_short = adj_close.rolling(window=short_window, min_periods=short_window).mean()
    ma_long = adj_close.rolling(window=long_window, min_periods=long_window).mean()

    return (ma_short / ma_long) - 1.0


def build_distance_from_high(
    adj_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Build distance-from-high feature over a rolling window.
    """
    adj_close = _ensure_datetime_index(adj_close)
    rolling_high = adj_close.rolling(window=window, min_periods=window).max()
    return (adj_close / rolling_high) - 1.0


def build_drawdown_feature(
    adj_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Build rolling drawdown proxy over a rolling window.
    """
    return build_distance_from_high(adj_close, window=window)


def build_excess_return_feature(
    stock_return_feature: pd.DataFrame,
    market_return_feature: pd.Series,
) -> pd.DataFrame:
    """
    Build excess-return feature relative to a market benchmark.
    """
    aligned_market = market_return_feature.reindex(stock_return_feature.index)
    return stock_return_feature.sub(aligned_market, axis=0)


def build_beta_feature(
    daily_returns: pd.DataFrame,
    market_daily_returns: pd.Series,
    window: int,
) -> pd.DataFrame:
    """
    Build rolling beta of each stock relative to the market.
    """
    daily_returns = _ensure_datetime_index(daily_returns)

    market_daily_returns = pd.Series(market_daily_returns).copy()
    market_daily_returns.index = pd.to_datetime(market_daily_returns.index)
    market_daily_returns = market_daily_returns.sort_index().reindex(daily_returns.index)

    market_var = market_daily_returns.rolling(window=window, min_periods=window).var()
    betas = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns, dtype=float)

    for col in daily_returns.columns:
        cov_sm = daily_returns[col].rolling(window=window, min_periods=window).cov(market_daily_returns)
        betas[col] = cov_sm / market_var

    return betas


def build_rsi_feature(
    adj_close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Build RSI from daily adjusted close prices.
    """
    adj_close = _ensure_datetime_index(adj_close)

    delta = adj_close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_next_month_target(
    monthly_returns: pd.DataFrame,
    target_name: str = "y_next_1m",
) -> pd.DataFrame:
    """
    Build next-month return target.
    """
    target = monthly_returns.shift(-1)
    target.columns.name = "ticker"
    return target


def build_daily_feature_dataset(
    adj_close: pd.DataFrame,
    daily_returns: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    use_log_returns: bool = False,
    market_daily_returns: Optional[pd.Series] = None,
    return_windows: Optional[list[int]] = None,
    vol_windows: Optional[list[int]] = None,
    ma_pairs: Optional[list[tuple[int, int]]] = None,
    high_windows: Optional[list[int]] = None,
    drawdown_windows: Optional[list[int]] = None,
    beta_windows: Optional[list[int]] = None,
    rsi_window: int = 14,
    target_name: str = "y_next_1m",
) -> pd.DataFrame:
    """
    Build a month-end sampled ML dataset from daily-engineered features.

    Output format:
    - MultiIndex: (date, ticker)
    - Columns: feature columns + target column
    """
    adj_close = _ensure_datetime_index(adj_close)
    daily_returns = _ensure_datetime_index(daily_returns)
    monthly_returns = _ensure_datetime_index(monthly_returns)

    return_windows = return_windows or [5, 20, 60, 120, 252]
    vol_windows = vol_windows or [20, 60, 120]
    ma_pairs = ma_pairs or [(20, 60), (60, 252)]
    high_windows = high_windows or [252]
    drawdown_windows = drawdown_windows or [60]
    beta_windows = beta_windows or [60]

    feature_frames: dict[str, pd.DataFrame] = {}

    for window in return_windows:
        feature_frames[f"ret_{window}d"] = _sample_month_end(
            build_return_feature(daily_returns, window, use_log_returns=use_log_returns)
        )

    for window in vol_windows:
        feature_frames[f"vol_{window}d"] = _sample_month_end(
            build_volatility_feature(daily_returns, window)
        )

    for short_window, long_window in ma_pairs:
        feature_frames[f"ma_ratio_{short_window}_{long_window}"] = _sample_month_end(
            build_moving_average_ratio(adj_close, short_window, long_window)
        )

    for window in high_windows:
        feature_frames[f"dist_{window}d_high"] = _sample_month_end(
            build_distance_from_high(adj_close, window)
        )

    for window in drawdown_windows:
        feature_frames[f"drawdown_{window}d"] = _sample_month_end(
            build_drawdown_feature(adj_close, window)
        )

    feature_frames[f"rsi_{rsi_window}d"] = _sample_month_end(
        build_rsi_feature(adj_close, window=rsi_window)
    )

    if market_daily_returns is not None:
        market_daily_returns = pd.Series(market_daily_returns).copy()
        market_daily_returns.index = pd.to_datetime(market_daily_returns.index)
        market_daily_returns = market_daily_returns.sort_index()

        for window in [w for w in return_windows if w in [20, 60, 120, 252]]:
            market_return = _sample_month_end(
                build_return_feature(
                    market_daily_returns.to_frame("market"),
                    window,
                    use_log_returns=use_log_returns,
                )
            )["market"]

            stock_return = feature_frames[f"ret_{window}d"]
            feature_frames[f"excess_ret_{window}d"] = build_excess_return_feature(
                stock_return,
                market_return,
            )

        for window in beta_windows:
            feature_frames[f"beta_{window}d"] = _sample_month_end(
                build_beta_feature(daily_returns, market_daily_returns, window)
            )

    long_parts = [
        _stack_wide_to_long(feature_df, feature_name)
        for feature_name, feature_df in feature_frames.items()
    ]
    features_long = pd.concat(long_parts, axis=1)

    target_wide = build_next_month_target(monthly_returns, target_name=target_name)
    target_long = _stack_wide_to_long(target_wide, target_name)

    dataset = features_long.join(target_long, how="inner")
    return dataset.dropna().sort_index()