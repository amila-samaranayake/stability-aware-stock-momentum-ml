from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _sample_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()


def _stack_wide_to_long(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    out = df_wide.stack().to_frame(name=value_name)
    out.index.names = ["date", "ticker"]
    return out


def _get_ohlcv_field(ohlcv: pd.DataFrame, field: str) -> pd.DataFrame:
    if not isinstance(ohlcv.columns, pd.MultiIndex):
        raise ValueError("OHLCV dataframe must have MultiIndex columns.")
    if field not in ohlcv.columns.get_level_values(0):
        raise ValueError(f"Field '{field}' not found in OHLCV data.")
    out = ohlcv[field].copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def build_return_feature(
    daily_returns: pd.DataFrame,
    window: int,
    use_log_returns: bool = False,
) -> pd.DataFrame:
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
    daily_returns = _ensure_datetime_index(daily_returns)
    return daily_returns.rolling(window=window, min_periods=window).std()


def build_moving_average_ratio(
    adj_close: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    adj_close = _ensure_datetime_index(adj_close)
    ma_short = adj_close.rolling(window=short_window, min_periods=short_window).mean()
    ma_long = adj_close.rolling(window=long_window, min_periods=long_window).mean()
    return (ma_short / ma_long) - 1.0


def build_distance_from_high(
    adj_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    adj_close = _ensure_datetime_index(adj_close)
    rolling_high = adj_close.rolling(window=window, min_periods=window).max()
    return (adj_close / rolling_high) - 1.0


def build_drawdown_feature(
    adj_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    return build_distance_from_high(adj_close, window=window)


def build_excess_return_feature(
    stock_return_feature: pd.DataFrame,
    market_return_feature: pd.Series,
) -> pd.DataFrame:
    aligned_market = market_return_feature.reindex(stock_return_feature.index)
    return stock_return_feature.sub(aligned_market, axis=0)


def build_beta_feature(
    daily_returns: pd.DataFrame,
    market_daily_returns: pd.Series,
    window: int,
) -> pd.DataFrame:
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
    adj_close = _ensure_datetime_index(adj_close)
    delta = adj_close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_volume_feature(
    volume: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    volume = _ensure_datetime_index(volume)
    return volume.rolling(window=window, min_periods=window).mean()


def build_abnormal_volume_feature(
    volume: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    volume = _ensure_datetime_index(volume)
    rolling_mean = volume.rolling(window=window, min_periods=window).mean()
    return (volume / rolling_mean) - 1.0


def build_intraday_range_feature(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    high = _ensure_datetime_index(high)
    low = _ensure_datetime_index(low)
    close = _ensure_datetime_index(close)
    daily_range = (high - low) / close.replace(0.0, np.nan)
    return daily_range.rolling(window=window, min_periods=window).mean()


def build_close_location_value(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    high = _ensure_datetime_index(high)
    low = _ensure_datetime_index(low)
    close = _ensure_datetime_index(close)
    denom = (high - low).replace(0.0, np.nan)
    return ((close - low) / denom) - 0.5


def build_clv_rolling_feature(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    clv = build_close_location_value(high, low, close)
    return clv.rolling(window=window, min_periods=window).mean()


def build_open_close_return(
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    open_px = _ensure_datetime_index(open_px)
    close_px = _ensure_datetime_index(close_px)
    if use_log_returns:
        return np.log(close_px / open_px)
    return (close_px / open_px) - 1.0


def build_next_month_target(
    monthly_returns: pd.DataFrame,
    target_name: str = "y_next_1m",
) -> pd.DataFrame:
    target = monthly_returns.shift(-1)
    target.columns.name = "ticker"
    return target


def build_daily_ohlcv_feature_dataset(
    ohlcv: pd.DataFrame,
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
    volume_windows: Optional[list[int]] = None,
    abnormal_volume_windows: Optional[list[int]] = None,
    range_windows: Optional[list[int]] = None,
    clv_windows: Optional[list[int]] = None,
    target_name: str = "y_next_1m",
) -> pd.DataFrame:
    adj_close = _ensure_datetime_index(adj_close)
    daily_returns = _ensure_datetime_index(daily_returns)
    monthly_returns = _ensure_datetime_index(monthly_returns)

    open_px = _get_ohlcv_field(ohlcv, "Open")
    high_px = _get_ohlcv_field(ohlcv, "High")
    low_px = _get_ohlcv_field(ohlcv, "Low")
    close_px = _get_ohlcv_field(ohlcv, "Close")
    volume = _get_ohlcv_field(ohlcv, "Volume")

    return_windows = return_windows or [5, 20, 60, 120, 252]
    vol_windows = vol_windows or [20, 60, 120]
    ma_pairs = ma_pairs or [(20, 60), (60, 252)]
    high_windows = high_windows or [252]
    drawdown_windows = drawdown_windows or [60]
    beta_windows = beta_windows or [60]
    volume_windows = volume_windows or [20, 60]
    abnormal_volume_windows = abnormal_volume_windows or [20]
    range_windows = range_windows or [5, 20]
    clv_windows = clv_windows or [5, 20]

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

    # Keep the spread features from your current best daily setup
    if 5 in return_windows and 20 in return_windows:
        feature_frames["ret_spread_5d_20d"] = (
            feature_frames["ret_5d"] - feature_frames["ret_20d"]
        )
    if 20 in return_windows and 60 in return_windows:
        feature_frames["ret_spread_20d_60d"] = (
            feature_frames["ret_20d"] - feature_frames["ret_60d"]
        )

    # OHLCV-only features
    for window in volume_windows:
        feature_frames[f"volavg_{window}d"] = _sample_month_end(
            build_volume_feature(volume, window)
        )

    for window in abnormal_volume_windows:
        feature_frames[f"abvol_{window}d"] = _sample_month_end(
            build_abnormal_volume_feature(volume, window)
        )

    for window in range_windows:
        feature_frames[f"range_{window}d"] = _sample_month_end(
            build_intraday_range_feature(high_px, low_px, close_px, window)
        )

    for window in clv_windows:
        feature_frames[f"clv_{window}d"] = _sample_month_end(
            build_clv_rolling_feature(high_px, low_px, close_px, window)
        )

    feature_frames["open_close_ret"] = _sample_month_end(
        build_open_close_return(open_px, close_px, use_log_returns=use_log_returns)
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