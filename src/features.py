# src/features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeaturesSpec:
    """
    Configuration for monthly feature engineering.

    All windows are expressed in months because this pipeline uses
    month-end prices and monthly returns.
    """
    lag_months: List[int]
    vol_months: int = 3
    rsi_months: int = 14
    target_horizon_months: int = 1
    top_level_prefix_ret: str = "ret"
    top_level_prefix_vol: str = "vol"
    top_level_prefix_rsi: str = "rsi"
    target_name: str = "y_next"


def compute_monthly_prices_from_adj_close(
    adj_close_daily: pd.DataFrame,
    rule: str = "ME",
) -> pd.DataFrame:
    """
    Convert daily adjusted close prices to month-end prices.
    """
    adj_close_daily = adj_close_daily.sort_index()
    return adj_close_daily.resample(rule).last()


def spec_from_config() -> FeaturesSpec:
    """
    Build a FeaturesSpec from src.config so monthly feature creation stays
    aligned with the project configuration.
    """
    from src import config

    lag_months: list[int] = []
    vol_months = 3

    feature_windows = getattr(config, "FEATURE_WINDOWS", {})
    if isinstance(feature_windows, dict):
        for key, value in feature_windows.items():
            if str(key).startswith("lag_"):
                lag_months.append(int(value))
            elif str(key).startswith("vol_"):
                vol_months = int(value)

    if not lag_months:
        lag_months = [1, 3, 6, 12]

    rsi_months = getattr(config, "RSI_WINDOW_MONTHS", 14)
    target_horizon = getattr(config, "TARGET_HORIZON_MONTHS", 1)

    return FeaturesSpec(
        lag_months=sorted(set(lag_months)),
        vol_months=int(vol_months),
        rsi_months=int(rsi_months),
        target_horizon_months=int(target_horizon),
    )


def _compound_return_from_returns(window: np.ndarray) -> float:
    """
    Compute compounded simple return over a rolling window.
    """
    return float(np.prod(1.0 + window) - 1.0)


def build_lagged_returns(
    returns_monthly: pd.DataFrame,
    lag_months: List[int],
    prefix: str = "ret",
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """
    Build lagged return features using only past information.

    If use_log_returns is False, rolling simple returns are compounded.
    If use_log_returns is True, rolling log returns are summed.
    """
    returns_monthly = returns_monthly.sort_index()
    blocks = []

    for months in lag_months:
        if months <= 0:
            raise ValueError("lag_months must contain positive integers.")

        if use_log_returns:
            lag = returns_monthly.rolling(months).sum()
        else:
            lag = returns_monthly.rolling(months).apply(
                _compound_return_from_returns,
                raw=True,
            )

        lag = lag.shift(1)
        lag.columns = [f"{prefix}_{months}m__{col}" for col in lag.columns]
        blocks.append(lag)

    return pd.concat(blocks, axis=1)


def build_volatility(
    returns_monthly: pd.DataFrame,
    vol_months: int,
    prefix: str = "vol",
) -> pd.DataFrame:
    """
    Build rolling monthly volatility features using past data only.
    """
    if vol_months < 2:
        raise ValueError("vol_months must be at least 2.")

    returns_monthly = returns_monthly.sort_index()
    vol = returns_monthly.rolling(vol_months).std(ddof=1).shift(1)
    vol.columns = [f"{prefix}_{vol_months}m__{col}" for col in vol.columns]
    return vol


def build_rsi_from_monthly_prices(
    prices_monthly: pd.DataFrame,
    rsi_months: int = 14,
    prefix: str = "rsi",
) -> pd.DataFrame:
    """
    Build RSI features from monthly prices using Wilder-style smoothing.

    The final feature is shifted by one month to avoid look-ahead bias.
    """
    if rsi_months < 2:
        raise ValueError("rsi_months must be at least 2.")

    prices_monthly = prices_monthly.sort_index()
    delta = prices_monthly.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / rsi_months
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.shift(1)
    rsi.columns = [f"{prefix}_{rsi_months}m__{col}" for col in rsi.columns]
    return rsi


def build_target_next_return(
    returns_monthly: pd.DataFrame,
    horizon_months: int = 1,
    target_name: str = "y_next",
) -> pd.DataFrame:
    """
    Build next-period return target.

    At time t, the target corresponds to the realized return at t + horizon.
    """
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive.")

    target = returns_monthly.sort_index().shift(-horizon_months)
    target.columns = [f"{target_name}_{horizon_months}m__{col}" for col in target.columns]
    return target


def wide_to_long(
    features_wide: pd.DataFrame,
    target_wide: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Convert wide feature and target matrices to a long ML dataset.

    Expected column format:
    - feature columns: 'feature_name__TICKER'
    - target columns: 'target_name__TICKER'

    Output:
    - MultiIndex index: (date, ticker)
    - Columns: feature names + target
    """

    def to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        feature_names = []
        tickers = []

        for col in df.columns:
            feature_name, ticker = col.split("__", 1)
            feature_names.append(feature_name)
            tickers.append(ticker)

        out = df.copy()
        out.columns = pd.MultiIndex.from_arrays(
            [feature_names, tickers],
            names=["feature", "ticker"],
        )
        return out

    features_mi = to_multiindex(features_wide)
    target_mi = to_multiindex(target_wide)

    features_long = features_mi.stack(level="ticker", future_stack=True)
    target_long = target_mi.stack(level="ticker", future_stack=True)

    features_long.index.names = ["date", "ticker"]
    target_long.index.names = ["date", "ticker"]

    ml_dataset = features_long.join(target_long, how="inner")

    if drop_na:
        ml_dataset = ml_dataset.dropna(axis=0, how="any")

    ml_dataset.columns = [str(col) for col in ml_dataset.columns]
    return ml_dataset


def build_ml_dataset(
    returns_monthly: pd.DataFrame,
    prices_monthly: Optional[pd.DataFrame],
    spec: FeaturesSpec,
    include_rsi: bool = True,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """
    Build the monthly-feature ML dataset in long format.
    """
    tickers = returns_monthly.columns

    if prices_monthly is not None:
        common_tickers = tickers.intersection(prices_monthly.columns)
        returns_monthly = returns_monthly[common_tickers]
        prices_monthly = prices_monthly[common_tickers]

        common_index = returns_monthly.index.intersection(prices_monthly.index)
        returns_monthly = returns_monthly.loc[common_index]
        prices_monthly = prices_monthly.loc[common_index]

    lag_features = build_lagged_returns(
        returns_monthly=returns_monthly,
        lag_months=spec.lag_months,
        prefix=spec.top_level_prefix_ret,
        use_log_returns=use_log_returns,
    )

    vol_features = build_volatility(
        returns_monthly=returns_monthly,
        vol_months=spec.vol_months,
        prefix=spec.top_level_prefix_vol,
    )

    features = pd.concat([lag_features, vol_features], axis=1)

    if include_rsi and prices_monthly is not None:
        rsi_features = build_rsi_from_monthly_prices(
            prices_monthly=prices_monthly,
            rsi_months=spec.rsi_months,
            prefix=spec.top_level_prefix_rsi,
        )
        features = pd.concat([features, rsi_features], axis=1)

    target = build_target_next_return(
        returns_monthly=returns_monthly,
        horizon_months=spec.target_horizon_months,
        target_name=spec.target_name,
    )

    return wide_to_long(
        features_wide=features,
        target_wide=target,
        drop_na=True,
    )


def split_by_date(
    ml_dataset: pd.DataFrame,
    train_end_date: str,
    test_start_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a long ML dataset by date boundaries.
    """
    dates = ml_dataset.index.get_level_values("date")
    train = ml_dataset.loc[dates <= pd.to_datetime(train_end_date)].copy()
    test = ml_dataset.loc[dates >= pd.to_datetime(test_start_date)].copy()
    return train, test