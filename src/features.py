# src/features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Config container
# =========================

@dataclass
class FeaturesSpec:
    """
    Defines what features/targets to build.
    All windows are in MONTHS because we use monthly data.
    """
    lag_months: List[int] = None         # e.g., [1,3,6,12]
    vol_months: int = 3                  # rolling volatility window
    rsi_months: int = 14                 # RSI window on monthly prices
    target_horizon_months: int = 1       # next-month return target
    top_level_prefix_ret: str = "ret"
    top_level_prefix_vol: str = "vol"
    top_level_prefix_rsi: str = "rsi"
    target_name: str = "y_next"

def compute_monthly_prices_from_adj_close(adj_close_daily: pd.DataFrame, rule: str = "ME") -> pd.DataFrame:
    adj_close_daily = adj_close_daily.sort_index()
    return adj_close_daily.resample(rule).last()

def spec_from_config() -> FeaturesSpec:
    """
    Build FeaturesSpec from src/config.py.
    Keeps feature creation consistent with your project config.
    """
    from src import config

    # Lag windows: try to read from FEATURE_WINDOWS keys like lag_1m, lag_3m...
    lag_months = []
    vol_months = 3

    fw = getattr(config, "FEATURE_WINDOWS", {})
    if isinstance(fw, dict):
        for k, v in fw.items():
            if str(k).startswith("lag_"):
                lag_months.append(int(v))
            if str(k).startswith("vol_"):
                vol_months = int(v)

    # Fallback defaults if config not set fully
    if not lag_months:
        lag_months = [1, 3, 6, 12]

    # Optional config fields
    rsi_months = getattr(config, "RSI_WINDOW_MONTHS", 14)
    target_horizon = getattr(config, "TARGET_HORIZON_MONTHS", 1)

    return FeaturesSpec(
        lag_months=sorted(set(lag_months)),
        vol_months=int(vol_months),
        rsi_months=int(rsi_months),
        target_horizon_months=int(target_horizon),
    )


# =========================
# Feature builders
# =========================

def _compound_return_from_returns(window: np.ndarray) -> float:
    """Compounded return over a window: Π(1+r) - 1."""
    return float(np.prod(1.0 + window) - 1.0)


def build_lagged_returns(
    returns_monthly: pd.DataFrame,
    lag_months: List[int],
    prefix: str = "ret",
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """
    Build compounded lagged returns for each window using ONLY past data (shifted by 1 month).
    - If use_log_returns=False: compound simple returns over window: Π(1+r)-1
    - If use_log_returns=True: sum log returns over window: Σ r
    """
    returns_monthly = returns_monthly.sort_index()
    blocks = []

    for m in lag_months:
        if m <= 0:
            raise ValueError("lag_months must be positive integers.")

        if use_log_returns:
            lag = returns_monthly.rolling(m).sum()
        else:
            lag = returns_monthly.rolling(m).apply(_compound_return_from_returns, raw=True)
            
        lag = lag.shift(1)  # no look-ahead
        lag.columns = [f"{prefix}_{m}m__{c}" for c in lag.columns]
        blocks.append(lag)

    return pd.concat(blocks, axis=1)


def build_volatility(
    returns_monthly: pd.DataFrame,
    vol_months: int,
    prefix: str = "vol",
) -> pd.DataFrame:
    """
    Rolling volatility (std) of monthly returns, shifted by 1 month (no look-ahead).
    """
    if vol_months < 2:
        raise ValueError("vol_months should be >= 2.")

    returns_monthly = returns_monthly.sort_index()
    vol = returns_monthly.rolling(vol_months).std(ddof=1).shift(1)
    vol.columns = [f"{prefix}_{vol_months}m__{c}" for c in vol.columns]
    return vol


def build_rsi_from_monthly_prices(
    prices_monthly: pd.DataFrame,
    rsi_months: int = 14,
    prefix: str = "rsi",
) -> pd.DataFrame:
    """
    RSI computed on monthly prices using Wilder-style EWMA smoothing.
    Shifted by 1 month to avoid using the current month close for decisions.
    """
    if rsi_months < 2:
        raise ValueError("rsi_months should be >= 2.")

    prices_monthly = prices_monthly.sort_index()
    delta = prices_monthly.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / rsi_months
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.shift(1)  # <-- key: no look-ahead
    rsi.columns = [f"{prefix}_{rsi_months}m__{c}" for c in rsi.columns]
    return rsi


def build_target_next_return(
    returns_monthly: pd.DataFrame,
    horizon_months: int = 1,
    target_name: str = "y_next",
) -> pd.DataFrame:
    """
    Target: next-month return (shifted backward).
    y at time t is return at time t + horizon.
    """
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive.")

    y = returns_monthly.sort_index().shift(-horizon_months)
    y.columns = [f"{target_name}_{horizon_months}m__{c}" for c in y.columns]
    return y


# =========================
# Assemble ML dataset
# =========================

def wide_to_long(
    features_wide: pd.DataFrame,
    target_wide: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Correct conversion from wide to long:

    features_wide columns must be like: "ret_1m__AZN.L"
    target_wide columns must be like: "y_next_1m__AZN.L"

    Output:
      index = (date, ticker)
      columns = feature names + target
    """

    def to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        # Convert "feature__TICKER" columns into MultiIndex (feature, ticker)
        feats = []
        tickers = []
        for c in df.columns:
            feat, tkr = c.split("__", 1)
            feats.append(feat)
            tickers.append(tkr)
        df2 = df.copy()
        df2.columns = pd.MultiIndex.from_arrays([feats, tickers], names=["feature", "ticker"])
        return df2

    # 1) Convert to multiindex
    f_mi = to_multiindex(features_wide)
    y_mi = to_multiindex(target_wide)

    # 2) Stack ticker level -> rows become (date, ticker)
    f_long = f_mi.stack(level="ticker", future_stack=True)   # columns now are "feature"
    y_long = y_mi.stack(level="ticker", future_stack=True)   # columns now are "feature" (target name)

    f_long.index.names = ["date", "ticker"]
    y_long.index.names = ["date", "ticker"]

    # 3) Join features + target
    ml = f_long.join(y_long, how="inner")

    if drop_na:
        ml = ml.dropna(axis=0, how="any")

    # Ensure columns are plain strings
    ml.columns = [str(c) for c in ml.columns]

    return ml


def build_ml_dataset(
    returns_monthly: pd.DataFrame,
    prices_monthly: Optional[pd.DataFrame],
    spec: FeaturesSpec,
    include_rsi: bool = True,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """
    Build the ML dataset (Option A regression).
    returns_monthly: monthly returns (date index, tickers columns)
    prices_monthly: month-end prices (same tickers) for RSI (optional)
    """
    # Ensure aligned tickers
    tickers = returns_monthly.columns
    if prices_monthly is not None:
        common = tickers.intersection(prices_monthly.columns)
        returns_monthly = returns_monthly[common]
        prices_monthly = prices_monthly[common]

        common_idx = returns_monthly.index.intersection(prices_monthly.index)
        returns_monthly = returns_monthly.loc[common_idx]
        prices_monthly = prices_monthly.loc[common_idx]

    # Features
    lag_feats = build_lagged_returns(
        returns_monthly=returns_monthly,
        lag_months=spec.lag_months,
        prefix=spec.top_level_prefix_ret,
        use_log_returns=use_log_returns
    )
    vol_feat = build_volatility(returns_monthly, spec.vol_months, prefix=spec.top_level_prefix_vol)

    features = pd.concat([lag_feats, vol_feat], axis=1)

    if include_rsi and prices_monthly is not None:
        rsi_feat = build_rsi_from_monthly_prices(prices_monthly, spec.rsi_months, prefix=spec.top_level_prefix_rsi)
        features = pd.concat([features, rsi_feat], axis=1)

    # Target
    target = build_target_next_return(
        returns_monthly=returns_monthly,
        horizon_months=spec.target_horizon_months,
        target_name=spec.target_name,
    )

    # Long table
    ml = wide_to_long(features_wide=features, target_wide=target, drop_na=True)
    return ml


def split_by_date(
    ml_dataset: pd.DataFrame,
    train_end_date: str,
    test_start_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ML dataset by date boundaries.
    """
    dates = ml_dataset.index.get_level_values("date")
    train = ml_dataset.loc[dates <= pd.to_datetime(train_end_date)].copy()
    test = ml_dataset.loc[dates >= pd.to_datetime(test_start_date)].copy()
    return train, test