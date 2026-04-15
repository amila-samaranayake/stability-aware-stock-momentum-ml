from __future__ import annotations

import os
import pandas as pd

from src import config
from src.features_daily_ohlcv import build_daily_ohlcv_feature_dataset
from src.preprocessing import (
    compute_returns,
    daily_to_monthly_compound,
    split_train_test_by_date,
    save_dataframe,
)
from src.utils.paths import get_feature_dataset_paths


def main() -> None:
    paths = get_feature_dataset_paths("daily_ohlcv")
    os.makedirs(paths["base_dir"], exist_ok=True)

    ohlcv = pd.read_parquet(config.RAW_OHLCV_PATH)
    adj_close = pd.read_parquet(config.RAW_ADJ_CLOSE_PATH)

    ohlcv.index = pd.to_datetime(ohlcv.index)
    ohlcv = ohlcv.sort_index()

    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close = adj_close.sort_index()

    available_tickers = [t for t in config.TICKERS if t in adj_close.columns]
    adj_close = adj_close[available_tickers]

    # Restrict OHLCV multiindex columns to selected tickers
    if isinstance(ohlcv.columns, pd.MultiIndex):
        keep_cols = [col for col in ohlcv.columns if col[1] in available_tickers]
        ohlcv = ohlcv[keep_cols]

    print("Loaded adjusted close shape:", adj_close.shape)
    print("Loaded OHLCV shape:", ohlcv.shape)
    print("Date range:", adj_close.index.min(), "->", adj_close.index.max())
    print("Tickers used:", len(adj_close.columns))

    daily_returns = compute_returns(
        adj_close,
        use_log_returns=config.USE_LOG_RETURNS,
    )

    monthly_returns = daily_to_monthly_compound(
        daily_returns,
        use_log_returns=config.USE_LOG_RETURNS,
    )

    market_daily_returns = None
    if config.MARKET_TICKER is not None and config.MARKET_TICKER in daily_returns.columns:
        market_daily_returns = daily_returns[config.MARKET_TICKER]

    dataset = build_daily_ohlcv_feature_dataset(
        ohlcv=ohlcv,
        adj_close=adj_close,
        daily_returns=daily_returns,
        monthly_returns=monthly_returns,
        use_log_returns=config.USE_LOG_RETURNS,
        market_daily_returns=market_daily_returns,
        return_windows=config.DAILY_RETURN_WINDOWS,
        vol_windows=config.DAILY_VOL_WINDOWS,
        ma_pairs=config.DAILY_MA_PAIRS,
        high_windows=config.DAILY_HIGH_WINDOWS,
        drawdown_windows=config.DAILY_DRAWDOWN_WINDOWS,
        beta_windows=config.DAILY_BETA_WINDOWS,
        rsi_window=config.DAILY_RSI_WINDOW,
        volume_windows=getattr(config, "DAILY_OHLCV_VOL_WINDOWS", [20, 60]),
        abnormal_volume_windows=getattr(config, "DAILY_OHLCV_ABVOL_WINDOWS", [20]),
        range_windows=getattr(config, "DAILY_OHLCV_RANGE_WINDOWS", [5, 20]),
        clv_windows=getattr(config, "DAILY_OHLCV_CLV_WINDOWS", [5, 20]),
        target_name="y_next_1m",
    )

    train_df, test_df = split_train_test_by_date(
        dataset,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE,
    )

    save_dataframe(dataset, paths["full"])
    save_dataframe(train_df, paths["train"])
    save_dataframe(test_df, paths["test"])

    feature_cols = [c for c in dataset.columns if c != "y_next_1m"]

    print("\n=== Daily OHLCV-feature dataset built successfully")
    print("Full dataset shape:", dataset.shape)
    print("Train dataset shape:", train_df.shape)
    print("Test dataset shape:", test_df.shape)
    print("Number of feature columns:", len(feature_cols))
    print("Feature columns:")
    print(feature_cols)


if __name__ == "__main__":
    main()