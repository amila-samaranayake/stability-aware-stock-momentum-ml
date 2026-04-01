from __future__ import annotations

import os
import pandas as pd

from src import config
from src.features_daily import build_daily_feature_dataset
from src.preprocessing import (
    compute_returns,
    daily_to_monthly_compound,
    split_train_test_by_date,
    save_dataframe,
)
from src.utils.paths import get_feature_dataset_paths


FEATURE_PATHS = get_feature_dataset_paths("daily")


def main() -> None:
    """
    Build the daily-feature ML dataset and save the full, train, and test
    tables to the standard processed-data folders.
    """
    os.makedirs(FEATURE_PATHS["base_dir"], exist_ok=True)

    adj_close = pd.read_parquet(config.RAW_ADJ_CLOSE_PATH)
    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close = adj_close.sort_index()

    available_tickers = [ticker for ticker in config.TICKERS if ticker in adj_close.columns]
    adj_close = adj_close[available_tickers]

    print("Loaded adjusted close shape:", adj_close.shape)
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

    dataset = build_daily_feature_dataset(
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
        target_name="y_next_1m",
    )

    train_df, test_df = split_train_test_by_date(
        dataset,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE,
    )

    save_dataframe(dataset, FEATURE_PATHS["full"])
    save_dataframe(train_df, FEATURE_PATHS["train"])
    save_dataframe(test_df, FEATURE_PATHS["test"])

    target_cols = [c for c in train_df.columns if c.startswith("y_next")]
    feature_cols = [c for c in train_df.columns if c not in target_cols]

    print("\n=== Daily-feature ML dataset created successfully ===")
    print("Full dataset shape:", dataset.shape)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nSaved files:")
    print("Full ->", FEATURE_PATHS["full"])
    print("Train ->", FEATURE_PATHS["train"])
    print("Test ->", FEATURE_PATHS["test"])

    print("\nFeature columns:")
    print(feature_cols)

    print("\nTarget column:")
    print(target_cols)


if __name__ == "__main__":
    main()