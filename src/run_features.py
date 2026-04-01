# src/run_features.py

import os
import pandas as pd

from src import config
from src.features import (
    spec_from_config,
    compute_monthly_prices_from_adj_close,
    build_ml_dataset,
    split_by_date,
)
from src.utils.paths import get_feature_dataset_paths, get_processed_returns_paths


FEATURE_PATHS = get_feature_dataset_paths("monthly")
RETURNS_PATHS = get_processed_returns_paths()


def main() -> None:
    """
    Build the monthly-feature ML dataset and save the full, train, and test
    tables to the standard processed-data folders.
    """
    os.makedirs(FEATURE_PATHS["base_dir"], exist_ok=True)

    returns_monthly = pd.read_parquet(RETURNS_PATHS["monthly"])

    adj_close_daily = pd.read_parquet(config.RAW_ADJ_CLOSE_PATH)
    prices_monthly = compute_monthly_prices_from_adj_close(adj_close_daily)

    spec = spec_from_config()

    ml_dataset = build_ml_dataset(
        returns_monthly=returns_monthly,
        prices_monthly=prices_monthly,
        spec=spec,
        include_rsi=True,
        use_log_returns=config.USE_LOG_RETURNS,
    )

    train_df, test_df = split_by_date(
        ml_dataset=ml_dataset,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE,
    )

    ml_dataset.to_parquet(FEATURE_PATHS["full"])
    train_df.to_parquet(FEATURE_PATHS["train"])
    test_df.to_parquet(FEATURE_PATHS["test"])

    target_cols = [c for c in train_df.columns if c.startswith("y_next")]
    feature_cols = [c for c in train_df.columns if c not in target_cols]

    print("=== Monthly-feature ML dataset created successfully ===")
    print("Full ML dataset shape:", ml_dataset.shape)
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

    print("\nSample rows:")
    print(train_df.head(5))


if __name__ == "__main__":
    main()