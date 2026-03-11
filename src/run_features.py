# src/run_features.py

import os
import pandas as pd

from src.config import TRAIN_END_DATE, TEST_START_DATE
from src.features import (
    spec_from_config,
    compute_monthly_prices_from_adj_close,   # <-- we need this helper; see note below
    build_ml_dataset,
    split_by_date,
)

RET_MONTHLY_PATH = "data/processed/returns_monthly.parquet"
RAW_ADJ_CLOSE_PATH = "data/raw/adj_close_2015_2025.parquet"

OUT_TRAIN_PATH = "data/processed/ml_train_2015_2024.parquet"
OUT_TEST_PATH = "data/processed/ml_test_2025.parquet"


def main():
    # 1) Load monthly returns
    returns_monthly = pd.read_parquet(RET_MONTHLY_PATH)

    # 2) Load daily prices and convert to monthly prices (for RSI)
    adj_close_daily = pd.read_parquet(RAW_ADJ_CLOSE_PATH)
    prices_monthly = compute_monthly_prices_from_adj_close(adj_close_daily)

    # 3) Build feature specification from config
    spec = spec_from_config()

    # 4) Build ML dataset (long format: index=(date,ticker))
    ml_dataset = build_ml_dataset(
        returns_monthly=returns_monthly,
        prices_monthly=prices_monthly,
        spec=spec,
        include_rsi=True,  # uses config window
    )

    # 5) Split train/test by date
    train_df, test_df = split_by_date(
        ml_dataset=ml_dataset,
        train_end_date=TRAIN_END_DATE,
        test_start_date=TEST_START_DATE,
    )

    # 6) Save
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_parquet(OUT_TRAIN_PATH)
    test_df.to_parquet(OUT_TEST_PATH)

    # 7) Report
    print("=== ML dataset created successfully")
    print("Full ML dataset shape:", ml_dataset.shape)
    print("Train shape (<= 2024):", train_df.shape)
    print("Test shape (2025):", test_df.shape)

    print("\nFeature columns:", [c for c in train_df.columns if c.startswith(("ret_", "vol_", "rsi_"))])
    print("Target column:", [c for c in train_df.columns if c.startswith("y_next")])

    print("\nSample rows:")
    print(train_df.head(5))


if __name__ == "__main__":
    main()