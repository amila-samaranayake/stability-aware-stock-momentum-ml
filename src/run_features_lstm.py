# src/run_features_lstm.py

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src import config
from src.features_lstm import (
    build_lstm_multifeature_sequence_dataset,
    split_lstm_dataset_by_date,
    save_lstm_sample_set,
    lstm_sample_set_to_long_dataframe,
)


LSTM_FEATURE_DIR = Path("data/processed/features_lstm")

FULL_NPZ_PATH = LSTM_FEATURE_DIR / "lstm_full_daily.npz"
TRAIN_NPZ_PATH = LSTM_FEATURE_DIR / "lstm_train_daily_2015_2024.npz"
TEST_NPZ_PATH = LSTM_FEATURE_DIR / "lstm_test_daily_2025.npz"

FULL_META_PATH = LSTM_FEATURE_DIR / "lstm_full_daily_metadata.parquet"
TRAIN_META_PATH = LSTM_FEATURE_DIR / "lstm_train_daily_2015_2024_metadata.parquet"
TEST_META_PATH = LSTM_FEATURE_DIR / "lstm_test_daily_2025_metadata.parquet"


def main() -> None:
    """
    Build and save the LSTM sequence dataset without changing the existing
    tabular monthly/daily feature pipelines.
    """
    os.makedirs(LSTM_FEATURE_DIR, exist_ok=True)

    adj_close = pd.read_parquet(config.RAW_ADJ_CLOSE_PATH)
    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close = adj_close.sort_index()

    available_tickers = [ticker for ticker in config.TICKERS if ticker in adj_close.columns]
    adj_close = adj_close[available_tickers]

    print("Loaded adjusted close shape:", adj_close.shape)
    print("Date range:", adj_close.index.min(), "->", adj_close.index.max())
    print("Tickers used:", len(adj_close.columns))

    sequence_length = getattr(config, "LSTM_SEQUENCE_LENGTH", 60)
    normalize_per_sequence = getattr(config, "LSTM_NORMALIZE_PER_SEQUENCE", True)

    dataset = build_lstm_multifeature_sequence_dataset(
        adj_close=adj_close,
        market_ticker=getattr(config, "LSTM_MARKET_TICKER", None),
        sequence_length=sequence_length,
        target_horizon_months=1,
        target_name="y_next_1m",
        normalize_per_sequence=normalize_per_sequence,
    )

    train_set, test_set = split_lstm_dataset_by_date(
        dataset=dataset,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE,
    )

    save_lstm_sample_set(dataset, str(FULL_NPZ_PATH))
    save_lstm_sample_set(train_set, str(TRAIN_NPZ_PATH))
    save_lstm_sample_set(test_set, str(TEST_NPZ_PATH))

    full_meta = lstm_sample_set_to_long_dataframe(dataset)
    train_meta = lstm_sample_set_to_long_dataframe(train_set)
    test_meta = lstm_sample_set_to_long_dataframe(test_set)

    full_meta.to_parquet(FULL_META_PATH)
    train_meta.to_parquet(TRAIN_META_PATH)
    test_meta.to_parquet(TEST_META_PATH)

    print("\n=== LSTM sequence dataset created successfully ===")
    print("Sequence length:", sequence_length)
    print("Normalize per sequence:", normalize_per_sequence)
    print("Feature names:", dataset.feature_names)
    print("Full X shape:", dataset.X.shape)
    print("Train X shape:", train_set.X.shape)
    print("Test X shape:", test_set.X.shape)

    print("\nSaved files:")
    print("Full NPZ  ->", FULL_NPZ_PATH)
    print("Train NPZ ->", TRAIN_NPZ_PATH)
    print("Test NPZ  ->", TEST_NPZ_PATH)
    print("Full meta ->", FULL_META_PATH)
    print("Train meta->", TRAIN_META_PATH)
    print("Test meta ->", TEST_META_PATH)

    print("\nTarget column:", dataset.target_name)
    print("Unique train months:", train_meta.index.get_level_values("date").nunique())
    print("Unique test months:", test_meta.index.get_level_values("date").nunique())


if __name__ == "__main__":
    main()