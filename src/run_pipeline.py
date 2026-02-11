# src/run_pipeline.py

from src.config import (
    TICKERS, START_DATE, END_DATE, TRAIN_END_DATE, TEST_START_DATE
)
from src.data_download import download_adj_close, save_dataframe, print_missing_summary
from src.preprocessing import preprocess_prices_to_returns, save_dataframe as save_df, basic_sanity_report

# Raw data (downloaded as is)
ADJ_CLOSE_PATH = "data/raw/adj_close_2015_2025.parquet"

# Processed data
RET_MONTHLY_PATH = "data/processed/returns_monthly.parquet"
TRAIN_PATH = "data/processed/train_monthly_2015_2024.parquet"
TEST_PATH = "data/processed/test_monthly_2025.parquet"


def main():
    # 1) Download (RAW)
    result = download_adj_close(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        auto_adjust=False,
    )
    save_dataframe(result.adj_close, ADJ_CLOSE_PATH)
    print(f"Saved RAW Adjusted Close -> {ADJ_CLOSE_PATH}")
    print_missing_summary(result.missing_ratio)

    # 2) Preprocess (PROCESSED)
    prep = preprocess_prices_to_returns(
        adj_close=result.adj_close,
        train_end_date=TRAIN_END_DATE,
        test_start_date=TEST_START_DATE,
        max_missing_ratio=0.10,
        fill_gap_limit=1,
    )

    save_df(prep.returns_monthly, RET_MONTHLY_PATH)
    save_df(prep.train_monthly, TRAIN_PATH)
    save_df(prep.test_monthly, TEST_PATH)

    print(f"Saved monthly returns -> {RET_MONTHLY_PATH}")
    print(f"Saved train set -> {TRAIN_PATH}")
    print(f"Saved test set -> {TEST_PATH}")

    basic_sanity_report(prep.returns_monthly)


if __name__ == "__main__":
    main()
