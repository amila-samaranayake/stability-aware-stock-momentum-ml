# src/run_pipeline.py

from src import config
from src.data_download import download_adj_close, save_dataframe as save_raw_dataframe, print_missing_summary
from src.preprocessing import (
    preprocess_prices_to_returns,
    save_dataframe,
    basic_sanity_report,
)
from src.utils.paths import get_processed_returns_paths


ADJ_CLOSE_PATH = config.RAW_ADJ_CLOSE_PATH
RETURNS_PATHS = get_processed_returns_paths()


def main() -> None:
    """
    Run the raw-data download and return preprocessing pipeline.

    This pipeline:
    1. Downloads adjusted close prices
    2. Computes daily and monthly returns
    3. Splits monthly returns into train and test sets
    4. Saves processed outputs to standard paths
    """
    result = download_adj_close(
        tickers=config.TICKERS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        auto_adjust=False,
    )

    save_raw_dataframe(result.adj_close, ADJ_CLOSE_PATH)
    print(f"Saved RAW adjusted close -> {ADJ_CLOSE_PATH}")
    print_missing_summary(result.missing_ratio)

    prep = preprocess_prices_to_returns(
        adj_close=result.adj_close,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE,
        max_missing_ratio=0.10,
        fill_gap_limit=1,
        use_log_returns=config.USE_LOG_RETURNS,
    )

    save_dataframe(prep.returns_daily, RETURNS_PATHS["daily"])
    save_dataframe(prep.returns_monthly, RETURNS_PATHS["monthly"])
    save_dataframe(prep.train_monthly, RETURNS_PATHS["train_monthly"])
    save_dataframe(prep.test_monthly, RETURNS_PATHS["test_monthly"])

    print(f"Saved daily returns -> {RETURNS_PATHS['daily']}")
    print(f"Saved monthly returns -> {RETURNS_PATHS['monthly']}")
    print(f"Saved train monthly set -> {RETURNS_PATHS['train_monthly']}")
    print(f"Saved test monthly set -> {RETURNS_PATHS['test_monthly']}")

    basic_sanity_report(prep.returns_monthly)


if __name__ == "__main__":
    main()