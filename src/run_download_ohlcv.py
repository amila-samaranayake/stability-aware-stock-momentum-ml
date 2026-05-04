from __future__ import annotations

from src import config
from src.data_download import (
    download_ohlcv,
    print_missing_summary,
    save_dataframe,
)


def main() -> None:
    """
    Download full OHLCV raw data and adjusted close, then save both.
    """
    result = download_ohlcv(
        tickers=config.TICKERS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        auto_adjust=False,
    )

    save_dataframe(result.ohlcv, config.RAW_OHLCV_PATH)
    print(f"Saved RAW OHLCV -> {config.RAW_OHLCV_PATH}")

    save_dataframe(result.adj_close, config.RAW_ADJ_CLOSE_PATH)
    print(f"Saved RAW Adjusted Close -> {config.RAW_ADJ_CLOSE_PATH}")

    print_missing_summary(result.missing_ratio_adj_close)


if __name__ == "__main__":
    main()