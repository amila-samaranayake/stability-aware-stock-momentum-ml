from __future__ import annotations

import pandas as pd

from src import config


def main() -> None:
    df = pd.read_parquet(config.RAW_OHLCV_PATH)

    print("Shape:", df.shape)
    print("\nColumn type:", type(df.columns))
    print("\nFirst 10 columns:")
    print(df.columns[:10])

    print("\nHead:")
    print(df.head())


if __name__ == "__main__":
    main()