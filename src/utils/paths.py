# src/utils/paths.py

from pathlib import Path


def get_processed_returns_paths(output_root: str = "data/processed") -> dict:
    base = Path(output_root) / "returns"
    return {
        "base_dir": str(base),
        "daily": str(base / "returns_daily.parquet"),
        "monthly": str(base / "returns_monthly.parquet"),
        "train_monthly": str(base / "train_monthly_2015_2024.parquet"),
        "test_monthly": str(base / "test_monthly_2025.parquet"),
    }


def get_feature_dataset_paths(
    feature_source: str,
    output_root: str = "data/processed",
) -> dict:
    """
    feature_source: 'monthly' or 'daily' or 'daily_ohlcv'
    """
    if feature_source not in {"monthly", "daily", "daily_ohlcv"}:
        raise ValueError(f"Unsupported feature_source: {feature_source}")

    base = Path(output_root) / f"features_{feature_source}"
    return {
        "base_dir": str(base),
        "full": str(base / f"ml_full_{feature_source}.parquet"),
        "train": str(base / f"ml_train_{feature_source}_2015_2024.parquet"),
        "test": str(base / f"ml_test_{feature_source}_2025.parquet"),
    }


def get_experiment_dir(
    experiment_name: str,
    feature_source: str,
    output_root: str = "experiments/results",
) -> str:
    """
    Example:
    experiment_name='exp04_random_forest'
    feature_source='daily'
    -> experiments/results/exp04_random_forest_daily
    """
    return str(Path(output_root) / f"{experiment_name}_{feature_source}")