# src/analysis/prediction_diagnostics.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions(predictions_path: str, target_col: str = "y_next_1m") -> pd.DataFrame:
    """
    Load prediction file and return a long dataframe indexed by (date, ticker).

    Expected columns:
    - date
    - ticker
    - pred_return
    - target_col
    """
    path = Path(predictions_path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if {"date", "ticker"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "ticker"]).sort_index()

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Predictions file must contain MultiIndex (date, ticker) or date/ticker columns.")

    required_cols = {target_col, "pred_return"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[[target_col, "pred_return"]].copy()


def build_monthly_rank_table(
    df_pred: pd.DataFrame,
    target_col: str = "y_next_1m",
    top_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Build month-by-month ranking diagnostics.
    """
    out_parts = []

    for current_date, group in df_pred.groupby(level="date"):
        g = group.copy().reset_index()
        g = g.sort_values("pred_return", ascending=False).reset_index(drop=True)

        n_assets = len(g)
        k = max(1, int(np.ceil(n_assets * top_pct)))

        g["pred_rank"] = g["pred_return"].rank(method="first", ascending=False).astype(int)
        g["actual_rank"] = g[target_col].rank(method="first", ascending=False).astype(int)
        g["rank_error"] = (g["pred_rank"] - g["actual_rank"]).abs()

        g["pred_topk"] = g["pred_rank"] <= k
        g["actual_topk"] = g["actual_rank"] <= k
        g["topk_hit"] = g["pred_topk"] & g["actual_topk"]

        g["pred_minus_actual"] = g["pred_return"] - g[target_col]
        g["date"] = pd.to_datetime(current_date)

        out_parts.append(g)

    monthly_rank_df = pd.concat(out_parts, axis=0, ignore_index=True)
    return monthly_rank_df.sort_values(["date", "pred_rank"])


def summarize_by_stock(
    monthly_rank_df: pd.DataFrame,
    target_col: str = "y_next_1m",
) -> pd.DataFrame:
    """
    Summarize prediction performance for each stock across all months.
    """
    rows = []

    for ticker, group in monthly_rank_df.groupby("ticker"):
        y_true = group[target_col].to_numpy(dtype=float)
        y_pred = group["pred_return"].to_numpy(dtype=float)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

        if len(group) >= 2:
            corr = float(pd.Series(y_pred).corr(pd.Series(y_true)))
        else:
            corr = np.nan

        rows.append(
            {
                "ticker": ticker,
                "n_months": int(len(group)),
                "mean_pred_return": float(np.mean(y_pred)),
                "mean_actual_return": float(np.mean(y_true)),
                "mae": mae,
                "rmse": rmse,
                "directional_accuracy": dir_acc,
                "pred_actual_corr": corr,
                "avg_pred_rank": float(group["pred_rank"].mean()),
                "avg_actual_rank": float(group["actual_rank"].mean()),
                "avg_rank_error": float(group["rank_error"].mean()),
                "topk_hits": int(group["topk_hit"].sum()),
                "pred_topk_count": int(group["pred_topk"].sum()),
                "actual_topk_count": int(group["actual_topk"].sum()),
            }
        )

    stock_df = pd.DataFrame(rows).sort_values(
        ["avg_rank_error", "rmse", "ticker"], ascending=[True, True, True]
    )
    return stock_df.reset_index(drop=True)


def summarize_by_month(
    monthly_rank_df: pd.DataFrame,
    target_col: str = "y_next_1m",
) -> pd.DataFrame:
    """
    Summarize diagnostics month by month.
    """
    rows = []

    for current_date, group in monthly_rank_df.groupby("date"):
        y_true = group[target_col].to_numpy(dtype=float)
        y_pred = group["pred_return"].to_numpy(dtype=float)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        spearman = float(
            group[["pred_return", target_col]].corr(method="spearman").iloc[0, 1]
        )

        rows.append(
            {
                "date": pd.to_datetime(current_date),
                "n_stocks": int(len(group)),
                "mae": mae,
                "rmse": rmse,
                "directional_accuracy": dir_acc,
                "spearman": spearman,
                "topk_hit_rate": float(group["topk_hit"].mean() / max(group["pred_topk"].mean(), 1e-12))
                if group["pred_topk"].sum() > 0
                else np.nan,
                "avg_rank_error": float(group["rank_error"].mean()),
                "mean_pred_return": float(np.mean(y_pred)),
                "mean_actual_return": float(np.mean(y_true)),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_top_mistakes_tables(
    monthly_rank_df: pd.DataFrame,
    n_rows: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the largest overpredictions and underpredictions.
    """
    over = monthly_rank_df.sort_values("pred_minus_actual", ascending=False).head(n_rows).copy()
    under = monthly_rank_df.sort_values("pred_minus_actual", ascending=True).head(n_rows).copy()
    return over, under


def main() -> None:
    parser = argparse.ArgumentParser(description="Create prediction diagnostics tables.")
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Path to predictions file (csv or parquet) with date, ticker, y_next_1m, pred_return.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where diagnostics outputs will be saved.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="y_next_1m",
        help="Target column name.",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.20,
        help="Top fraction used for top-k diagnostics.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_pred = load_predictions(
        predictions_path=args.predictions_path,
        target_col=args.target_col,
    )

    monthly_rank_df = build_monthly_rank_table(
        df_pred=df_pred,
        target_col=args.target_col,
        top_pct=args.top_pct,
    )
    stock_summary_df = summarize_by_stock(
        monthly_rank_df=monthly_rank_df,
        target_col=args.target_col,
    )
    month_summary_df = summarize_by_month(
        monthly_rank_df=monthly_rank_df,
        target_col=args.target_col,
    )
    over_df, under_df = build_top_mistakes_tables(monthly_rank_df)

    monthly_rank_path = output_dir / "monthly_rank_table.csv"
    stock_summary_path = output_dir / "stock_summary.csv"
    month_summary_path = output_dir / "month_summary.csv"
    over_path = output_dir / "largest_overpredictions.csv"
    under_path = output_dir / "largest_underpredictions.csv"

    monthly_rank_df.to_csv(monthly_rank_path, index=False)
    stock_summary_df.to_csv(stock_summary_path, index=False)
    month_summary_df.to_csv(month_summary_path, index=False)
    over_df.to_csv(over_path, index=False)
    under_df.to_csv(under_path, index=False)

    print("Saved diagnostics:")
    print("Monthly rank table     ->", monthly_rank_path)
    print("Per-stock summary      ->", stock_summary_path)
    print("Per-month summary      ->", month_summary_path)
    print("Largest overpredictions->", over_path)
    print("Largest underpredictions->", under_path)

    print("\nTop 10 easiest stocks by average rank error:")
    print(stock_summary_df.head(10).to_string(index=False))

    print("\nTop 10 hardest stocks by average rank error:")
    print(stock_summary_df.sort_values("avg_rank_error", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

