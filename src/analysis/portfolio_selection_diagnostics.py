# src/analysis/portfolio_selection_diagnostics.py

from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np

from src.config import SELECTION_DIAGNOSTICS


def _safe_group_size(n: int, frac: float) -> int:
    return max(1, int(math.floor(n * frac)))


def compute_monthly_selection_diagnostics(
    df: pd.DataFrame,
    date_col: str,
    ticker_col: str,
    pred_col: str,
    actual_col: str,
    top_frac: float,
    bottom_frac: float,
    higher_is_better: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {date_col, ticker_col, pred_col, actual_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    work = df[[date_col, ticker_col, pred_col, actual_col]].copy()
    work = work.dropna(subset=[date_col, ticker_col, pred_col, actual_col])
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values([date_col, ticker_col]).reset_index(drop=True)

    spread_rows = []
    overlap_rows = []
    membership_rows = []

    for dt, grp in work.groupby(date_col, sort=True):
        grp = grp.copy()
        n = len(grp)
        if n < 5:
            continue

        n_top = _safe_group_size(n, top_frac)
        n_bottom = _safe_group_size(n, bottom_frac)

        if n_top + n_bottom > n:
            n_bottom = max(1, n - n_top)

        pred_sorted = grp.sort_values(pred_col, ascending=not higher_is_better).reset_index(drop=True)
        actual_sorted = grp.sort_values(actual_col, ascending=False).reset_index(drop=True)

        pred_top = pred_sorted.iloc[:n_top]
        pred_bottom = pred_sorted.iloc[-n_bottom:]
        actual_top = actual_sorted.iloc[:n_top]
        actual_bottom = actual_sorted.iloc[-n_bottom:]

        top_avg = pred_top[actual_col].mean()
        all_avg = grp[actual_col].mean()
        bottom_avg = pred_bottom[actual_col].mean()

        spread_rows.append(
            {
                "date": dt,
                "n_stocks": n,
                "n_top": n_top,
                "n_bottom": n_bottom,
                "top_avg_realized_return": top_avg,
                "all_avg_realized_return": all_avg,
                "bottom_avg_realized_return": bottom_avg,
                "top_minus_all": top_avg - all_avg,
                "all_minus_bottom": all_avg - bottom_avg,
                "top_minus_bottom": top_avg - bottom_avg,
                "top_gt_all": float(top_avg > all_avg),
                "bottom_lt_all": float(bottom_avg < all_avg),
                "top_gt_bottom": float(top_avg > bottom_avg),
            }
        )

        pred_top_set = set(pred_top[ticker_col])
        pred_bottom_set = set(pred_bottom[ticker_col])
        actual_top_set = set(actual_top[ticker_col])
        actual_bottom_set = set(actual_bottom[ticker_col])

        top_overlap_count = len(pred_top_set & actual_top_set)
        bottom_overlap_count = len(pred_bottom_set & actual_bottom_set)

        overlap_rows.append(
            {
                "date": dt,
                "n_stocks": n,
                "predicted_top_count": n_top,
                "actual_top_count": n_top,
                "predicted_bottom_count": n_bottom,
                "actual_bottom_count": n_bottom,
                "top_overlap_count": top_overlap_count,
                "bottom_overlap_count": bottom_overlap_count,
                "top_overlap_rate": top_overlap_count / n_top,
                "bottom_overlap_rate": bottom_overlap_count / n_bottom,
            }
        )

        for _, row in grp.iterrows():
            ticker = row[ticker_col]
            membership_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "in_predicted_top": int(ticker in pred_top_set),
                    "in_actual_top": int(ticker in actual_top_set),
                    "in_predicted_bottom": int(ticker in pred_bottom_set),
                    "in_actual_bottom": int(ticker in actual_bottom_set),
                    "y_pred": row[pred_col],
                    "y_true": row[actual_col],
                }
            )

    monthly_spreads = pd.DataFrame(spread_rows).sort_values("date").reset_index(drop=True)
    monthly_overlap = pd.DataFrame(overlap_rows).sort_values("date").reset_index(drop=True)
    membership_df = pd.DataFrame(membership_rows).sort_values(["date", "ticker"]).reset_index(drop=True)

    spread_summary = pd.DataFrame(
        [{
            "months_evaluated": len(monthly_spreads),
            "mean_top_avg_return": monthly_spreads["top_avg_realized_return"].mean(),
            "mean_all_avg_return": monthly_spreads["all_avg_realized_return"].mean(),
            "mean_bottom_avg_return": monthly_spreads["bottom_avg_realized_return"].mean(),
            "mean_top_minus_all": monthly_spreads["top_minus_all"].mean(),
            "mean_all_minus_bottom": monthly_spreads["all_minus_bottom"].mean(),
            "mean_top_minus_bottom": monthly_spreads["top_minus_bottom"].mean(),
            "pct_months_top_gt_all": monthly_spreads["top_gt_all"].mean(),
            "pct_months_bottom_lt_all": monthly_spreads["bottom_lt_all"].mean(),
            "pct_months_top_gt_bottom": monthly_spreads["top_gt_bottom"].mean(),
        }]
    )

    overlap_summary = pd.DataFrame(
        [{
            "months_evaluated": len(monthly_overlap),
            "mean_top_overlap_rate": monthly_overlap["top_overlap_rate"].mean(),
            "mean_bottom_overlap_rate": monthly_overlap["bottom_overlap_rate"].mean(),
        }]
    )

    return monthly_spreads, spread_summary, monthly_overlap, overlap_summary, membership_df


def main() -> None:
    cfg = SELECTION_DIAGNOSTICS

    predictions_path = Path(cfg["predictions_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if predictions_path.suffix.lower() == ".csv":
        df = pd.read_csv(predictions_path)
    elif predictions_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(predictions_path)
    else:
        raise ValueError(f"Unsupported file type: {predictions_path.suffix}")

    monthly_spreads, spread_summary, monthly_overlap, overlap_summary, membership_df = (
        compute_monthly_selection_diagnostics(
            df=df,
            date_col=cfg["date_col"],
            ticker_col=cfg["ticker_col"],
            pred_col=cfg["pred_col"],
            actual_col=cfg["actual_col"],
            top_frac=cfg["top_frac"],
            bottom_frac=cfg["bottom_frac"],
            higher_is_better=cfg["higher_is_better"],
        )
    )

    monthly_spreads.to_csv(output_dir / "monthly_selection_spreads.csv", index=False)
    spread_summary.to_csv(output_dir / "selection_spreads_summary.csv", index=False)
    monthly_overlap.to_csv(output_dir / "monthly_top_bottom_overlap.csv", index=False)
    overlap_summary.to_csv(output_dir / "top_bottom_overlap_summary.csv", index=False)
    membership_df.to_csv(output_dir / "monthly_group_membership.csv", index=False)

    print("Saved files:")
    print(output_dir / "monthly_selection_spreads.csv")
    print(output_dir / "selection_spreads_summary.csv")
    print(output_dir / "monthly_top_bottom_overlap.csv")
    print(output_dir / "top_bottom_overlap_summary.csv")
    print(output_dir / "monthly_group_membership.csv")

    if not spread_summary.empty and not overlap_summary.empty:
        s = spread_summary.iloc[0]
        o = overlap_summary.iloc[0]
        print("\n=== SELECTION DIAGNOSTICS SUMMARY ===")
        print(f"Months evaluated: {int(s['months_evaluated'])}")
        print(f"Mean top avg return:    {s['mean_top_avg_return']:.6f}")
        print(f"Mean all avg return:    {s['mean_all_avg_return']:.6f}")
        print(f"Mean bottom avg return: {s['mean_bottom_avg_return']:.6f}")
        print(f"Mean top - all:         {s['mean_top_minus_all']:.6f}")
        print(f"Mean top - bottom:      {s['mean_top_minus_bottom']:.6f}")
        print(f"% months top > all:     {s['pct_months_top_gt_all']:.2%}")
        print(f"Mean top overlap rate:  {o['mean_top_overlap_rate']:.2%}")
        print(f"Mean bottom overlap:    {o['mean_bottom_overlap_rate']:.2%}")


if __name__ == "__main__":
    main()

