from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Global plotting configuration
# ============================================================

@dataclass
class PlotStyleConfig:
    """
    Central plotting configuration used across all notebooks and scripts.
    """
    figsize: tuple[float, float] = (11, 6)
    dpi: int = 300
    title_size: int = 15
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    linewidth: float = 2.2
    grid_alpha: float = 0.45
    grid_linewidth: float = 0.6
    grid_linestyle: str = "--"
    scatter_alpha: float = 0.65
    hist_alpha: float = 0.85
    marker_size: int = 28
    use_tight_layout: bool = True


STYLE = PlotStyleConfig()


# ============================================================
# Color system
# ============================================================

MODEL_COLORS = {
    "baseline": "#595959",        # dark gray
    "ridge": "#4E79A7",           # blue
    "random_forest": "#F28E2B",   # orange
    "xgboost": "#B07AA1",         # purple
    "mlp": "#9C755F",             # brown
    "lstm": "#76B7B2",            # teal
}

AUX_COLORS = {
    "drawdown": "#E15759",        # red
    "turnover": "#76B7B2",        # teal
    "residuals": "#BAB0AC",       # light gray-brown
    "hist": "#4E79A7",
    "scatter": "#4E79A7",
    "diagonal": "#595959",
    "bar": "#4E79A7",
}


def get_model_color(model_name: str) -> str:
    """
    Resolve a standard model color.
    """
    if model_name not in MODEL_COLORS:
        raise ValueError(f"Unknown model_name: {model_name}")
    return MODEL_COLORS[model_name]


# ============================================================
# Internal helpers
# ============================================================

def _prepare_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _maybe_tight_layout(fig) -> None:
    if STYLE.use_tight_layout:
        fig.tight_layout()


def _ensure_parent_dir(save_path: str | None) -> None:
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def save_figure(fig, save_path: str | None) -> None:
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=STYLE.dpi, bbox_inches="tight")


def _apply_common_style(ax, title: str, ylabel: str, xlabel: str = "Date") -> None:
    ax.set_title(title, fontsize=STYLE.title_size, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=STYLE.label_size)
    ax.set_ylabel(ylabel, fontsize=STYLE.label_size)

    ax.grid(
        True,
        linestyle=STYLE.grid_linestyle,
        linewidth=STYLE.grid_linewidth,
        alpha=STYLE.grid_alpha,
    )
    ax.tick_params(axis="both", labelsize=STYLE.tick_size)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_date_axis_format(ax) -> None:
    xmin, xmax = ax.get_xlim()
    x0 = mdates.num2date(xmin)
    x1 = mdates.num2date(xmax)
    span_days = (x1 - x0).days

    if span_days <= 400:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif span_days <= 1100:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _finalize_plot(fig, ax, save_path: str | None, show: bool) -> None:
    _maybe_tight_layout(fig)
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Time-series plots
# ============================================================

def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    save_path: str | None = None,
    label: str = "Strategy",
    color: str | None = None,
    linestyle: str = "-",
    show: bool = True,
) -> None:
    s = _prepare_series(equity_curve)
    color = color or MODEL_COLORS["baseline"]

    fig, ax = plt.subplots(figsize=STYLE.figsize)
    ax.plot(
        s.index,
        s.values,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=STYLE.linewidth,
    )

    _apply_common_style(ax, title, "Portfolio Value")
    _apply_date_axis_format(ax)
    ax.legend(frameon=False, fontsize=STYLE.legend_size, loc="best")

    _finalize_plot(fig, ax, save_path, show)


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    save_path: str | None = None,
    label: str = "Strategy",
    color: str | None = None,
    linestyle: str = "-",
    show: bool = True,
) -> None:
    s = _prepare_series(equity_curve)
    drawdown = s / s.cummax() - 1.0
    color = color or AUX_COLORS["drawdown"]

    fig, ax = plt.subplots(figsize=STYLE.figsize)
    ax.plot(
        drawdown.index,
        drawdown.values,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=STYLE.linewidth,
    )

    _apply_common_style(ax, title, "Drawdown")
    _apply_date_axis_format(ax)
    ax.legend(frameon=False, fontsize=STYLE.legend_size, loc="best")

    _finalize_plot(fig, ax, save_path, show)


def plot_turnover(
    turnover_series: pd.Series,
    title: str = "Turnover",
    save_path: str | None = None,
    label: str = "Strategy",
    color: str | None = None,
    linestyle: str = "-",
    show: bool = True,
) -> None:
    s = _prepare_series(turnover_series)
    color = color or AUX_COLORS["turnover"]

    fig, ax = plt.subplots(figsize=STYLE.figsize)
    ax.plot(
        s.index,
        s.values,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=STYLE.linewidth,
    )

    _apply_common_style(ax, title, "Turnover")
    _apply_date_axis_format(ax)
    ax.legend(frameon=False, fontsize=STYLE.legend_size, loc="best")

    _finalize_plot(fig, ax, save_path, show)


def plot_multi_series(
    series_map: dict[str, pd.Series],
    title: str,
    ylabel: str,
    save_path: str | None = None,
    color_map: Optional[dict[str, str]] = None,
    linestyle_map: Optional[dict[str, str]] = None,
    show: bool = True,
) -> None:
    """
    Plot multiple time series on a single chart.

    series_map: {"Label": pd.Series, ...}
    """
    fig, ax = plt.subplots(figsize=STYLE.figsize)

    for label, series in series_map.items():
        s = _prepare_series(series)
        color = color_map.get(label) if color_map else None
        linestyle = linestyle_map.get(label, "-") if linestyle_map else "-"
        ax.plot(
            s.index,
            s.values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=STYLE.linewidth,
        )

    _apply_common_style(ax, title, ylabel)
    _apply_date_axis_format(ax)
    ax.legend(frameon=False, fontsize=STYLE.legend_size, loc="best")

    _finalize_plot(fig, ax, save_path, show)


# ============================================================
# Scatter / residual / histogram plots
# ============================================================

def plot_predicted_vs_actual(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: str | None = None,
    color: str | None = None,
    show: bool = True,
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    color = color or AUX_COLORS["scatter"]

    fig, ax = plt.subplots(figsize=STYLE.figsize)
    ax.scatter(
        y_true,
        y_pred,
        alpha=STYLE.scatter_alpha,
        s=STYLE.marker_size,
        color=color,
        edgecolors="none",
    )

    min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
    max_val = max(np.nanmax(y_true), np.nanmax(y_pred))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1.5,
        color=AUX_COLORS["diagonal"],
    )

    _apply_common_style(ax, title, "Predicted Return", xlabel="Actual Return")

    _finalize_plot(fig, ax, save_path, show)


def plot_residual_histogram(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    bins: int = 40,
    title: str = "Residual Distribution",
    save_path: str | None = None,
    color: str | None = None,
    show: bool = True,
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true
    color = color or AUX_COLORS["residuals"]

    fig, ax = plt.subplots(figsize=STYLE.figsize)
    ax.hist(residuals, bins=bins, alpha=STYLE.hist_alpha, color=color)
    ax.axvline(0.0, linestyle="--", linewidth=1.5, color=AUX_COLORS["diagonal"])

    _apply_common_style(ax, title, "Frequency", xlabel="Prediction Error")

    _finalize_plot(fig, ax, save_path, show)


# ============================================================
# Bar plots
# ============================================================

def plot_metric_bar(
    values: pd.Series,
    title: str,
    ylabel: str,
    save_path: str | None = None,
    color_map: Optional[dict[str, str]] = None,
    show: bool = True,
) -> None:
    """
    Plot a single metric across models.
    """
    values = values.copy()

    fig, ax = plt.subplots(figsize=STYLE.figsize)

    labels = list(values.index)
    colors = [
        color_map[label] if color_map and label in color_map else AUX_COLORS["bar"]
        for label in labels
    ]

    ax.bar(labels, values.values, color=colors)

    _apply_common_style(ax, title, ylabel, xlabel="")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    _finalize_plot(fig, ax, save_path, show)


def plot_grouped_metric_bars(
    metric_df: pd.DataFrame,
    title: str,
    ylabel: str,
    save_path: str | None = None,
    color_map: Optional[dict[str, str]] = None,
    show: bool = True,
) -> None:
    """
    metric_df:
        index   -> metric names
        columns -> model names

    Example:
                    Ridge    RF    XGB
        R2           ...
        Spearman     ...
    """
    fig, ax = plt.subplots(figsize=(max(11, 1.8 * len(metric_df.index)), 6))

    n_groups = len(metric_df.index)
    n_models = len(metric_df.columns)
    x = np.arange(n_groups)
    width = 0.8 / max(n_models, 1)

    for i, model_name in enumerate(metric_df.columns):
        offset = (i - (n_models - 1) / 2) * width
        color = color_map.get(model_name) if color_map else None
        ax.bar(
            x + offset,
            metric_df[model_name].values,
            width=width,
            label=model_name,
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_df.index)
    _apply_common_style(ax, title, ylabel, xlabel="")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=STYLE.legend_size, loc="best")

    _finalize_plot(fig, ax, save_path, show)


# ============================================================
# Feature / coefficient importance
# ============================================================

def plot_feature_importance(
    importance: pd.Series,
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: str | None = None,
    color: str | None = None,
    show: bool = True,
) -> None:
    s = importance.sort_values(ascending=False).head(top_n)
    color = color or MODEL_COLORS["random_forest"]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(s))))
    ax.barh(s.index[::-1], s.values[::-1], color=color)

    _apply_common_style(ax, title, "Importance", xlabel="")
    ax.set_xlabel("Importance", fontsize=STYLE.label_size)
    ax.set_ylabel("Feature", fontsize=STYLE.label_size)

    _finalize_plot(fig, ax, save_path, show)


def plot_coefficients(
    coef_series: pd.Series,
    title: str = "Model Coefficients",
    top_n: int = 20,
    save_path: str | None = None,
    positive_color: str = "#4E79A7",
    negative_color: str = "#E15759",
    show: bool = True,
) -> None:
    s = coef_series.copy()
    s = s.reindex(s.abs().sort_values(ascending=False).head(top_n).index)
    colors = [positive_color if v >= 0 else negative_color for v in s.values]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(s))))
    ax.barh(s.index[::-1], s.values[::-1], color=colors[::-1])

    _apply_common_style(ax, title, "Coefficient", xlabel="")
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color=AUX_COLORS["diagonal"])
    ax.set_xlabel("Coefficient", fontsize=STYLE.label_size)
    ax.set_ylabel("Feature", fontsize=STYLE.label_size)

    _finalize_plot(fig, ax, save_path, show)


# ============================================================
# Neural-network diagnostics
# ============================================================

def plot_loss_curve(
    history: dict,
    title: str = "Training Loss Curve",
    save_path: str | None = None,
    train_key: str = "loss",
    val_key: str = "val_loss",
    train_label: str = "Train Loss",
    val_label: str = "Validation Loss",
    train_color: str = "#4E79A7",
    val_color: str = "#F28E2B",
    show: bool = True,
) -> None:
    if train_key not in history:
        raise ValueError(f"'{train_key}' not found in history.")

    train_values = history[train_key]
    val_values = history.get(val_key)

    epochs = np.arange(1, len(train_values) + 1)

    fig, ax = plt.subplots(figsize=STYLE.figsize)
    ax.plot(
        epochs,
        train_values,
        label=train_label,
        color=train_color,
        linewidth=STYLE.linewidth,
    )

    if val_values is not None:
        ax.plot(
            epochs,
            val_values,
            label=val_label,
            color=val_color,
            linewidth=STYLE.linewidth,
            linestyle="--",
        )

    _apply_common_style(ax, title, "Loss", xlabel="Epoch")
    ax.legend(frameon=False, fontsize=STYLE.legend_size, loc="best")

    _finalize_plot(fig, ax, save_path, show)